"""
knowledge_retrieval.py
 
RAG module for papermaking documents using:
- FAISS index stored in S3
- meta.parquet stored in S3 (id -> chunk_id/source/page/chunk/text)
- AWS Bedrock Titan embeddings v2 + Claude (Converse)
- Optional status callback for progress reporting
 
Design:
- On first query, download index+metadata from S3 to local cache dir and load into memory.
- Subsequent queries reuse the in-memory index/metadata.
- To update, publish a new S3 "version" and restart pods or call reload().
"""
 
import os
import json
import time
import threading
from typing import List, Dict, Any, Optional, Callable, Tuple
 
import boto3
import pandas as pd
import numpy as np
import faiss
import aws_context as awsc
from botocore.config import Config

 
# ================================
# Status hook
# ================================
StatusFn = Optional[Callable[[str], None]]
 
def _status(cb: StatusFn, msg: str) -> None:
    if cb is not None:
        cb(msg)
 
# ================================
# Configuration
# ================================
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-west-1"
 
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
 
LLM_MODEL_ID = os.environ.get(
    "LLM_MODEL_ID"
) or "arn:aws:bedrock:eu-west-1:781690932061:inference-profile/eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
 
TOP_K = int(os.environ.get("TOP_K", "6"))
 
S3_BUCKET = os.environ.get("FAISS_S3_BUCKET", "s3-dssmith-dev-costimiser")
S3_PREFIX = os.environ.get("FAISS_S3_PREFIX", "rag/faiss/papermaking")
 
FAISS_VERSION = os.environ.get("FAISS_VERSION", "")
CACHE_DIR = os.environ.get("FAISS_CACHE_DIR", "/tmp/faiss_cache")
MANIFEST_TTL_SECONDS = int(os.environ.get("FAISS_MANIFEST_TTL_SECONDS", "60"))
FAISS_DIM = int(os.environ.get("FAISS_DIM", "1024"))
 
# ================================
# AWS clients (lazy init)
# ================================
_session: Optional[boto3.session.Session] = None
rt = None
s3 = None
_sts = None
_session = None

BEDROCK_CONFIG = Config(
    connect_timeout=10,   # time to establish connection
    read_timeout=300,     # <-- increase this (e.g. 300s = 5 min)
    retries={
        "max_attempts": 3,
        "mode": "standard"
    }
)
 
def get_bedrock_runtime():
    global _session
    return _session.client(
        "bedrock-runtime",
        region_name="eu-west-1",
        config=BEDROCK_CONFIG
    )
 
def set_boto3_session(session: boto3.session.Session) -> None:
    global _session, rt, s3, _sts, _aws_validated
    _session = session
    awsc.set_session(session)
    rt = awsc.get_bedrock_runtime()
    s3 = awsc.get_s3()
    _sts = awsc.get_sts()
    _aws_validated = False
 
def _ensure_clients() -> None:
    global _session, rt, s3, _sts
 
    if _session is None:
        raise RuntimeError("No boto3 session has been set. Call set_boto3_session(session) first.")
 
    if rt is None:
        rt = _session.client(
            "bedrock-runtime",
            region_name=REGION,
            config=BEDROCK_CONFIG,
        )
 
    if s3 is None:
        s3 = _session.client("s3", region_name=REGION)
 
    if _sts is None:
        _sts = _session.client("sts", region_name=REGION)
 
# ================================
# Validation flags
# ================================
_aws_validated = False
_embedding_dim_validated = False
 
def validate_aws_access(status_cb: StatusFn = None) -> None:
    global _aws_validated
    if _aws_validated:
        return
 
    _ensure_clients()
    try:
        arn = _sts.get_caller_identity()["Arn"]
        _status(status_cb, f"AWS identity: {arn}")
        _aws_validated = True
    except Exception as e:
        raise RuntimeError(f"AWS credentials not available in this process: {e}") from e
 
# ================================
# Bedrock – Titan embeddings v2
# ================================
def embed_text(text: str) -> List[float]:
    _ensure_clients()
    out = rt.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(out["body"].read())
    return payload["embedding"]
 
def validate_embedding_dim(status_cb: StatusFn = None) -> None:
    global _embedding_dim_validated
    if _embedding_dim_validated:
        return
 
    _status(status_cb, f"Validating embedding dimension for model '{EMBED_MODEL_ID}' …")
    v = embed_text("dimension test")
    dim = len(v)
    _status(status_cb, f"Embedding dimension = {dim}, FAISS expects = {FAISS_DIM}")
 
    if dim != FAISS_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch.\n"
            f"- Bedrock model '{EMBED_MODEL_ID}' returns dimension {dim}\n"
            f"- FAISS index expects dimension {FAISS_DIM}\n"
        )
 
    _embedding_dim_validated = True
 
# ================================
# Bedrock – Claude (Converse)
# ================================
def llm_answer(prompt: str, max_tokens: int = 5000, temperature: float = 0.2) -> str:
    _ensure_clients()
    resp = rt.converse(
        modelId=LLM_MODEL_ID,
        system=[{
            "text": (
                "You are a senior papermaking engineer assistant.\n"
                "Use the provided CONTEXT first. If context is insufficient for plant-specific guidance, say so.\n"
                "Cite sources like [file.pdf:p12:c3]."
            )
        }],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return resp["output"]["message"]["content"][0]["text"]
 
# ================================
# FAISS + metadata cache
# ================================
_state_lock = threading.Lock()
_index: Optional[faiss.Index] = None
_meta: Optional[pd.DataFrame] = None
_meta_by_id: Optional[Dict[int, Dict[str, Any]]] = None
_loaded_version: Optional[str] = None
_last_manifest_fetch: float = 0.0
_cached_manifest: Optional[Dict[str, Any]] = None
 
def _s3_read_json(bucket: str, key: str) -> Dict[str, Any]:
    _ensure_clients()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))
 
def _download_if_missing(bucket: str, key: str, local_path: str) -> None:
    _ensure_clients()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return
    s3.download_file(bucket, key, local_path)
 
def _get_active_version(status_cb: StatusFn = None) -> Tuple[str, Dict[str, Any]]:
    global _last_manifest_fetch, _cached_manifest
 
    if FAISS_VERSION:
        manifest = {"active_version": FAISS_VERSION}
        return FAISS_VERSION, manifest
 
    now = time.time()
    if _cached_manifest is None or (now - _last_manifest_fetch) > MANIFEST_TTL_SECONDS:
        _status(status_cb, f"Fetching manifest from s3://{S3_BUCKET}/{S3_PREFIX}/manifest.json …")
        _cached_manifest = _s3_read_json(S3_BUCKET, f"{S3_PREFIX}/manifest.json")
        _last_manifest_fetch = now
 
    ver = _cached_manifest.get("active_version")
    if not ver:
        raise RuntimeError(f"manifest.json missing active_version at s3://{S3_BUCKET}/{S3_PREFIX}/manifest.json")
 
    return ver, _cached_manifest
 
def _load_faiss_from_s3(status_cb: StatusFn = None, force_reload: bool = False) -> None:
    global _index, _meta, _meta_by_id, _loaded_version
 
    with _state_lock:
        ver, manifest = _get_active_version(status_cb=status_cb)
 
        if (not force_reload) and _index is not None and _meta is not None and _loaded_version == ver:
            return
 
        _status(status_cb, f"Loading FAISS snapshot version '{ver}' …")
 
        local_ver_dir = os.path.join(CACHE_DIR, ver)
        index_local = os.path.join(local_ver_dir, "index.faiss")
        meta_local = os.path.join(local_ver_dir, "meta.parquet")
 
        index_key = f"{S3_PREFIX}/versions/{ver}/index.faiss"
        meta_key = f"{S3_PREFIX}/versions/{ver}/meta.parquet"
 
        _status(status_cb, f"Downloading index/meta from S3 to {local_ver_dir} …")
        _download_if_missing(S3_BUCKET, index_key, index_local)
        _download_if_missing(S3_BUCKET, meta_key, meta_local)
 
        _status(status_cb, "Reading FAISS index …")
        idx = faiss.read_index(index_local)
 
        if idx.d != FAISS_DIM:
            raise RuntimeError(f"FAISS dim mismatch: index.d={idx.d} expected={FAISS_DIM}")
 
        _status(status_cb, "Reading metadata (parquet) …")
        meta = pd.read_parquet(meta_local)
 
        required = {"id", "chunk_id", "text"}
        missing = required - set(meta.columns)
        if missing:
            raise RuntimeError(f"meta.parquet missing required columns: {sorted(missing)}")
 
        meta["id"] = meta["id"].astype("int64")
        by_id = {int(r["id"]): r for r in meta.to_dict(orient="records")}
 
        _index = idx
        _meta = meta
        _meta_by_id = by_id
        _loaded_version = ver
 
        _status(status_cb, f"✅ Loaded FAISS ntotal={idx.ntotal} rows={len(meta)} version='{ver}'")
 
def reload(status_cb: StatusFn = None) -> None:
    _load_faiss_from_s3(status_cb=status_cb, force_reload=True)
 
# ================================
# Retrieval
# ================================
def retrieve_context(query: str, k: int = TOP_K, status_cb: StatusFn = None) -> Dict[str, Any]:
    validate_aws_access(status_cb=status_cb)
    validate_embedding_dim(status_cb=status_cb)
    _load_faiss_from_s3(status_cb=status_cb)
 
    assert _index is not None and _meta_by_id is not None
 
    _status(status_cb, "Embedding query…")
    q_emb = np.asarray([embed_text(query)], dtype="float32")
 
    if q_emb.shape[1] != FAISS_DIM:
        raise RuntimeError(f"Query embedding dim mismatch: got {q_emb.shape[1]} expected {FAISS_DIM}")
 
    _status(status_cb, f"Retrieving top {k} chunks from FAISS (version='{_loaded_version}')…")
    scores, ids = _index.search(q_emb, k)
 
    hit_ids = [int(i) for i in ids[0].tolist() if int(i) != -1]
    hit_scores = scores[0].tolist()
 
    _status(status_cb, "Building context…")
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
 
    for rank, row_id in enumerate(hit_ids):
        row = _meta_by_id.get(row_id)
        if not row:
            continue
 
        cid = row.get("chunk_id", f"id:{row_id}")
        doc_text = row.get("text", "") or ""
 
        context_blocks.append(f"SOURCE: {cid}\nCONTENT:\n{doc_text}")
 
        sources.append({
            "id": cid,
            "file": row.get("file") or row.get("source"),
            "page": row.get("page"),
            "chunk": row.get("chunk"),
            "score": float(hit_scores[rank]) if rank < len(hit_scores) else None,
            "row_id": row_id,
        })
 
    context = "\n\n---\n\n".join(context_blocks)
 
    return {
        "context": context,
        "sources": sources,
        "version": _loaded_version,
    }
 
# ================================
# Retrieval + answer generation
# ================================
def ask(query: str, k: int = TOP_K, status_cb: StatusFn = None) -> Dict[str, Any]:
    retrieved = retrieve_context(query=query, k=k, status_cb=status_cb)
 
    prompt = f"""You are a senior papermaking engineer assistant.
Use the provided CONTEXT first. If context is insufficient for plant-specific guidance, say so.
Cite sources like [file.pdf:p12:c3].
 
QUESTION: {query}
 
CONTEXT:
{retrieved['context']}
"""
 
    _status(status_cb, "Calling LLM…")
    answer = llm_answer(prompt)
 
    _status(status_cb, "Done.")
    return {
        "answer": answer,
        "sources": retrieved["sources"],
        "version": retrieved["version"],
    }
 
def get_chunk(chunk_id: str, status_cb: StatusFn = None) -> Dict[str, Any]:
    _load_faiss_from_s3(status_cb=status_cb)
    assert _meta is not None
 
    m = _meta[_meta["chunk_id"] == chunk_id]
    if m.empty:
        raise KeyError(f"Chunk id not found: {chunk_id}")
 
    row = m.iloc[0].to_dict()
    return {
        "id": row.get("chunk_id", chunk_id),
        "document": row.get("text", ""),
        "metadata": {k: v for k, v in row.items() if k not in ("text",)},
    }