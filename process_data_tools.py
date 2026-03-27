"""
process_data_tools.py
 
Utilities to access engineered papermaking/process data.
 
This version uses the existing project function:
    utility._turnup_data(True, "", None, setpoint_df, False)
 
and keeps the resulting wide engineered dataframe in memory.
"""
 
from typing import Optional, List, Dict, Any
 
import pandas as pd
 
from utility import _turnup_data, setpoint_df
from utility import (
    _process_features,
    _cost_features,
    _quality_features,
    _component_features,
)

import aws_context as awsc
from typing import List, Optional
import difflib
import ast
import re

 
# -------------------------------------------------
# Global cache
# -------------------------------------------------
 
_turnup_cache: Optional[pd.DataFrame] = None
 
 
# Default schema names for your current wide-format dataset
DEFAULT_TIME_COL = "Wedge_Time"
DEFAULT_GRADE_COL = "AB_Grade_ID"
DEFAULT_REEL_COL = "MBS_Current_reel_ID"

STOPWORDS = {
    "show", "plot", "display", "give", "me", "the", "for", "of", "and", "or",
    "in", "on", "at", "last", "week", "month", "day", "grade", "data",
    "variable", "variables", "please"
}

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("_", " ")
    s = re.sub(r"[^a-z0-9%€/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_feature_request_phrase(query: str) -> str:
    import re

    q = query.lower()

    # command words
    q = re.sub(r"\b(show|plot|display|give me|i want|please)\b", " ", q)

    # grade references
    q = re.sub(r"\bfor grade \d+\b", " ", q)
    q = re.sub(r"\bgrade \d+\b", " ", q)

    # time phrases
    q = re.sub(r"\blast week\b", " ", q)
    q = re.sub(r"\bthis week\b", " ", q)
    q = re.sub(r"\bweek \d+\b", " ", q)
    q = re.sub(r"\blast month\b", " ", q)
    q = re.sub(r"\bthis month\b", " ", q)
    q = re.sub(r"\bin [a-z]+ 20\d{2}\b", " ", q)

    # plotting instructions
    q = re.sub(r"\bin secondary axis\b", " ", q)
    q = re.sub(r"\bon secondary axis\b", " ", q)
    q = re.sub(r"\bin second axis\b", " ", q)
    q = re.sub(r"\bon second axis\b", " ", q)
    q = re.sub(r"\bin secondary y\b", " ", q)
    q = re.sub(r"\bon secondary y\b", " ", q)

    q = re.sub(r"\bwithout [a-z0-9_ ,and]+\b", " ", q)
    q = re.sub(r"\bexcluding [a-z0-9_ ,and]+\b", " ", q)
    q = re.sub(r"\bexclude [a-z0-9_ ,and]+\b", " ", q)
    q = re.sub(r"\bnot [a-z0-9_ ,and]+\b", " ", q)

    q = re.sub(r"\s+", " ", q).strip()
    return q


def _tokenize(text: str) -> List[str]:
    return [w for w in _normalize_text(text).split() if w and w not in STOPWORDS]


def _split_requested_phrases(query: str) -> List[str]:
    q = _normalize_text(query)

    # split on comma OR "and"
    parts = re.split(r",|\band\b", q)

    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [q]


def build_column_index(df_columns: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for col in df_columns:
        norm = _normalize_text(col)
        tokens = norm.split()
        rows.append(
            {
                "column": col,
                "normalized": norm,
                "tokens": tokens,
            }
        )
    return rows


def _strict_match_phrase(phrase: str, column_index: List[Dict[str, Any]]) -> List[str]:
    """
    All tokens must appear.
    Best for multi-word phrases like:
    - starch uptake
    - basis weight
    - steam pressure
    """
    tokens = _tokenize(phrase)
    if not tokens:
        return []

    out = []
    for row in column_index:
        if all(tok in row["tokens"] or tok in row["normalized"] for tok in tokens):
            out.append(row["column"])
    return out


def _broad_match_phrase(phrase: str, column_index: List[Dict[str, Any]]) -> List[str]:
    """
    Any token may appear.
    Best for single-word phrases like:
    - starch
    - steam
    - electricity
    """
    tokens = _tokenize(phrase)
    if not tokens:
        return []

    scored = []
    for row in column_index:
        hits = sum(tok in row["tokens"] or tok in row["normalized"] for tok in tokens)
        if hits > 0:
            scored.append((hits, row["column"]))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [col for _, col in scored]


def _safe_parse_list(text: str) -> List[str]:
    if not text:
        return []

    text = text.strip()

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass

    items = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    out = []
    for a, b in items:
        if a:
            out.append(a)
        elif b:
            out.append(b)
    return out


def _fuzzy_match_columns(candidates: List[str], df_columns: List[str], cutoff: float = 0.6) -> List[str]:
    matched = []
    for cand in candidates:
        hits = difflib.get_close_matches(cand, df_columns, n=1, cutoff=cutoff)
        if hits:
            matched.append(hits[0])

    out = []
    seen = set()
    for c in matched:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _llm_rerank_candidates(query: str, candidate_columns: List[str], max_return: int = 5) -> List[str]:
    if not candidate_columns:
        return []

    import knowledge_retrieval as rag

    prompt = f"""
You are selecting dataframe columns for a papermaking analytics query.

User query:
{query}

Candidate columns:
{candidate_columns}

Rules:
- Return ONLY a Python list.
- Choose the most relevant columns from the candidate list only.
- Do not invent names.
- Return at most {max_return} columns.
- If the user asked for a specific variable, prefer the narrowest exact match.
- If the user asked for a broad category, multiple columns are acceptable.
"""

    raw = rag.ask(prompt)
    if isinstance(raw, dict):
        raw = raw.get("answer", "") or raw.get("text", "") or str(raw)
    else:
        raw = str(raw)

    proposed = _safe_parse_list(raw)
    if not proposed:
        return []

    exact = [c for c in proposed if c in candidate_columns]
    if exact:
        return exact[:max_return]

    return _fuzzy_match_columns(proposed, candidate_columns)[:max_return]



def select_features_from_query(
    query: str,
    df_columns: List[str],
    use_llm_fallback: bool = False,
    max_candidates_for_llm: int = 12,
    max_return: int = 5,
) -> List[str]:
    q = query.lower()

    # High-level grouped requests
    if "cost" in q and not any(k in q for k in ["steam", "electricity", "starch", "fibre", "fiber"]):
        cols = [c for c in _cost_features() if c in df_columns]
        if cols:
            return cols

    if any(k in q for k in ["quality", "strength"]) and not any(k in q for k in ["sct", "burst", "cmt"]):
        cols = [c for c in _quality_features() if c in df_columns]
        if cols:
            return cols

    if "consumption" in q:
        cols = [c for c in _component_features() if c in df_columns]
        if cols:
            return cols

    column_index = build_column_index(df_columns)
    phrases = _split_requested_phrases(query)

    selected: List[str] = []
    llm_candidate_pool: List[str] = []

    for phrase in phrases:
        tokens = _tokenize(phrase)

        strict_matches = _strict_match_phrase(phrase, column_index)
        if strict_matches:
            selected.extend(strict_matches)
            continue

        # only use broad matching for single-word phrases
        if len(tokens) == 1:
            broad_matches = _broad_match_phrase(phrase, column_index)
            llm_candidate_pool.extend(broad_matches[:max_candidates_for_llm])

    # deduplicate strict matches first
    out = []
    seen = set()
    for c in selected:
        if c not in seen:
            out.append(c)
            seen.add(c)

    if out:
        return out[:max_return]

    # LLM reranks only a shortlist
    pool = []
    seen_pool = set()
    for c in llm_candidate_pool:
        if c not in seen_pool:
            pool.append(c)
            seen_pool.add(c)

    if use_llm_fallback and pool:
        llm_cols = _llm_rerank_candidates(
            query=query,
            candidate_columns=pool[:max_candidates_for_llm],
            max_return=max_return,
        )
        if llm_cols:
            return llm_cols

    if pool:
        return pool[:max_return]

    return [c for c in _cost_features() if c in df_columns][:max_return]
 
 
# -------------------------------------------------
# Loading / caching
# -------------------------------------------------
 
def load_turnup_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Load the engineered turnup dataframe once and cache it in memory.
    """
    global _turnup_cache
 
    if _turnup_cache is None or force_reload:
        df = _turnup_data(True, "", None, setpoint_df, False)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("_turnup_data(...) did not return a pandas DataFrame")
 
        df = df.copy()
 
        if DEFAULT_TIME_COL in df.columns:
            df[DEFAULT_TIME_COL] = pd.to_datetime(df[DEFAULT_TIME_COL], errors="coerce")
 
        _turnup_cache = df
 
    return _turnup_cache.copy()
 
 
def reload_turnup_data() -> pd.DataFrame:
    return load_turnup_data(force_reload=True)
 
 
# -------------------------------------------------
# Helpers
# -------------------------------------------------
 
def get_available_columns() -> List[str]:
    df = load_turnup_data()
    return list(df.columns)
 
 
def describe_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
    }
 
 
# -------------------------------------------------
# Filtering
# -------------------------------------------------
 
def get_wide_process_data(
    start: Optional[str] = None,
    end: Optional[str] = None,
    grade: Optional[Any] = None,
    reel_id: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    time_col: str = DEFAULT_TIME_COL,
    grade_col: str = DEFAULT_GRADE_COL,
    reel_col: str = DEFAULT_REEL_COL,
) -> pd.DataFrame:
    """
    Filter the engineered wide-format dataframe.
    """
    df = load_turnup_data()
 
    mask = pd.Series(True, index=df.index)
 
    if start is not None:
        start_ts = pd.to_datetime(start)
        mask &= df[time_col] >= start_ts
 
    if end is not None:
        end_ts = pd.to_datetime(end)
        mask &= df[time_col] <= end_ts
 
    if grade is not None:
        mask &= df[grade_col] == grade
 
    if reel_id is not None:
        mask &= df[reel_col] == reel_id
 
    out = df.loc[mask].copy()
 
    if columns is not None:
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise KeyError(f"Requested columns not found: {missing}")
        out = out[columns].copy()
 
    return out
 
 
def get_feature_snapshot(
    timestamp: Optional[str] = None,
    grade: Optional[Any] = None,
    reel_id: Optional[Any] = None,
    target_range=None,
    columns: Optional[List[str]] = None,
    time_col: str = DEFAULT_TIME_COL,
    grade_col: str = DEFAULT_GRADE_COL,
    reel_col: str = DEFAULT_REEL_COL,
) -> pd.DataFrame:
    """
    For the current system, a feature snapshot is a filtered subset
    of the engineered wide dataframe.
 
    If target_range is provided, filter rows to:
        start <= time_col < end
    """
    df = load_turnup_data()
 
    mask = pd.Series(True, index=df.index)
 
    if timestamp is not None:
        ts = pd.to_datetime(timestamp)
        mask &= df[time_col] == ts
 
    if target_range is not None:
        start, end = target_range
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        mask &= df[time_col] >= start
        mask &= df[time_col] < end
 
    if grade is not None:
        mask &= df[grade_col] == grade
 
    if reel_id is not None:
        mask &= df[reel_col] == reel_id
 
    out = df.loc[mask].copy()
 
    if columns is not None:
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise KeyError(f"Requested columns not found: {missing}")
        out = out[columns].copy()
 
    return out




def build_process_plot(
    df: pd.DataFrame,
    columns: List[str],
    time_col: str,
    secondary_axis: bool = False,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df.empty or not columns:
        return None

    df_plot = df[[time_col] + columns].copy().sort_values(time_col)

    if not secondary_axis or len(columns) < 2:
        fig = go.Figure()
        for col in columns:
            fig.add_trace(
                go.Scatter(
                    x=df_plot[time_col],
                    y=df_plot[col],
                    mode="markers",
                    name=col,
                )
            )
        fig.update_layout(title="Process Data Overview")
        return fig

    # first column primary, second column secondary
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=df_plot[time_col],
            y=df_plot[columns[0]],
            mode="markers",
            name=columns[0],
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_plot[time_col],
            y=df_plot[columns[1]],
            mode="markers",
            name=columns[1],
        ),
        secondary_y=True,
    )

    # any additional columns go to primary by default
    for col in columns[2:]:
        fig.add_trace(
            go.Scatter(
                x=df_plot[time_col],
                y=df_plot[col],
                mode="markers",
                name=col,
            ),
            secondary_y=False,
        )

    fig.update_layout(title="Process Data Overview")
    return fig

def parse_plot_preferences(query: str) -> dict:
    q = query.lower()

    return {
        "secondary_axis": "secondary axis" in q or "second axis" in q,
    }

def extract_negative_terms(query: str) -> List[str]:
    import re

    q = query.lower()

    patterns = [
        r"without ([a-z0-9_ ]+)",
        r"excluding ([a-z0-9_ ]+)",
        r"exclude ([a-z0-9_ ]+)",
        r"not ([a-z0-9_ ]+)",
    ]

    terms = []

    for pattern in patterns:
        matches = re.findall(pattern, q)
        for m in matches:
            # split by "and" or comma inside the phrase
            parts = re.split(r",|\band\b", m)
            for p in parts:
                p = p.strip()
                if p:
                    terms.append(p)

    return terms

def filter_columns_by_negative_terms(
    columns: List[str],
    negative_terms: List[str],
) -> List[str]:
    if not negative_terms:
        return columns

    neg_tokens = []
    for term in negative_terms:
        neg_tokens.extend(_tokenize(term))

    out = []
    for col in columns:
        col_l = col.lower()

        if any(tok in col_l for tok in neg_tokens):
            continue

        out.append(col)

    return out