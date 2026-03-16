"""
process_data_tools.py
 
Utilities to query structured papermaking/process datasets stored as Parquet,
typically on S3.
 
This version uses an injectable filesystem and is adapted for wide-format
papermaking datasets such as reel / wedge-time snapshots.
"""
 
from typing import Optional, List, Tuple, Any, Dict
 
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datetime import datetime

 
FilterSpec = List[Tuple[str, str, Any]]
 
_fs = None
 
# Default schema names for your current wide-format dataset
DEFAULT_TIME_COL = "Wedge_Time"
DEFAULT_GRADE_COL = "AB_Grade_ID"
DEFAULT_REEL_COL = "MBS_Current_reel_ID"
 

def _coerce_filter_value(val):
    """
    Convert common filter values into types pyarrow can compare correctly.
    """
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
 
    if isinstance(val, str):
        # Try datetime-like strings first
        try:
            return pd.Timestamp(val).to_pydatetime()
        except Exception:
            return val
 
    return val
 
def set_filesystem(filesystem) -> None:
    global _fs
    _fs = filesystem
 
 
def _get_filesystem(filesystem=None):
    if filesystem is not None:
        return filesystem
    if _fs is not None:
        return _fs
    return None
 
 
def _normalize_path(path: str) -> str:
    if path.startswith("s3://"):
        return path.replace("s3://", "", 1)
    return path
 
 
def _build_filter(filters: Optional[FilterSpec]):
    if not filters:
        return None
 
    expr = None
 
    for col, op, val in filters:
        val = _coerce_filter_value(val)
        field = ds.field(col)
 
        if op == "==":
            part = field == val
        elif op == "!=":
            part = field != val
        elif op == ">":
            part = field > val
        elif op == ">=":
            part = field >= val
        elif op == "<":
            part = field < val
        elif op == "<=":
            part = field <= val
        elif op == "in":
            if not isinstance(val, (list, tuple, set)):
                raise ValueError(f"Operator 'in' requires list/tuple/set, got {type(val)}")
            part = field.isin(list(val))
        else:
            raise ValueError(f"Unsupported operator: {op}")
 
        expr = part if expr is None else (expr & part)
 
    return expr
 
 
def query_parquet_dataset(
    path: str,
    columns: Optional[List[str]] = None,
    filters: Optional[FilterSpec] = None,
    filesystem=None,
) -> pd.DataFrame:
    filesystem = _get_filesystem(filesystem)
    path_norm = _normalize_path(path)
    filt = _build_filter(filters)
 
    if path_norm.endswith(".parquet"):
        table = pq.read_table(
            path_norm,
            filesystem=filesystem,
            columns=columns,
            filters=filt,
        )
        return table.to_pandas()
 
    dataset = ds.dataset(path_norm, format="parquet", filesystem=filesystem)
    table = dataset.to_table(columns=columns, filter=filt)
    return table.to_pandas()
 
 
def get_wide_process_data(
    dataset_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    grade: Optional[Any] = None,
    reel_id: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    extra_filters: Optional[FilterSpec] = None,
    time_col: str = DEFAULT_TIME_COL,
    grade_col: str = DEFAULT_GRADE_COL,
    reel_col: str = DEFAULT_REEL_COL,
    filesystem=None,
) -> pd.DataFrame:
    """
    Query a wide-format papermaking dataset where each row is typically one reel,
    batch, or time snapshot.
 
    Defaults are adapted to your current schema:
    - time_col  = Wedge_Time
    - grade_col = AB_Grade_ID
    - reel_col  = MBS_Current_reel_ID
    """
    filters: FilterSpec = []
 
    if start is not None:
        filters.append((time_col, ">=", start))
    if end is not None:
        filters.append((time_col, "<=", end))
    if grade is not None:
        filters.append((grade_col, "==", grade))
    if reel_id is not None:
        filters.append((reel_col, "==", reel_id))
    if extra_filters:
        filters.extend(extra_filters)
 
    return query_parquet_dataset(
        path=dataset_path,
        columns=columns,
        filters=filters,
        filesystem=filesystem,
    )
 
 
def get_feature_snapshot(
    feature_path: str,
    timestamp: Optional[str] = None,
    grade: Optional[Any] = None,
    reel_id: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    extra_filters: Optional[FilterSpec] = None,
    time_col: str = DEFAULT_TIME_COL,
    grade_col: str = DEFAULT_GRADE_COL,
    reel_col: str = DEFAULT_REEL_COL,
    filesystem=None,
) -> pd.DataFrame:
    """
    For your current system, a 'feature snapshot' is just a filtered row/subset
    from a wide-format feature/model table.
    """
    filters: FilterSpec = []
 
    if timestamp is not None:
        filters.append((time_col, "==", timestamp))
    if grade is not None:
        filters.append((grade_col, "==", grade))
    if reel_id is not None:
        filters.append((reel_col, "==", reel_id))
    if extra_filters:
        filters.extend(extra_filters)
 
    return query_parquet_dataset(
        path=feature_path,
        columns=columns,
        filters=filters,
        filesystem=filesystem,
    )
 
 
def get_available_columns(
    dataset_path: str,
    filesystem=None,
) -> List[str]:
    """
    Return schema column names without loading the whole dataset into pandas.
    """
    filesystem = _get_filesystem(filesystem)
    path_norm = _normalize_path(dataset_path)
 
    if path_norm.endswith(".parquet"):
        pf = pq.ParquetFile(path_norm, filesystem=filesystem)
        return pf.schema.names
 
    dataset = ds.dataset(path_norm, format="parquet", filesystem=filesystem)
    return dataset.schema.names
 
 
def describe_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
    }