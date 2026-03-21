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
import aws_context as awsc
 
 
# -------------------------------------------------
# Global cache
# -------------------------------------------------
 
_turnup_cache: Optional[pd.DataFrame] = None
 
 
# Default schema names for your current wide-format dataset
DEFAULT_TIME_COL = "Wedge_Time"
DEFAULT_GRADE_COL = "AB_Grade_ID"
DEFAULT_REEL_COL = "MBS_Current_reel_ID"
 
 
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