"""
scenario_tools.py
 
What-if / scenario utilities for turnup-level papermaking data.
 
This version:
- loads turnup data internally via process_data_tools
- supports time-aware reference selection via target_range
- selects a reference turnup row
- applies one or more interventions
- simulates baseline vs scenario for a selected cost component
"""
 
from __future__ import annotations
 
from typing import Any, Dict, List, Optional, Tuple
 
import pandas as pd
 
import process_data_tools as pdt
from prediction_tools import (
    fibre_cost,
    steam_cost,
    electricity_cost,
    starch_cost,
    fibre_features,
    steam_features,
    electricity_features,
    starch_features,
)
 
 
DEFAULT_TURNUP_SCHEMA: Dict[str, str] = {
    "reel_id": "MBS_Current_reel_ID",
    "grade": "AB_Grade_ID",
    "time": "Wedge_Time",
}
 
 
SCENARIO_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "fibre": {"cost_fn": fibre_cost, "feature_fn": fibre_features},
    "fibre_cost": {"cost_fn": fibre_cost, "feature_fn": fibre_features},
    "steam": {"cost_fn": steam_cost, "feature_fn": steam_features},
    "steam_cost": {"cost_fn": steam_cost, "feature_fn": steam_features},
    "electricity": {"cost_fn": electricity_cost, "feature_fn": electricity_features},
    "electricity_cost": {"cost_fn": electricity_cost, "feature_fn": electricity_features},
    "starch": {"cost_fn": starch_cost, "feature_fn": starch_features},
    "starch_cost": {"cost_fn": starch_cost, "feature_fn": starch_features},
}
 
 
def _get_schema(schema: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = DEFAULT_TURNUP_SCHEMA.copy()
    if schema:
        out.update(schema)
    return out
 
 
def _validate_required_columns(df: pd.DataFrame, schema: Dict[str, str]) -> None:
    required = [schema["reel_id"], schema["grade"], schema["time"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataframe: {missing}")
 
 
def _prepare_turnup_dataframe(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")
 
    data = df.copy()
    time_col = schema["time"]
 
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col])
 
    if data.empty:
        raise ValueError(f"No valid rows remain after parsing time column '{time_col}'")
 
    return data
 
 
def _filter_by_target_range(
    df: pd.DataFrame,
    target_range: Optional[Tuple[Any, Any]],
    time_col: str,
) -> pd.DataFrame:
    """
    Filter dataframe to [start, end) using target_range.
    """
    if target_range is None:
        return df
 
    start, end = target_range
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
 
    out = df[(df[time_col] >= start_ts) & (df[time_col] < end_ts)].copy()
    return out
 
 
def _resolve_cost_component(cost_component: str) -> Tuple[Any, Any, str]:
    if cost_component is None:
        raise ValueError("cost_component must be provided")
 
    key = str(cost_component).strip().lower()
    if key not in SCENARIO_COMPONENTS:
        raise ValueError(
            f"Unsupported cost_component={cost_component!r}. "
            f"Supported values: {sorted(SCENARIO_COMPONENTS.keys())}"
        )
 
    info = SCENARIO_COMPONENTS[key]
    return info["cost_fn"], info["feature_fn"], key
 
 
def load_turnup_data_for_scenario(
    schema: Optional[Dict[str, str]] = None,
    target_range: Optional[Tuple[Any, Any]] = None,
) -> pd.DataFrame:
    """
    Load turnup data via process_data_tools and prepare it for scenario use.
    Optionally restrict to target_range.
    """
    schema = _get_schema(schema)
    df = pdt.load_turnup_data()
    _validate_required_columns(df, schema)
    data = _prepare_turnup_dataframe(df, schema)
 
    time_col = schema["time"]
    data = _filter_by_target_range(data, target_range=target_range, time_col=time_col)
 
    if data.empty:
        raise ValueError("No turnup rows found for the requested target_range")
 
    return data
 
 
def get_reference_turnup(
    reel_id: Optional[Any] = None,
    timestamp: Optional[Any] = None,
    grade: Optional[Any] = None,
    target_range: Optional[Tuple[Any, Any]] = None,
    schema: Optional[Dict[str, str]] = None,
    max_time_diff_minutes: float = 240.0,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Select a single historical turnup row to use as baseline.
 
    Selection priority:
    1) exact reel_id
    2) closest timestamp (optionally within grade)
    3) latest row for grade
    4) latest overall
 
    If target_range is provided, candidates are restricted to that interval first.
    """
    schema = _get_schema(schema)
 
    if df is None:
        data = load_turnup_data_for_scenario(schema=schema, target_range=target_range)
    else:
        _validate_required_columns(df, schema)
        data = _prepare_turnup_dataframe(df, schema)
        data = _filter_by_target_range(data, target_range=target_range, time_col=schema["time"])
        if data.empty:
            raise ValueError("No turnup rows found for the requested target_range")
 
    reel_col = schema["reel_id"]
    grade_col = schema["grade"]
    time_col = schema["time"]
 
    warnings: List[str] = []
 
    if target_range is not None:
        warnings.append(f"Reference selection restricted to target_range={target_range}")
 
    if reel_id is not None:
        sel = data[data[reel_col].astype(str) == str(reel_id)]
        if sel.empty:
            raise ValueError(f"Reel ID {reel_id!r} not found in column '{reel_col}'")
 
        row = sel.sort_values(time_col).iloc[-1]
        return {
            "row": row.to_frame().T.reset_index(drop=True),
            "reference": {
                "selection_mode": "reel_id",
                "requested_reel_id": reel_id,
                "requested_target_range": target_range,
                "matched_reel_id": row[reel_col],
                "matched_grade": row[grade_col],
                "matched_time": row[time_col],
            },
            "warnings": warnings,
        }
 
    if timestamp is not None:
        ts = pd.to_datetime(timestamp, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Could not parse timestamp: {timestamp!r}")
 
        sel = data
        if grade is not None:
            sel = sel[sel[grade_col].astype(str) == str(grade)]
            if sel.empty:
                raise ValueError(
                    f"No rows found for grade {grade!r} in column '{grade_col}' "
                    f"within target_range={target_range}"
                )
 
        sel = sel.copy()
        sel["_time_diff"] = (sel[time_col] - ts).abs()
 
        row = sel.sort_values("_time_diff").iloc[0]
        diff_min = row["_time_diff"].total_seconds() / 60.0
 
        if diff_min > max_time_diff_minutes:
            warnings.append(
                f"Closest turnup row is {diff_min:.1f} minutes away from requested timestamp"
            )
 
        row_out = row.drop(labels=["_time_diff"])
 
        return {
            "row": row_out.to_frame().T.reset_index(drop=True),
            "reference": {
                "selection_mode": "timestamp",
                "requested_timestamp": ts,
                "requested_grade": grade,
                "requested_target_range": target_range,
                "matched_reel_id": row[reel_col],
                "matched_grade": row[grade_col],
                "matched_time": row[time_col],
                "time_diff_minutes": diff_min,
            },
            "warnings": warnings,
        }
 
    if grade is not None:
        sel = data[data[grade_col].astype(str) == str(grade)]
        if sel.empty:
            raise ValueError(
                f"No rows found for grade {grade!r} in column '{grade_col}' "
                f"within target_range={target_range}"
            )
 
        row = sel.sort_values(time_col).iloc[-1]
        return {
            "row": row.to_frame().T.reset_index(drop=True),
            "reference": {
                "selection_mode": "grade_latest",
                "requested_grade": grade,
                "requested_target_range": target_range,
                "matched_reel_id": row[reel_col],
                "matched_grade": row[grade_col],
                "matched_time": row[time_col],
            },
            "warnings": warnings,
        }
 
    row = data.sort_values(time_col).iloc[-1]
    return {
        "row": row.to_frame().T.reset_index(drop=True),
        "reference": {
            "selection_mode": "latest",
            "requested_target_range": target_range,
            "matched_reel_id": row[reel_col],
            "matched_grade": row[grade_col],
            "matched_time": row[time_col],
        },
        "warnings": warnings,
    }
 
 
def describe_reference_turnup(reference_out: Dict[str, Any]) -> str:
    ref = reference_out.get("reference", {})
    warnings = reference_out.get("warnings", [])
 
    lines = [
        f"Selection mode: {ref.get('selection_mode')}",
        f"Matched reel: {ref.get('matched_reel_id')}",
        f"Matched grade: {ref.get('matched_grade')}",
        f"Matched time: {ref.get('matched_time')}",
    ]
 
    if ref.get("requested_reel_id") is not None:
        lines.append(f"Requested reel: {ref.get('requested_reel_id')}")
 
    if ref.get("requested_grade") is not None:
        lines.append(f"Requested grade: {ref.get('requested_grade')}")
 
    if ref.get("requested_timestamp") is not None:
        lines.append(f"Requested timestamp: {ref.get('requested_timestamp')}")
 
    if ref.get("requested_target_range") is not None:
        lines.append(f"Requested target_range: {ref.get('requested_target_range')}")
 
    if ref.get("time_diff_minutes") is not None:
        lines.append(f"Time diff (min): {ref.get('time_diff_minutes'):.1f}")
 
    if warnings:
        lines.append("Warnings: " + " | ".join(warnings))
 
    return "\n".join(lines)
 
 
def apply_interventions(
    row_df: pd.DataFrame,
    interventions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if row_df is None or row_df.empty:
        raise ValueError("row_df is empty")
 
    if len(row_df) != 1:
        raise ValueError("apply_interventions expects a single-row dataframe")
 
    out = row_df.copy()
    warnings: List[str] = []
    applied: List[Dict[str, Any]] = []
 
    for i, itv in enumerate(interventions):
        if not isinstance(itv, dict):
            raise ValueError(f"Intervention at position {i} must be a dict")
 
        variable = itv.get("variable")
        mode = itv.get("mode")
        value = itv.get("value")
 
        if variable is None:
            raise ValueError(f"Intervention at position {i} is missing 'variable'")
        if mode is None:
            raise ValueError(f"Intervention for {variable!r} is missing 'mode'")
        if value is None:
            raise ValueError(f"Intervention for {variable!r} is missing 'value'")
 
        if variable not in out.columns:
            raise ValueError(f"Variable {variable!r} not found in row_df")
 
        current_value = out.iloc[0][variable]
 
        if pd.isna(current_value):
            warnings.append(f"Variable {variable!r} is NaN in the reference row; skipped")
            continue
 
        try:
            current_numeric = float(current_value)
        except Exception as e:
            raise ValueError(
                f"Variable {variable!r} has non-numeric value {current_value!r}; "
                f"cannot apply numeric intervention"
            ) from e
 
        if mode == "absolute":
            new_value = float(value)
        elif mode == "delta":
            new_value = current_numeric + float(value)
        elif mode == "relative":
            new_value = current_numeric * (1.0 + float(value))
        else:
            raise ValueError(
                f"Unsupported mode {mode!r} for variable {variable!r}. "
                f"Use 'absolute', 'delta', or 'relative'."
            )
 
        out.at[out.index[0], variable] = new_value
 
        applied.append(
            {
                "variable": variable,
                "mode": mode,
                "value": value,
                "old_value": current_numeric,
                "new_value": new_value,
            }
        )
 
    return {
        "row": out,
        "applied_interventions": applied,
        "warnings": warnings,
    }
 
 
def simulate_turnup_scenario(
    cost_component: str,
    interventions: List[Dict[str, Any]],
    reel_id: Optional[Any] = None,
    timestamp: Optional[Any] = None,
    grade: Optional[Any] = None,
    target_range: Optional[Tuple[Any, Any]] = None,
    schema: Optional[Dict[str, str]] = None,
    max_time_diff_minutes: float = 240.0,
    row_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Simulate a scenario for a selected cost component.
 
    If row_df is not provided, the reference row is selected internally.
    If target_range is provided, the reference selection is restricted to that period.
    """
    schema = _get_schema(schema)
    cost_fn, feature_fn, resolved_component = _resolve_cost_component(cost_component)
 
    if row_df is None:
        ref = get_reference_turnup(
            reel_id=reel_id,
            timestamp=timestamp,
            grade=grade,
            target_range=target_range,
            schema=schema,
            max_time_diff_minutes=max_time_diff_minutes,
        )
        baseline_row = ref["row"]
        reference_meta = ref["reference"]
        reference_warnings = list(ref["warnings"])
    else:
        if row_df is None or row_df.empty:
            raise ValueError("row_df is empty")
        if len(row_df) != 1:
            raise ValueError("simulate_turnup_scenario expects a single-row dataframe")
        baseline_row = row_df.copy()
        reference_meta = {
            "selection_mode": "provided_row",
            "requested_target_range": target_range,
        }
        reference_warnings = []
 
    changed_vars = {itv.get("variable") for itv in interventions if itv.get("variable") is not None}
    feature_cols = set(feature_fn())
    unused_changed_vars = sorted(v for v in changed_vars if v not in feature_cols)
 
    baseline_series = baseline_row.iloc[0]
    baseline_prediction = float(cost_fn(baseline_series))
 
    mod = apply_interventions(baseline_row, interventions)
    scenario_row = mod["row"]
    scenario_series = scenario_row.iloc[0]
    scenario_prediction = float(cost_fn(scenario_series))
 
    warnings = reference_warnings + list(mod["warnings"])
 
    if changed_vars:
        if len(unused_changed_vars) == len(changed_vars):
            warnings.append(
                f"None of the modified variables are used by the {resolved_component} model: "
                f"{unused_changed_vars}"
            )
        elif unused_changed_vars:
            warnings.append(
                f"Some modified variables are not used by the {resolved_component} model: "
                f"{unused_changed_vars}"
            )
 
    return {
        "cost_component": resolved_component,
        "reference": reference_meta,
        "baseline_prediction": baseline_prediction,
        "scenario_prediction": scenario_prediction,
        "delta_prediction": scenario_prediction - baseline_prediction,
        "baseline_row": baseline_row,
        "scenario_row": scenario_row,
        "applied_interventions": mod["applied_interventions"],
        "warnings": warnings,
    }
 
 
def describe_scenario_result(sim_out: Dict[str, Any], decimals: int = 6) -> str:
    lines = [
        f"Cost component: {sim_out.get('cost_component')}",
        f"Baseline prediction: {sim_out['baseline_prediction']:.{decimals}f}",
        f"Scenario prediction: {sim_out['scenario_prediction']:.{decimals}f}",
        f"Delta prediction: {sim_out['delta_prediction']:.{decimals}f}",
    ]
 
    ref = sim_out.get("reference", {})
    if ref:
        lines.append(f"Reference mode: {ref.get('selection_mode')}")
        if ref.get("matched_reel_id") is not None:
            lines.append(f"Matched reel: {ref.get('matched_reel_id')}")
        if ref.get("matched_grade") is not None:
            lines.append(f"Matched grade: {ref.get('matched_grade')}")
        if ref.get("matched_time") is not None:
            lines.append(f"Matched time: {ref.get('matched_time')}")
        if ref.get("requested_target_range") is not None:
            lines.append(f"Target range: {ref.get('requested_target_range')}")
 
    for a in sim_out.get("applied_interventions", []):
        lines.append(
            f"Intervention - {a['variable']}: "
            f"{a['old_value']:.{decimals}f} -> {a['new_value']:.{decimals}f} "
            f"(mode={a['mode']}, value={a['value']})"
        )
 
    warnings = sim_out.get("warnings", [])
    if warnings:
        lines.append("Warnings: " + " | ".join(warnings))
 
    return "\n".join(lines)