"""
joint_distribution_tools.py

Grade-specific joint distribution utilities for feasibility-aware scenario analysis.

Design goals
------------
- Load process data through process_data_tools
- Fit the joint distribution on ALL available historical data for a grade
- Score whether a row is plausible under the fitted joint model
- Compute conditional feasible bounds for one variable given the others
- Calibrate interventions so scenario simulation stays inside realistic operating space


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Backend import helper
# ---------------------------------------------------------------------
def _get_joint_backend():
    """
    Import the joint-distribution backend from your existing codebase.

    Update the import below to match where your current implementation lives.
    """
    # EXAMPLE:
    # from copula_tools import GaussianCopulaContinuous, conditional_bounds_from_joint

    # EDIT THIS IMPORT TO MATCH YOUR PROJECT:
    from utility import GaussianCopulaContinuous, conditional_bounds_from_joint

    return GaussianCopulaContinuous, conditional_bounds_from_joint


# ---------------------------------------------------------------------
# Simple in-memory cache
# ---------------------------------------------------------------------
_JOINT_MODEL_CACHE: Dict[Tuple[str, Tuple[str, ...]], "JointModelBundle"] = {}


# ---------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------
@dataclass
class JointModelBundle:
    grade: str
    variables: List[str]
    joint: Any
    data_used: pd.DataFrame
    n_rows: int


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_joint_process_data(
    grade: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load process data from process_data_tools.

    Important:
    - Uses ALL available historical data
    - Does NOT restrict to target_range
    """
    import process_data_tools as pdt

    df = pdt.get_feature_snapshot(
        grade=grade,
        target_range=None,
    )

    if df is None or df.empty:
        raise ValueError("No process data available for joint distribution fitting.")

    return df


def _prepare_joint_data(
    process_data: pd.DataFrame,
    grade: Optional[str],
    variables: List[str],
    grade_col: str = "AB_Grade_ID",
) -> pd.DataFrame:
    """
    Prepare a clean numeric dataframe for fitting the joint model.
    """
    df = process_data.copy()

    if grade is not None and grade_col in df.columns:
        df = df[df[grade_col].astype(str) == str(grade)].copy()

    cols = [c for c in variables if c in df.columns]
    if not cols:
        raise ValueError(
            f"None of the requested variables were found for grade={grade}. "
            f"Requested: {variables}"
        )

    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

    if df.empty:
        raise ValueError(
            f"No valid rows available for grade={grade} after numeric cleaning."
        )

    return df


# ---------------------------------------------------------------------
# Fit / cache / load
# ---------------------------------------------------------------------
def fit_joint_model_for_grade(
    process_data: pd.DataFrame,
    grade: str,
    variables: List[str],
    grade_col: str = "AB_Grade_ID",
    min_rows: int = 200,
) -> JointModelBundle:
    """
    Fit a grade-specific joint distribution model.

    Parameters
    ----------
    process_data : pd.DataFrame
        Historical process data.
    grade : str
        Grade identifier.
    variables : list[str]
        Variables to include in the joint model.
    grade_col : str
        Grade column name.
    min_rows : int
        Minimum number of rows required.

    Returns
    -------
    JointModelBundle
    """
    GaussianCopulaContinuous, _ = _get_joint_backend()

    df_fit = _prepare_joint_data(
        process_data=process_data,
        grade=grade,
        variables=variables,
        grade_col=grade_col,
    )

    if len(df_fit) < min_rows:
        raise ValueError(
            f"Too few rows to fit joint model for grade {grade}: "
            f"{len(df_fit)} < {min_rows}"
        )

    joint = GaussianCopulaContinuous()
    joint.fit(df_fit)

    return JointModelBundle(
        grade=str(grade),
        variables=list(df_fit.columns),
        joint=joint,
        data_used=df_fit,
        n_rows=len(df_fit),
    )


def fit_joint_model_for_grade_from_tools(
    grade: str,
    variables: List[str],
    grade_col: str = "AB_Grade_ID",
    min_rows: int = 200,
    use_cache: bool = True,
) -> JointModelBundle:
    """
    Convenience wrapper:
    - loads all historical data through process_data_tools
    - fits the grade-specific joint model
    - optionally caches it
    """
    cache_key = (str(grade), tuple(sorted(variables)))

    if use_cache and cache_key in _JOINT_MODEL_CACHE:
        return _JOINT_MODEL_CACHE[cache_key]

    df = load_joint_process_data(grade=grade)

    bundle = fit_joint_model_for_grade(
        process_data=df,
        grade=grade,
        variables=variables,
        grade_col=grade_col,
        min_rows=min_rows,
    )

    if use_cache:
        _JOINT_MODEL_CACHE[cache_key] = bundle

    return bundle


def clear_joint_model_cache():
    _JOINT_MODEL_CACHE.clear()


# ---------------------------------------------------------------------
# Feasibility scoring
# ---------------------------------------------------------------------
def score_row_feasibility(
    row: pd.Series,
    joint_bundle: JointModelBundle,
) -> Dict[str, Any]:
    """
    Score the plausibility of one row under the fitted joint model.

    Returns
    -------
    dict with:
        grade
        variables_used
        loglik
        is_valid
    """
    x = pd.DataFrame([{v: row[v] for v in joint_bundle.variables}])
    x = x.apply(pd.to_numeric, errors="coerce")

    if x.isna().any(axis=None):
        raise ValueError(
            "Row contains NaNs in variables required by the joint model."
        )

    loglik = float(joint_bundle.joint.score_samples(x)[0])

    return {
        "grade": joint_bundle.grade,
        "variables_used": joint_bundle.variables,
        "loglik": loglik,
        "is_valid": bool(np.isfinite(loglik)),
    }


# ---------------------------------------------------------------------
# Conditional bounds
# ---------------------------------------------------------------------
def get_conditional_bounds_for_variable(
    row: pd.Series,
    variable: str,
    joint_bundle: JointModelBundle,
    q_low: float = 0.10,
    q_high: float = 0.90,
    grid_step: float = 0.01,
    safety_pad: float = 0.05,
    control_min: float | None = None,
    control_max: float | None = None,
    nonneg_control: bool = False,
    pre_expand: float = 0.0,
    post_expand: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute feasible conditional bounds for one variable given the others.

    Returns
    -------
    dict with:
        grade
        variable
        current_value
        lower_bound
        upper_bound
        median (approximate, from profile if available)
        profile_df
    """
    _, conditional_bounds_from_joint = _get_joint_backend()

    if variable not in joint_bundle.variables:
        raise ValueError(f"{variable!r} not found in joint model variables.")

    fixed_vars = [v for v in joint_bundle.variables if v != variable]
    df_fixed = pd.DataFrame([{v: row[v] for v in fixed_vars}])
    df_fixed = df_fixed.apply(pd.to_numeric, errors="coerce")

    if df_fixed.isna().any(axis=None):
        raise ValueError(
            f"Fixed conditioning row contains NaNs for variable {variable!r}."
        )

    lo, hi, profile_df = conditional_bounds_from_joint(
        free_variable=variable,
        df_hist=joint_bundle.data_used,
        joint=joint_bundle.joint,
        df_fixed=df_fixed,
        q_bounds=(q_low, q_high),
        grid_step=grid_step,
        safety_pad=safety_pad,
        control_min=control_min,
        control_max=control_max,
        nonneg_control=nonneg_control,
        pre_expand=pre_expand,
        post_expand=post_expand,
    )

    median = np.nan
    if profile_df is not None and not profile_df.empty:
        # try a few likely column names
        x_col = None
        y_col = None

        for c in profile_df.columns:
            cl = str(c).lower()
            if x_col is None and (cl in {"x", "value", variable.lower()} or "grid" in cl):
                x_col = c
            if y_col is None and ("pdf" in cl or "density" in cl or "likelihood" in cl):
                y_col = c

        # fallback: first numeric columns
        if x_col is None or y_col is None:
            num_cols = profile_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                x_col = x_col or num_cols[0]
                y_col = y_col or num_cols[1]

        if x_col is not None and y_col is not None:
            try:
                median = float(
                    profile_df.loc[profile_df[y_col].astype(float).idxmax(), x_col]
                )
            except Exception:
                median = np.nan

    return {
        "grade": joint_bundle.grade,
        "variable": variable,
        "current_value": float(row[variable]),
        "lower_bound": float(lo),
        "upper_bound": float(hi),
        "median": None if pd.isna(median) else float(median),
        "profile_df": profile_df,
    }


# ---------------------------------------------------------------------
# Intervention calibration
# ---------------------------------------------------------------------
def calibrate_intervention_to_joint_bounds(
    row: pd.Series,
    intervention: Dict[str, Any],
    joint_bundle: JointModelBundle,
    q_low: float = 0.10,
    q_high: float = 0.90,
    grid_step: float = 0.01,
    safety_pad: float = 0.05,
    control_min: float | None = None,
    control_max: float | None = None,
    nonneg_control: bool = False,
    pre_expand: float = 0.0,
    post_expand: float = 0.0,
) -> Dict[str, Any]:
    """
    Convert a requested intervention into a feasible intervention by clipping
    to the conditional bounds implied by the grade-specific joint model.

    Supported modes
    ---------------
    - relative
    - delta
    - absolute
    - review (passed through unchanged)
    """
    variable = intervention.get("variable")
    mode = intervention.get("mode")
    value = intervention.get("value")

    if variable is None or mode is None or value is None:
        raise ValueError(f"Invalid intervention: {intervention}")

    if variable not in joint_bundle.variables:
        return {
            "original_intervention": intervention,
            "calibrated_intervention": intervention,
            "was_clipped": False,
            "note": f"{variable!r} not present in joint model variables.",
        }

    current_value = float(row[variable])

    if mode == "relative":
        requested_value = current_value * (1.0 + float(value))
    elif mode == "delta":
        requested_value = current_value + float(value)
    elif mode == "absolute":
        requested_value = float(value)
    elif mode == "review":
        return {
            "original_intervention": intervention,
            "calibrated_intervention": intervention,
            "was_clipped": False,
            "note": "Review intervention; no numeric calibration applied.",
        }
    else:
        raise ValueError(f"Unsupported intervention mode: {mode!r}")

    bounds = get_conditional_bounds_for_variable(
        row=row,
        variable=variable,
        joint_bundle=joint_bundle,
        q_low=q_low,
        q_high=q_high,
        grid_step=grid_step,
        safety_pad=safety_pad,
        control_min=control_min,
        control_max=control_max,
        nonneg_control=nonneg_control,
        pre_expand=pre_expand,
        post_expand=post_expand,
    )

    lo = bounds["lower_bound"]
    hi = bounds["upper_bound"]

    calibrated_value = min(max(requested_value, lo), hi)
    was_clipped = abs(calibrated_value - requested_value) > 1e-12

    calibrated_intervention = {
        "variable": variable,
        "mode": "absolute",
        "value": calibrated_value,
        "note": (
            f"Requested value {requested_value:.6f} clipped to feasible "
            f"conditional range [{lo:.6f}, {hi:.6f}]"
            if was_clipped
            else "Requested value already within feasible conditional range"
        ),
    }

    return {
        "original_intervention": intervention,
        "calibrated_intervention": calibrated_intervention,
        "current_value": current_value,
        "requested_value": requested_value,
        "feasible_lower": lo,
        "feasible_upper": hi,
        "was_clipped": was_clipped,
    }


# ---------------------------------------------------------------------
# Convenience: calibrate several interventions
# ---------------------------------------------------------------------
def calibrate_interventions_for_row(
    row: pd.Series,
    interventions: List[Dict[str, Any]],
    joint_bundle: JointModelBundle,
    q_low: float = 0.10,
    q_high: float = 0.90,
    grid_step: float = 0.01,
    safety_pad: float = 0.05,
    control_min: float | None = None,
    control_max: float | None = None,
    nonneg_control: bool = False,
    pre_expand: float = 0.0,
    post_expand: float = 0.0,
    sequential: bool = True,
) -> Dict[str, Any]:
    """
    Calibrate a list of interventions.

    If sequential=True, each calibrated intervention is applied to the working row
    before calibrating the next one.
    """
    working_row = row.copy()
    results = []

    for itv in interventions:
        cal = calibrate_intervention_to_joint_bounds(
            row=working_row,
            intervention=itv,
            joint_bundle=joint_bundle,
            q_low=q_low,
            q_high=q_high,
            grid_step=grid_step,
            safety_pad=safety_pad,
            control_min=control_min,
            control_max=control_max,
            nonneg_control=nonneg_control,
            pre_expand=pre_expand,
            post_expand=post_expand,
        )
        results.append(cal)

        if sequential:
            citv = cal["calibrated_intervention"]
            variable = citv.get("variable")
            mode = citv.get("mode")
            value = citv.get("value")

            if variable is None or mode is None:
                continue

            if mode == "absolute":
                working_row[variable] = value
            elif mode == "delta":
                working_row[variable] = float(working_row[variable]) + float(value)
            elif mode == "relative":
                working_row[variable] = float(working_row[variable]) * (1.0 + float(value))
            elif mode == "review":
                pass

    return {
        "grade": joint_bundle.grade,
        "n_interventions": len(interventions),
        "sequential": sequential,
        "results": results,
        "final_row": working_row,
    }

def calibrate_interventions_for_row_TOREMOVE(
    row: pd.Series,
    interventions: List[Dict[str, Any]],
    joint_bundle: JointModelBundle,
    q_low: float = 0.10,
    q_high: float = 0.90,
    grid_step: int = 0.01,
) -> Dict[str, Any]:
    """
    Calibrate a list of interventions independently against the conditional
    bounds of the current row.

    Note:
    This is a simple first version. It does NOT iteratively update the row
    after each calibrated intervention. That can be added later if needed.
    """
    out = []
    for itv in interventions:
        out.append(
            calibrate_intervention_to_joint_bounds(
                row=row,
                intervention=itv,
                joint_bundle=joint_bundle,
                q_low=q_low,
                q_high=q_high,
                grid_step=grid_step,
            )
        )

    return {
        "grade": joint_bundle.grade,
        "n_interventions": len(interventions),
        "results": out,
    }


def build_joint_variable_set(
    interventions: List[Dict[str, Any]],
    baseline_row: pd.DataFrame,
    feature_fn=None,
    top_driver_variables: Optional[List[str]] = None,
    shap_result: Optional[Dict[str, Any]] = None,
    max_vars: int = 10,
) -> List[str]:
    """
    Build a compact variable set for the joint model.

    Priority:
    1. intervention variables
    2. top driver variables
    3. top SHAP variables
    4. optionally model feature variables if provided
    """
    vars_out: List[str] = []

    def add_var(v):
        if v is None:
            return
        v = str(v)
        if v not in vars_out and v in baseline_row.columns:
            vars_out.append(v)

    # 1) intervention vars
    for itv in interventions:
        add_var(itv.get("variable"))

    # 2) top driver vars
    for v in top_driver_variables or []:
        add_var(v)

    # 3) top SHAP vars
    if shap_result is not None and isinstance(shap_result, dict):
        shap_df = shap_result.get("data_frame")
        if shap_df is not None and not shap_df.empty:
            tmp = (
                shap_df.assign(abs_shap=lambda x: x["shap_value"].abs())
                .groupby("feature", as_index=False)
                .agg(mean_abs_shap=("abs_shap", "mean"))
                .sort_values("mean_abs_shap", ascending=False)
            )
            for v in tmp["feature"].tolist():
                add_var(v)

    # 4) optional model feature variables
    if feature_fn is not None:
        try:
            for v in feature_fn():
                add_var(v)
        except Exception:
            pass

    return vars_out[:max_vars]