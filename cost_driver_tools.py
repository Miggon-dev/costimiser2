"""
cost_driver_tools.py
Wrapper for interval-based cost driver analysis using:
- process_data
- clustering
- shapley contribution
This uses existing project functions (DO NOT reimplement them).
"""
from typing import Dict, Any, Optional, List
import pandas as pd
from utility import (
    _turnup_data,
    _process_data,
    _process_data_clustered,
    _process_data_clustered_summary,
    _shapley_contrib,
    _cost_driver_plot,
    setpoint_df,
    build_shapley_text,
)
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

# -------------------------------------------------
# Component mapping
# -------------------------------------------------
COMPONENT_NAME_MAP = {
    "fibre": "Fibre_cost__€/T_",
    "steam": "Steam__€/T_",
    "electricity": "Electricity__€/T_",
    "starch": "Starch__€/T_",
}
COMPONENT_FEATURES = {
    "fibre": fibre_features,
    "steam": steam_features,
    "electricity": electricity_features,
    "starch": starch_features,
}
COMPONENT_MODELS = {
    "fibre": fibre_cost,
    "steam": steam_cost,
    "electricity": electricity_cost,
    "starch": starch_cost,
}

def normalize_cost_component(cost_component: str) -> str:
    """
    Convert assistant-facing component names into canonical internal keys.
    Accepted:
    - fibre
    - steam
    - electricity
    - starch
    """
    if cost_component is None:
        raise ValueError("cost_component is required")
    key = str(cost_component).strip().lower()
    if key not in COMPONENT_NAME_MAP:
        raise KeyError(
            f"Unknown cost component '{cost_component}'. "
            f"Supported: {list(COMPONENT_NAME_MAP.keys())}"
        )
    return key

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _normalize_range(date_range):
    """
    Convert (start, end) into plain Python date objects,
    because _process_data compares against datetime.date.
    """
    start, end = date_range
    return pd.to_datetime(start).date(), pd.to_datetime(end).date()

def extract_top_driver_variables(
    shapley_contrib: pd.DataFrame,
    top_n: int = 3,
    positive_only: bool = True,
) -> List[str]:
    """
    Extract top driver variable names from shapley contribution output.
    Assumes a dataframe with columns:
    - variable
    - contribution
    If positive_only=True, only keep variables with positive contribution
    (i.e. contributing to worsening).
    """
    if shapley_contrib is None or shapley_contrib.empty:
        return []
    df = shapley_contrib.copy()
    if "variable" not in df.columns or "contribution" not in df.columns:
        return []
    if positive_only:
        df = df[df["contribution"] > 0]
    if df.empty:
        return []
    df = df.sort_values("contribution", ascending=False)
    return df["variable"].astype(str).head(top_n).tolist()

def summarize_extreme_cluster_differences(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    variables: List[str],
) -> pd.DataFrame:
    """
    Summarize mean differences between the two extreme cluster dataframes
    for the selected variables.
    Convention:
    - df1 = baseline extreme cluster summary
    - df2 = target extreme cluster summary
    Returns columns:
    - variable
    - baseline_mean
    - target_mean
    - delta
    """
    rows = []
    if df1 is None or df2 is None or df1.empty or df2.empty:
        return pd.DataFrame(columns=["variable", "baseline_mean", "target_mean", "delta"])
    for var in variables:
        if var not in df1.columns or var not in df2.columns:
            continue
        s1 = pd.to_numeric(df1[var], errors="coerce")
        s2 = pd.to_numeric(df2[var], errors="coerce")
        if s1.notna().sum() == 0 or s2.notna().sum() == 0:
            continue
        baseline_mean = float(s1.mean())
        target_mean = float(s2.mean())
        rows.append(
            {
                "variable": var,
                "baseline_mean": baseline_mean,
                "target_mean": target_mean,
                "delta": target_mean - baseline_mean,
            }
        )
    return pd.DataFrame(rows)

# -------------------------------------------------
# Core workflow
# -------------------------------------------------
def run_cost_driver_analysis(
    target_range,
    baseline_range,
    cost_component: str,
    grade: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs the full pipeline:
    - load data
    - build process_data
    - clustering
    - summary
    - shapley contribution
    Returns a structured evidence package for:
    - cost driver text
    - recommendations
    - later scenario suggestion
    """
    component_key = normalize_cost_component(cost_component)
    component_full_name = COMPONENT_NAME_MAP[component_key]
    # 1. Load data
    turnup_data = _turnup_data(True, "", None, setpoint_df, False)
    # 2. Build process data
    target_range = _normalize_range(target_range)
    baseline_range = _normalize_range(baseline_range)
    process_data = _process_data(
        turnup_data,
        target_range,
        baseline_range,
        False,
    )
    # 3. Select features
    feature_fn = COMPONENT_FEATURES[component_key]
    features = feature_fn()
    # 4. Cluster
    process_data_clustered = _process_data_clustered(
        "",
        process_data,
        component_full_name,
        grade,
        features,
    )
    
    # 5. Check validity
    if process_data_clustered is None:
        return {
            "cost_component": component_key,
            "cost_component_full_name": component_full_name,
            "grade": grade,
            "target_range": target_range,
            "baseline_range": baseline_range,
            "process_data_clustered": None,
            "df1": pd.DataFrame(),
            "df2": pd.DataFrame(),
            "shapley_contrib": pd.DataFrame(),
            "top_driver_variables": [],
            "extreme_cluster_differences": pd.DataFrame(),
            "figure":None,
        }
    if len(process_data_clustered) == 0:
        return {
            "cost_component": component_key,
            "cost_component_full_name": component_full_name,
            "grade": grade,
            "target_range": target_range,
            "baseline_range": baseline_range,
            "process_data_clustered": process_data_clustered,
            "df1": pd.DataFrame(),
            "df2": pd.DataFrame(),
            "shapley_contrib": pd.DataFrame(),
            "top_driver_variables": [],
            "extreme_cluster_differences": pd.DataFrame(),
            "figure":None,
        }
    
    targets = process_data_clustered["target"].unique()
    if not ("current" in targets and "historic" in targets):
        return {
            "cost_component": component_key,
            "cost_component_full_name": component_full_name,
            "grade": grade,
            "target_range": target_range,
            "baseline_range": baseline_range,
            "process_data_clustered": process_data_clustered,
            "df1": pd.DataFrame(),
            "df2": pd.DataFrame(),
            "shapley_contrib": pd.DataFrame(),
            "top_driver_variables": [],
            "extreme_cluster_differences": pd.DataFrame(),
            "figure":None,
        }
    # 6. Summary (df1, df2)
    df1, df2 = _process_data_clustered_summary(
        process_data_clustered,
        component_full_name,
        grade,
        fibre_cost,
        steam_cost,
        electricity_cost,
        starch_cost,
        steam_features,
        electricity_features,
        starch_features,
        fibre_features,
    )
    if df1 is None or df2 is None:
        return {
            "cost_component": component_key,
            "cost_component_full_name": component_full_name,
            "grade": grade,
            "target_range": target_range,
            "baseline_range": baseline_range,
            "process_data_clustered": process_data_clustered,
            "df1": pd.DataFrame() if df1 is None else df1,
            "df2": pd.DataFrame() if df2 is None else df2,
            "shapley_contrib": pd.DataFrame(),
            "top_driver_variables": [],
            "extreme_cluster_differences": pd.DataFrame(),
            "figure":None,
        }
    # 7. SHAP contribution between periods
    shapley_contrib = _shapley_contrib(
        "",
        df1,
        df2,
        component_full_name,
        grade,
        fibre_cost,
        steam_cost,
        electricity_cost,
        starch_cost,
        steam_features,
        electricity_features,
        starch_features,
        fibre_features,
    )
    if shapley_contrib is None:
        shapley_contrib = pd.DataFrame()
    figure = None
    if shapley_contrib is not None and not shapley_contrib.empty:
        figure = _cost_driver_plot(shapley_contrib)
    top_driver_variables = extract_top_driver_variables(
        shapley_contrib=shapley_contrib,
        top_n=3,
        positive_only=True,
    )
    extreme_cluster_differences = summarize_extreme_cluster_differences(
        df1=df1,
        df2=df2,
        variables=top_driver_variables,
    )
    return {
        "cost_component": component_key,
        "cost_component_full_name": component_full_name,
        "grade": grade,
        "target_range": target_range,
        "baseline_range": baseline_range,
        "process_data_clustered": process_data_clustered,
        "df1": df1,
        "df2": df2,
        "shapley_contrib": shapley_contrib,
        "top_driver_variables": top_driver_variables,
        "extreme_cluster_differences": extreme_cluster_differences,
        "figure":figure,
    }

# -------------------------------------------------
# Simple summarization
# -------------------------------------------------
def summarize_contributions(shapley_contrib: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    """
    Return top positive and negative contributors.
    """
    if shapley_contrib is None or shapley_contrib.empty:
        return {"message": "No contribution data available."}
    df = shapley_contrib.copy()
    df_pos = df.sort_values("contribution", ascending=False).head(top_n)
    df_neg = df.sort_values("contribution", ascending=True).head(top_n)
    return {
        "top_increase": df_pos.to_dict(orient="records"),
        "top_decrease": df_neg.to_dict(orient="records"),
    }

def build_cost_driver_text(
    cost_driver_result: Dict[str, Any],
    lang: str = "en",
    top_frac: float = 0.20,
) -> str:
    """
    Wrapper around the existing utility text builder for Shapley contribution output.
    """
    shapley_contrib = cost_driver_result.get("shapley_contrib", pd.DataFrame())
    return build_shapley_text(
        shapley_contrib=shapley_contrib,
        top_frac=top_frac,
        lang=lang,
    )