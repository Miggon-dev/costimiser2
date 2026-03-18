"""
cost_driver_tools.py
 
Wrapper for interval-based cost driver analysis using:
- process_data
- clustering
- shapley contribution
 
This uses existing project functions (DO NOT reimplement them).
"""
 
from typing import Dict, Any, Optional
 
import pandas as pd


from utility import (
    _turnup_data,
    _process_data,
    _process_data_clustered,
    _process_data_clustered_summary,
    _shapley_contrib,
    setpoint_df,
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
# Core workflow
# -------------------------------------------------

def _normalize_range(date_range):
    """
    Convert (start, end) into plain Python date objects,
    because _process_data compares against datetime.date.
    """
    start, end = date_range
    return pd.to_datetime(start).date(), pd.to_datetime(end).date()

 
def run_cost_driver_analysis(
    target_range,
    baseline_range,
    cost_component: str,
    grade: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs the full pipeline:
    - load data
    - build process_data
    - clustering
    - summary
    - shapley contribution
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
    if len(process_data_clustered) == 0:
        return pd.DataFrame()
 
    targets = process_data_clustered["target"].unique()

    print(targets)
 
    if not ("current" in targets and "historic" in targets):
        return pd.DataFrame()
 
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
 
    return shapley_contrib
 
 
# -------------------------------------------------
# Simple summarization
# -------------------------------------------------
 
def summarize_contributions(df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    """
    Return top positive and negative contributors.
    """
    if df.empty:
        return {"message": "No contribution data available."}
 
    df = df.copy()
 
    df_pos = df.sort_values("contribution", ascending=False).head(top_n)
    df_neg = df.sort_values("contribution", ascending=True).head(top_n)
 
    return {
        "top_increase": df_pos.to_dict(orient="records"),
        "top_decrease": df_neg.to_dict(orient="records"),
    }