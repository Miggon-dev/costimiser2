"""
shap_tools.py
 
Thin wrapper around the project's existing SHAP utilities.
 
This module does not reimplement SHAP logic.
It standardizes access for the assistant system.
"""
 
from typing import Dict, Any, Optional
 
import pandas as pd
 
from process_data_tools import load_turnup_data
from prediction_tools import steam_model, electricity_model, starch_model
from utility import calculate_manual_shap, plotly_shap_beeswarm
from prediction_tools import (
    steam_model,
    electricity_model,
    starch_model,
    get_required_features,
)
 
 
GRADE_COL = "AB_Grade_ID"
 
 
# -------------------------------------------------
# Supported components
# -------------------------------------------------
 
SHAP_MODELS = {
    "steam": steam_model.best_estimator_,
    "electricity": electricity_model.best_estimator_,
    "starch": starch_model.best_estimator_,
}
 
def _prepare_shap_frame(
    df: pd.DataFrame,
    component: str,
    grade_col: str = GRADE_COL,
) -> pd.DataFrame:
    """
    Keep only the model features required for this component, plus the grade column.
    This is necessary because calculate_manual_shap expects all non-grade columns
    to be explainer features.
    """
    features = get_required_features(component)
    cols = [grade_col] + features
 
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing SHAP columns for component '{component}': {missing}")
 
    out = df[cols].copy()
    return out
 
def list_shap_components():
    return list(SHAP_MODELS.keys())
 
 
def get_model_for_component(component: str):
    if component not in SHAP_MODELS:
        raise KeyError(f"SHAP not supported for component '{component}'. Supported: {list_shap_components()}")
    return SHAP_MODELS[component]
 
 
def get_reference_data() -> pd.DataFrame:
    """
    For now, use the full engineered turnup dataframe as reference/background.
    Later we may replace this with the actual training dataframe.
    """
    return load_turnup_data()
 
 
def compute_shap_for_component(
    component: str,
    X_sample: pd.DataFrame,
    grade_id: Optional[str] = None,
    X_reference: Optional[pd.DataFrame] = None,
    grade_col: str = GRADE_COL,
) -> Any:
    """
    Wrapper around calculate_manual_shap.
 
    Important:
    calculate_manual_shap assumes that all columns except grade_col are model features,
    so we must subset X_sample and X_reference to:
        [grade_col] + required model features
    """
    model = get_model_for_component(component)
 
    if X_reference is None:
        X_reference = get_reference_data()
 
    X_sample_prepared = _prepare_shap_frame(
        df=X_sample,
        component=component,
        grade_col=grade_col,
    )
 
    X_reference_prepared = _prepare_shap_frame(
        df=X_reference,
        component=component,
        grade_col=grade_col,
    )
 
    return calculate_manual_shap(
        model=model,
        X_sample=X_sample_prepared,
        grade_id=grade_id,
        X_reference=X_reference_prepared,
        grade_col=grade_col,
    )
 
 
def explain_grade_component(
    component: str,
    grade_id: str,
    X_sample: Optional[pd.DataFrame] = None,
    X_reference: Optional[pd.DataFrame] = None,
    grade_col: str = GRADE_COL,
) -> Dict[str, Any]:
    """
    Compute SHAP for one component and one grade.
    Returns a structured dictionary.
    """
    if X_sample is None:
        X_sample = load_turnup_data()
 
    if X_reference is None:
        X_reference = get_reference_data()
 
    base_values, shap_values, Xe, feature_names = compute_shap_for_component(
        component=component,
        X_sample=X_sample,
        grade_id=grade_id,
        X_reference=X_reference,
        grade_col=grade_col,
    )
 
    return {
        "component": component,
        "grade_id": str(grade_id),
        "base_values": base_values,
        "shap_values": shap_values,
        "Xe": Xe,
        "feature_names": feature_names,
    }
 
 
def build_shap_beeswarm_figure(
    shap_result: Dict[str, Any],
    max_features: int = 15,
):
    """
    Build the Plotly beeswarm figure from explain_grade_component output.
    """
    return plotly_shap_beeswarm(
        shap_values=shap_result["shap_values"],
        X_feat=shap_result["Xe"],
        feature_names=shap_result["feature_names"],
        max_features=max_features,
    )