"""
prediction_tools.py
 
Prediction layer for cost components.
 
This module wraps:
- ML-based cost models
- formula-based cost models
- feature requirements
- prediction from a single row or dataframe
"""
 
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional
from utility import make_model_cost_batch, make_models_cost_batch
 
import pickle
import pandas as pd
 
 
MODELS_DIR = Path("models").resolve()
 
 
# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _row_has_key(row, key: str) -> bool:
    if isinstance(row, pd.Series):
        return key in row.index
    if isinstance(row, dict):
        return key in row
    try:
        row[key]
        return True
    except Exception:
        return False
      
 
def require_columns(row, cols: List[str], component_name: str) -> None:        
    missing = [c for c in cols if not _row_has_key(row, c)]
    if missing:
        raise KeyError(f"Missing columns for '{component_name}': {missing}")
 
 
# -------------------------------------------------
# Load models
# -------------------------------------------------
 
with open(MODELS_DIR / "model_electricity.pkl", "rb") as f:
    electricity_model = pickle.load(f)
 
with open(MODELS_DIR / "model_steam.pkl", "rb") as f:
    steam_model = pickle.load(f)
 
with open(MODELS_DIR / "model_starch_top.pkl", "rb") as f:
    model_starch_top = pickle.load(f)

with open(MODELS_DIR / "model_starch_bottom.pkl", "rb") as f:
    model_starch_bottom = pickle.load(f)

with open(MODELS_DIR / "model_SCTCD.pkl", "rb") as f:
    model_SCTCD = pickle.load(f)

with open(MODELS_DIR / "model_cost_starch_bottom.pkl", "rb") as f:
    model_cost_starch_bottom = pickle.load(f)

with open(MODELS_DIR / "model_cost_starch_top.pkl", "rb") as f:
    model_cost_starch_top = pickle.load(f) 
 
# -------------------------------------------------
# Feature definitions
# -------------------------------------------------
 
def steam_features() -> List[str]:
    return list(steam_model.best_estimator_.feature_names_in_)
 
def electricity_features() -> List[str]:
    return list(electricity_model.best_estimator_.feature_names_in_)
 
def starch_features():
    return list(set(list(model_starch_top.best_estimator_.feature_names_in_) + list(model_starch_bottom.best_estimator_.feature_names_in_)))

def starch_cost_features():
    return list(set(list(model_cost_starch_bottom.best_estimator_.feature_names_in_) + list(model_cost_starch_top.best_estimator_.feature_names_in_)))
 
def fibre_features() -> List[str]:
    return [
        "Current_basis_weight",
        "Starch_uptake_by_paper_Bottom_Roll__g/m2_",
        "Starch_uptake_by_paper_Top_Roll__g/m2_",
        "Current_reel_moisture_average(reel)",
    ]

def total_features():
    return list(set(list(model_starch_top.best_estimator_.feature_names_in_) + 
                    list(model_starch_bottom.best_estimator_.feature_names_in_) + 
                    list(steam_model.best_estimator_.feature_names_in_) + 
                    list(electricity_model.best_estimator_.feature_names_in_)
                    ))

def SCTCD_features():
    return list(model_SCTCD.best_estimator_.feature_names_in_)
 
 
# -------------------------------------------------
# Formula-based component
# -------------------------------------------------
 
def fibre_cost_from_row(row) -> float:    
    #require_columns(row, fibre_features(), "fibre")    
    if type(row) == pd.Series:
        row = row.to_dict()
    basis_weight = row["Current_basis_weight"]
    starch_uptake = row["Starch_uptake_by_paper_Bottom_Roll__g/m2_"] + row["Starch_uptake_by_paper_Top_Roll__g/m2_"]
    moisture = row["Current_reel_moisture_average(reel)"]

    return 146.46 * (basis_weight * (1 - moisture / 100) - starch_uptake) / basis_weight


def fibre_cost(X):
    """
    Behaves like the model-based cost functions:
    - accepts one row: pd.Series / dict
    - accepts a dataframe: pd.DataFrame
    """    
    if isinstance(X, list):
        X = pd.DataFrame(X)
    if isinstance(X, pd.DataFrame):
        
        missing = [c for c in fibre_features() if c not in X.columns]
        if missing:
            raise KeyError(f"Missing columns for 'fibre': {missing}")

        basis_weight = X["Current_basis_weight"]
        starch_uptake = X["Starch_uptake_by_paper_Bottom_Roll__g/m2_"] + X["Starch_uptake_by_paper_Top_Roll__g/m2_"]
        moisture = X["Current_reel_moisture_average(reel)"]

        return 146.46 * (basis_weight * (1 - moisture / 100) - starch_uptake) / basis_weight    
    return float(fibre_cost_from_row(X))
 
 
# -------------------------------------------------
# Model-based components
# -------------------------------------------------
 
_, steam_cost = make_model_cost_batch(steam_model, steam_features)
_, electricity_cost = make_model_cost_batch(electricity_model, electricity_features)
_, starch_cost = make_models_cost_batch(
    models={
        "bottom": model_starch_bottom,
        "top": model_starch_top,
    },
    feature_fns={
        "bottom": list(model_starch_bottom.best_estimator_.feature_names_in_),
        "top": list(model_starch_top.best_estimator_.feature_names_in_),
    },
    agg="sum",
) 

_, total_cost = make_models_cost_batch(
    models={
        "starch_bottom": model_cost_starch_bottom,
        "starch_top": model_cost_starch_top,
        "fibre": fibre_cost,
        "steam": steam_model,
        "electricity": electricity_model,
    },
    feature_fns={
        "starch_bottom": list(model_cost_starch_bottom.feature_names_in_),
        "starch_top": list(model_cost_starch_top.feature_names_in_),
        "fibre": fibre_features(),
        "steam": list(steam_model.best_estimator_.feature_names_in_),
        "electricity": list(electricity_model.best_estimator_.feature_names_in_),
    },
    weights = {
        "starch_bottom": 434.22,
        "starch_top": 434.22,
        "fibre": 146.46,
        "steam": 89.03 / 1000,
        "electricity": 113.66 / 1000
    },
    agg="sum",
)

_, sctcd_strength = make_model_cost_batch(model_SCTCD, SCTCD_features)
 
# -------------------------------------------------
# Registry
# -------------------------------------------------
 
PREDICTORS: Dict[str, Dict[str, Any]] = {
    "fibre": {
        "kind": "formula",
        "features_fn": fibre_features,
        "predict_fn": fibre_cost,
        "unit": "t/t",
    },
    "steam": {
        "kind": "model",
        "features_fn": steam_features,
        "predict_fn": steam_cost,
        "unit": "kWh/t",
    },
    "electricity": {
        "kind": "model",
        "features_fn": electricity_features,
        "predict_fn": electricity_cost,
        "unit": "kWh/t",
    },
    "starch": {
        "kind": "model",
        "features_fn": starch_features,
        "predict_fn": starch_cost,
        "unit": "kg/t",
    },
    "total": {
        "kind": "model",
        "features_fn": total_features,
        "predict_fn": total_cost,
        "unit": "€/t",
    },
    "SCT CD": {
        "kind": "model",
        "features_fn": SCTCD_features,
        "predict_fn": sctcd_strength,
        "unit": "",
    },
}
 
 
# -------------------------------------------------
# Public API
# -------------------------------------------------
 
def list_available_components() -> List[str]:
    return list(PREDICTORS.keys())
 
 
def get_required_features(component: str) -> List[str]:
    if component not in PREDICTORS:
        raise KeyError(f"Unknown component '{component}'. Available: {list_available_components()}")
    return list(PREDICTORS[component]["features_fn"]())
 
 
def predict_component_from_row(row: pd.Series, component: str) -> float:
    if component not in PREDICTORS:
        raise KeyError(f"Unknown component '{component}'. Available: {list_available_components()}")

    spec = PREDICTORS[component]
    required = spec["features_fn"]()
    require_columns(row, required, component)

    return float(spec["predict_fn"](row))
 
 
def predict_costs_from_row(
    row: pd.Series,
    components: Optional[List[str]] = None,
) -> Dict[str, float]:
    if components is None:
        components = list_available_components()
 
    out: Dict[str, float] = {}
    for component in components:
        out[component] = predict_component_from_row(row, component)
 
    return out
 
 
def predict_costs_from_dataframe(
    df: pd.DataFrame,
    components: Optional[List[str]] = None,
) -> pd.DataFrame:
    if components is None:
        components = list_available_components()
 
    preds = []
    for _, row in df.iterrows():
        pred_row = predict_costs_from_row(row, components=components)
        preds.append(pred_row)
 
    pred_df = pd.DataFrame(preds, index=df.index)
    return pred_df