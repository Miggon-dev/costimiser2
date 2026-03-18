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
     
def make_model_cost(model, feature_fn: Callable[[], List[str]]) -> Callable[[Any], float]:
    cols = feature_fn()
    df = pd.DataFrame([[0.0] * len(cols)], columns=cols)  # allocate once
 
    def cost(row) -> float:
        require_columns(row, cols, "model")
        df.iloc[0] = [row[c] for c in cols]
        return float(model.predict(df)[0])
 
    return cost
 
 
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
 
with open(MODELS_DIR / "model_starch.pkl", "rb") as f:
    starch_model = pickle.load(f)
 
 
# -------------------------------------------------
# Feature definitions
# -------------------------------------------------
 
def steam_features() -> List[str]:
    return list(steam_model.best_estimator_.feature_names_in_)
 
def electricity_features() -> List[str]:
    return list(electricity_model.best_estimator_.feature_names_in_)
 
def starch_features() -> List[str]:
    return list(starch_model.best_estimator_.feature_names_in_)
 
def fibre_features() -> List[str]:
    return [
        "Current_basis_weight",
        "Starch_uptake__g/m2_",
        "Current_reel_moisture_average(reel)",
    ]
 
 
# -------------------------------------------------
# Formula-based component
# -------------------------------------------------
 
def fibre_cost(row: pd.Series) -> float:
    require_columns(row, fibre_features(), "fibre")
 
    basis_weight = row["Current_basis_weight"]
    starch_uptake = row["Starch_uptake__g/m2_"]
    moisture = row["Current_reel_moisture_average(reel)"]
 
    return 146.46 * (basis_weight * (1 - moisture / 100) - starch_uptake) / basis_weight
 
 
# -------------------------------------------------
# Model-based components
# -------------------------------------------------
 
steam_cost = make_model_cost(steam_model, steam_features)
electricity_cost = make_model_cost(electricity_model, electricity_features)
starch_cost = make_model_cost(starch_model, starch_features)
 
 
# -------------------------------------------------
# Registry
# -------------------------------------------------
 
PREDICTORS: Dict[str, Dict[str, Any]] = {
    "fibre": {
        "kind": "formula",
        "features_fn": fibre_features,
        "predict_fn": fibre_cost,
        "unit": "€/t",
    },
    "steam": {
        "kind": "model",
        "features_fn": steam_features,
        "predict_fn": steam_cost,
        "unit": "€/t",
    },
    "electricity": {
        "kind": "model",
        "features_fn": electricity_features,
        "predict_fn": electricity_cost,
        "unit": "€/t",
    },
    "starch": {
        "kind": "model",
        "features_fn": starch_features,
        "predict_fn": starch_cost,
        "unit": "€/t",
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