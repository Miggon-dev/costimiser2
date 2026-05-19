"""
shap_tools.py

Thin wrapper around the project's SHAP utilities.

This version keeps the original public API and return structure, but computes
SHAP values against the cost functions registered in prediction_tools.PREDICTORS
instead of directly explaining sklearn estimators.
"""

from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import shap

from process_data_tools import load_turnup_data
from utility import plotly_shap_beeswarm
from prediction_tools import PREDICTORS, get_required_features, list_available_components


GRADE_COL = "AB_Grade_ID"


# -------------------------------------------------
# Supported components
# -------------------------------------------------

# Kept for backward compatibility with code that imports SHAP_MODELS.
# Values are predictor specs instead of sklearn estimators.
SHAP_MODELS = PREDICTORS


def _normalise_component(component: str) -> str:
    c = str(component).strip()
    cl = c.lower().strip()

    # remove common suffixes
    cl = cl.replace("_cost", "")
    cl = cl.replace(" cost", "")
    cl = cl.strip()

    if cl in {"fibre", "fiber"}:
        return "fibre"

    if cl == "steam":
        return "steam"

    if cl in {"electricity", "power"}:
        return "electricity"

    if cl == "starch":
        return "starch"

    if cl in {"total", "combined", "overall", "combined cost", "overall cost"}:
        return "total"

    if cl in {"sct cd", "sct_cd", "sctcd", "strength"}:
        return "SCT CD"
    
    if cl in {"sct md", "sct_md", "sctmd"}:
        return "SCT MD"

    if cl in {"burst"}:
        return "Burst"
    
    if cl in {"cmt", "cmt_30", "cmt 30", "cmt30"}:
        return "CMT30"

    return c



def list_shap_components():
    """Return components supported by cost-function SHAP."""
    return list_available_components()


def get_model_for_component(component: str):
    """
    Backward-compatible accessor.

    Historically this returned a sklearn estimator. Now it returns the registered
    prediction spec from prediction_tools.PREDICTORS.
    """
    component = _normalise_component(component)

    if component not in PREDICTORS:
        raise KeyError(
            f"SHAP not supported for component '{component}'. "
            f"Supported: {list_shap_components()}"
        )

    return PREDICTORS[component]


def get_reference_data() -> pd.DataFrame:
    """
    Use the full engineered turnup dataframe as reference/background.
    """
    return load_turnup_data()


def _prepare_shap_frame(
    df: pd.DataFrame,
    component: str,
    grade_col: str = GRADE_COL,
) -> pd.DataFrame:
    """
    Keep only the cost-function predictors required for this component, plus grade.

    This preserves the old behavior: all non-grade columns in the prepared frame
    are interpreted as SHAP features.
    """
    component = _normalise_component(component)
    features = get_required_features(component)
    cols = [grade_col] + features

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing SHAP columns for component '{component}': {missing}")

    return df[cols].copy()


def _predict_cost_batch(predict_fn_raw, X: pd.DataFrame) -> np.ndarray:
    """
    Run a registered cost function on a dataframe and always return a 1D array.

    Supports both vectorized cost functions and row-wise functions such as
    fibre_cost(row: pd.Series) -> float.
    """
    try:
        y = predict_fn_raw(X)
        y = np.asarray(y, dtype=float)

        if y.ndim == 0:
            return np.repeat(float(y), len(X))

        if len(y) == len(X):
            return y.reshape(-1)

    except Exception:
        pass

    return X.apply(lambda row: float(predict_fn_raw(row)), axis=1).to_numpy()


def _make_shap_predict_fn(component: str, feature_names: List[str]):
    """
    Build a SHAP-compatible prediction function.

    SHAP passes numpy arrays; cost functions expect a pandas DataFrame/Series.
    """
    component = _normalise_component(component)
    spec = get_model_for_component(component)
    predict_fn_raw = spec["predict_fn"]

    def predict_fn(X):
        X_df = pd.DataFrame(X, columns=feature_names)
        return _predict_cost_batch(predict_fn_raw, X_df)

    return predict_fn


def _impute_for_shap(
    X: pd.DataFrame,
    X_reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Make sample/background usable by model-agnostic SHAP without dropping rows
    because of unrelated columns.

    At this point X and X_reference already contain only true component features,
    so imputation is only applied to predictors used by the cost function.
    """
    X = X.copy()
    X_reference = X_reference.copy()

    feature_names = list(X.columns)

    # Drop features that are entirely unavailable in both sample and reference.
    unusable = [
        c for c in feature_names
        if X[c].isna().all() and X_reference[c].isna().all()
    ]

    if unusable:
        feature_names = [c for c in feature_names if c not in unusable]
        X = X[feature_names]
        X_reference = X_reference[feature_names]

    if not feature_names:
        raise ValueError("No usable SHAP features after removing fully-empty predictors.")

    # Numeric median imputation. Cost predictors should be numeric.
    for c in feature_names:
        if X[c].isna().any() or X_reference[c].isna().any():
            fill_value = X_reference[c].median()
            if pd.isna(fill_value):
                fill_value = X[c].median()
            if pd.isna(fill_value):
                raise ValueError(f"Cannot impute SHAP feature '{c}'; all values are NA.")

            X[c] = X[c].fillna(fill_value)
            X_reference[c] = X_reference[c].fillna(fill_value)

    return X, X_reference, feature_names


def compute_shap_for_component(
    component: str,
    X_sample: pd.DataFrame,
    grade_id: Optional[str] = None,
    X_reference: Optional[pd.DataFrame] = None,
    grade_col: str = GRADE_COL,
    max_background: int = 200,
    max_samples: int = 500,
    random_state: int = 0,
) -> Any:
    """
    Compute SHAP values for one component using its registered cost function.

    Return contract is intentionally compatible with the previous implementation:

        base_values, shap_values, Xe, feature_names

    where:
    - base_values: SHAP expected/base values
    - shap_values: ndarray, shape (n_rows, n_features)
    - Xe: dataframe explained by SHAP
    - feature_names: list of explained feature names
    """
    component = _normalise_component(component)

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

    # Filter sample by grade, preserving previous behavior.
    if grade_id is not None:
        X_sample_prepared = X_sample_prepared[
            X_sample_prepared[grade_col].astype(str) == str(grade_id)
        ].copy()

        if X_sample_prepared.empty:
            raise ValueError(f"No rows found for grade {grade_id}")

    # Prefer same-grade background if available, otherwise use all reference rows.
    X_reference_bg = X_reference_prepared.copy()
    if grade_id is not None:
        bg_grade = X_reference_bg[
            X_reference_bg[grade_col].astype(str) == str(grade_id)
        ].copy()
        if not bg_grade.empty:
            X_reference_bg = bg_grade

    # All non-grade columns are true cost-function predictors.
    feature_names = [c for c in X_sample_prepared.columns if c != grade_col]

    Xe = X_sample_prepared[feature_names].copy()
    Xbg = X_reference_bg[feature_names].copy()

    Xe, Xbg, feature_names = _impute_for_shap(Xe, Xbg)

    if len(Xe) > max_samples:
        Xe = Xe.sample(max_samples, random_state=random_state)

    if len(Xbg) > max_background:
        Xbg = Xbg.sample(max_background, random_state=random_state)

    predict_fn = _make_shap_predict_fn(component, feature_names)

    masker = shap.maskers.Independent(Xbg)
    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
    )

    explanation = explainer(Xe)

    base_values = explanation.base_values
    shap_values = explanation.values

    return base_values, shap_values, Xe, feature_names


def explain_grade_component(
    component: str,
    grade_id: Optional[str] = None,
    X_sample: Optional[pd.DataFrame] = None,
    X_reference: Optional[pd.DataFrame] = None,
    grade_col: str = GRADE_COL,
) -> Dict[str, Any]:
    """
    Compute SHAP for one component.

    Compatibility preserved with the uploaded version. Return keys are:
    - component
    - grade_id
    - base_values
    - shap_values
    - Xe
    - feature_names
    - data_frame
    - figure
    """
    component = _normalise_component(component)

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

    shap_df = build_shap_dataframe(
        Xe=Xe,
        shap_values=shap_values,
        feature_names=feature_names,
    )

    fig = None
    try:
        shap_result = {
            "shap_values": shap_values,
            "Xe": Xe,
            "feature_names": feature_names,
        }
        fig = build_shap_beeswarm_figure(shap_result)
    except Exception as e:
        print("build_shap_beeswarm_figure failed:", e)
        fig = None

    return {
        "component": component,
        "grade_id": None if grade_id is None else str(grade_id),
        "base_values": base_values,
        "shap_values": shap_values,
        "Xe": Xe,
        "feature_names": feature_names,
        "data_frame": shap_df,
        "figure": fig,
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


def build_shap_dataframe(
    Xe: pd.DataFrame,
    shap_values,
    feature_names,
) -> pd.DataFrame:
    """
    Return a long dataframe with:
    - row index
    - feature
    - value
    - shap_value
    """
    if Xe is None or len(Xe) == 0:
        return pd.DataFrame(columns=["row_id", "feature", "value", "shap_value"])

    Xv = Xe.copy()
    sv = np.asarray(shap_values)

    rows = []
    for i in range(len(Xv)):
        for j, feat in enumerate(feature_names):
            value = Xv.iloc[i][feat] if feat in Xv.columns else None
            shap_val = sv[i, j]
            rows.append(
                {
                    "row_id": i,
                    "feature": feat,
                    "value": value,
                    "shap_value": shap_val,
                }
            )

    return pd.DataFrame(rows)