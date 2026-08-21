"""
Preprocessing module: PLS-based dimensionality reduction pipeline builder.

Constructs sklearn Pipelines that compress groups of correlated variables
into latent PLS components before modeling.
"""

import sys
from pathlib import Path

# utility.py lives in the parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from utility import GroupwisePLSTransformer

from config import (
    SPEED_VARS, DRAW_VARS, SPEEDSIZER_LINEPRESSURE_VARS,
    LINEPRESSURE_VARS, CONC_STARCH_VARS, INLET_TEMP_VARS,
    VACUUM_VARS, EXHAUST_MOISTURE_VARS, GAS_DECULATOR_VARS,
    FABRIC_TENSION_VARS, STARCH_TOP_VARS, STARCH_BOTTOM_VARS,
    STEAM_PRESSURE_VARS, STEAM_DIFF_PRESSURE_VARS,
)


# Mapping from prefix name to (columns, n_components)
_PLS_REGISTRY = {
    "steam_pressure": (
        STEAM_PRESSURE_VARS + STEAM_DIFF_PRESSURE_VARS, 4
    ),
    "draw": (DRAW_VARS, 2),
    "inlet_temp": (INLET_TEMP_VARS, 2),
    "speedsizer_linepressure": (SPEEDSIZER_LINEPRESSURE_VARS, 1),
    "linepressure": (LINEPRESSURE_VARS, 1),
    "conc_starch": (CONC_STARCH_VARS, 3),
    "draw_speed": (SPEED_VARS + DRAW_VARS, 4),
    "vacuum": (VACUUM_VARS, 4),
    "vacuum_speed": (VACUUM_VARS + ["Speed"], 4),
    "fabric_tension": (FABRIC_TENSION_VARS, 1),
    "gas_decu": (GAS_DECULATOR_VARS, 3),
    "exha_mois": (EXHAUST_MOISTURE_VARS, 2),
    "starch2": (STARCH_TOP_VARS, 3),
    "starch1": (STARCH_BOTTOM_VARS, 3),
}


def make_prep_pip(prefixes: list[str] = ()) -> tuple[Pipeline, list[str]]:
    """
    Build a preprocessing Pipeline of GroupwisePLSTransformers.

    Parameters
    ----------
    prefixes : list of group names to include (e.g. ["inlet_temp", "linepressure"])

    Returns
    -------
    (pipeline, source_vars) where source_vars are the raw columns consumed by PLS steps.
    """
    prep_pip = Pipeline(steps=[])
    source_vars: list[str] = []

    for prefix in prefixes:
        if prefix not in _PLS_REGISTRY:
            raise ValueError(
                f"Unknown PLS prefix '{prefix}'. "
                f"Available: {list(_PLS_REGISTRY.keys())}"
            )

        columns, n_components = _PLS_REGISTRY[prefix]

        transformer = GroupwisePLSTransformer(
            pls_columns=columns,
            n_components=n_components,
            score_prefix=prefix,
            remainder="passthrough",
        )

        prep_pip.steps.append((prefix, transformer))
        source_vars += list(transformer.pls_columns)

    return prep_pip, source_vars

def build_pre_estimator(
    exog_vars_reduced: list[str],
    prep_pip: Pipeline,
    created_vars: list[str],
    apply_ewm: bool = False,
    ewm_grade_col: str = "AB_Grade_ID",
    ewm_halflife: str = "120min",
    ewm_gap_threshold: str = "12h",
) -> tuple[Pipeline, list[str]]:
    """
    Build the final pre-estimator pipeline from a reduced variable list.

    This assembles:
      1. FeatureCreator (derived variables)
      2. (Optional) EWMSegmentFilter for segment-aware smoothing
      3. Relevant PLS transformers from prep_pip
      4. Column filter
      5. StandardScaler + OneHotEncoder for grammage

    Parameters
    ----------
    exog_vars_reduced : candidate feature names (with PLS scores, not sources)
    prep_pip : fitted PLS pipeline (used to resolve source <-> score mappings)
    created_vars : derived variables that need to be computed from raw inputs
    apply_ewm : whether to insert EWM segment filter after FeatureCreator
    ewm_grade_col : column for grade-change detection
    ewm_halflife : EWM halflife (pandas offset string)
    ewm_gap_threshold : time gap that triggers a new segment

    Returns
    -------
    (pre_estimator, feat_list) where feat_list is the raw input columns needed.
    """
    from copy import deepcopy
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from utility import FeatureCreator, SKColumnSelector
    from data_cleaning import unique_in_order, ordered_difference, ordered_intersection

    # Separate grammage to end, ensure it's categorical
    selected_model_features = (
        [v for v in exog_vars_reduced if "grammage" not in v] + ["grammage"]
    )
    features_to_keep_feat = selected_model_features.copy()
    features_to_keep_filter = selected_model_features.copy()

    # For each PLS step: if its scores are among selected features,
    # keep the scores in filter but expand source columns in feat
    for _, step in prep_pip.steps:
        if not isinstance(step, GroupwisePLSTransformer):
            continue
        transformed_names = [
            f"{step.score_prefix}_{i}"
            for i in range(1, step.n_components + 1)
        ]
        source_names = list(step.pls_columns)
        selected_transformed = ordered_intersection(
            selected_model_features, transformed_names
        )
        if selected_transformed:
            selected_model_features = ordered_difference(
                selected_model_features, selected_transformed
            )
            features_to_keep_feat = ordered_difference(
                features_to_keep_feat, selected_transformed
            )
            features_to_keep_filter = ordered_difference(
                features_to_keep_filter, selected_transformed
            )
            features_to_keep_feat = unique_in_order(
                features_to_keep_feat + source_names
            )
            features_to_keep_filter = unique_in_order(
                features_to_keep_filter + selected_transformed
            )

    features_to_create = ordered_intersection(created_vars, selected_model_features)

    # If EWM is enabled, ensure the grade column passes through FeatureCreator
    if apply_ewm:
        features_to_keep_feat = unique_in_order(features_to_keep_feat + [ewm_grade_col])

    # Assemble pipeline steps
    pre_estimator_steps = [
        ("feat", FeatureCreator(
            features_to_create=features_to_create,
            features_to_keep=unique_in_order(features_to_keep_feat),
            errors="raise",
        )),
    ]

    if apply_ewm:
        pre_estimator_steps.append(
            ("ewm", EWMSegmentFilter(
                grade_col=ewm_grade_col,
                halflife=ewm_halflife,
                gap_threshold=ewm_gap_threshold,
            ))
        )

    for name, step in prep_pip.steps:
        if isinstance(step, GroupwisePLSTransformer):
            transformed_names = [
                f"{step.score_prefix}_{i}"
                for i in range(1, step.n_components + 1)
            ]
            if ordered_intersection(features_to_keep_filter, transformed_names):
                pre_estimator_steps.append((name, deepcopy(step)))

    pre_estimator_steps.append(
        ("filter", SKColumnSelector(unique_in_order(features_to_keep_filter)))
    )

    # Categorical / numeric split
    categorical_features = [
        f for f in features_to_keep_filter if f == "grammage"
    ]
    numeric_features = [
        f for f in features_to_keep_filter if f not in categorical_features
    ]

    column_transformers = []
    if numeric_features:
        column_transformers.append(
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features)
        )
    if categorical_features:
        column_transformers.append(
            ("cat", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, drop="first"
            ), categorical_features)
        )

    scaler_encoder = ColumnTransformer(
        transformers=column_transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    pre_estimator_steps.append(("scaler_encoder", scaler_encoder))
    pre_estimator = Pipeline(steps=pre_estimator_steps)

    # Determine required raw input columns
    required_feature_inputs = unique_in_order(
        pre_estimator.named_steps["feat"].features_required()
    )
    feat_list = ordered_difference(
        unique_in_order(features_to_keep_feat + required_feature_inputs),
        features_to_create,
    )

    # EWM needs the grade column available in the input
    if apply_ewm and ewm_grade_col not in feat_list:
        feat_list = feat_list + [ewm_grade_col]

    return pre_estimator, feat_list


# =============================================================================
# EWM segment-aware smoother (sklearn-compatible transformer)
# =============================================================================

class EWMSegmentFilter(BaseEstimator, TransformerMixin):
    """
    Exponentially-weighted-mean filter that resets at segment boundaries.

    Segments are detected from:
      - grade changes (grade_col shifts)
      - time gaps > gap_threshold in the DatetimeIndex

    Parameters
    ----------
    grade_col : column name used to detect grade changes (default "AB_Grade_ID")
    gap_threshold : max gap before a new segment starts (default "12h")
    halflife : EWM halflife parameter (default "120min")
    exclude_patterns : column name patterns to skip (e.g. ["y_L"])
    """

    def __init__(
        self,
        grade_col: str = "AB_Grade_ID",
        gap_threshold: str = "12h",
        halflife: str = "120min",
        exclude_patterns: list[str] | None = None,
    ):
        self.grade_col = grade_col
        self.gap_threshold = gap_threshold
        self.halflife = halflife
        self.exclude_patterns = exclude_patterns or ["y_L"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd

        if not isinstance(X, pd.DataFrame):
            raise TypeError("EWMSegmentFilter expects a pandas DataFrame.")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("EWMSegmentFilter requires a DatetimeIndex.")

        df = X.copy()

        # Detect segments
        if self.grade_col in df.columns:
            grade_change = df[self.grade_col].ne(df[self.grade_col].shift())
        else:
            grade_change = pd.Series(False, index=df.index)

        time_gap = df.index.to_series().diff().gt(pd.Timedelta(self.gap_threshold))
        time_gap.iloc[0] = True

        seg = (grade_change | time_gap).cumsum()

        # Apply EWM per segment to numeric columns
        def _ewm(s):
            out = s.ewm(halflife=self.halflife, times=s.index, adjust=True).mean()
            if len(s) >= 2:
                out.iloc[0] = (s.iloc[0] + s.iloc[1]) / 2.0
            return out

        for col in df.columns:
            if any(pat in col for pat in self.exclude_patterns):
                continue
            if col == self.grade_col:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = df.groupby(seg, group_keys=False)[col].transform(_ewm)

        # Drop the grade column — it was only needed for segmentation
        if self.grade_col in df.columns:
            df = df.drop(columns=[self.grade_col])

        return df

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_final_pipeline(
    selected_features: list[str],
    prep_pip: Pipeline,
    created_vars: list[str],
    ridge_alpha: float = 1.0,
    apply_ewm: bool = False,
    ewm_grade_col: str = "AB_Grade_ID",
    ewm_halflife: str = "120min",
    ewm_gap_threshold: str = "12h",
) -> tuple[Pipeline, list[str]]:
    """
    Build a complete end-to-end pipeline for the selected features.

    This is the production pipeline: preprocessing + Ridge in one object.
    Call .fit(X_raw[feat_list], y) then .predict(X_new[feat_list]).

    Parameters
    ----------
    selected_features : output feature names chosen by CMA-ES
        (e.g. ["ambient_temp_C", "exha_mois_1", "grammage_90", ...])
    prep_pip : the PLS pipeline (used to resolve source/score mappings)
    created_vars : derived variables that FeatureCreator can produce
    ridge_alpha : regularization strength for Ridge
    apply_ewm : include EWM smoothing step
    ewm_grade_col, ewm_halflife, ewm_gap_threshold : EWM parameters

    Returns
    -------
    (pipeline, feat_list) where feat_list is the raw input columns required.
    """
    from sklearn.linear_model import Ridge

    # Rebuild pre_estimator for exactly the selected features.
    # PLS steps whose scores are absent from pre_features will be
    # automatically excluded by build_pre_estimator (along with their
    # source columns from feat_list). Only the needed pipeline steps remain.
    # Selected features may include one-hot encoded grammage columns
    # (e.g. "grammage_90"); map those back to "grammage" for the pipeline
    pre_features = []
    for f in selected_features:
        if "grammage" in f.lower():
            if "grammage" not in pre_features:
                pre_features.append("grammage")
        else:
            pre_features.append(f)

    pre_estimator, feat_list = build_pre_estimator(
        exog_vars_reduced=pre_features,
        prep_pip=prep_pip,
        created_vars=created_vars,
        apply_ewm=apply_ewm,
        ewm_grade_col=ewm_grade_col,
        ewm_halflife=ewm_halflife,
        ewm_gap_threshold=ewm_gap_threshold,
    )

    # Append Ridge as the final step
    pre_estimator.steps.append(("ridge", Ridge(alpha=ridge_alpha)))

    return pre_estimator, feat_list
