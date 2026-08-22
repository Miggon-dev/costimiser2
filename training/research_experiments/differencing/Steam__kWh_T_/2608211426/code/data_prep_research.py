"""
Shared data preparation for the *_research experiments.

Everything from raw parquet to a modelling-ready (X, y, t) lives here, so the
partialling experiment and the backfit-ablation experiment cannot drift apart in
their filtering, feature construction or target handling. Any comparison between
them is otherwise meaningless.

Existing project modules (config, data_cleaning, preprocessing, utility,
feature_selection, state_estimation) are imported read-only and never modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONTROL_VARS
from data_cleaning import (
    unique_in_order, ordered_difference, ordered_intersection, make_design,
)
from preprocessing import make_prep_pip, build_pre_estimator
from utility import GroupwisePLSTransformer, _feature_engineering, setpoint_df


# =============================================================================
# Per-target configuration
# =============================================================================

PIPELINE_PREFIXES = {
    "Steam__kWh/T_": ["exha_mois", "inlet_temp", "linepressure", "fabric_tension", "gas_decu"],
    "Electricity__kWh/T_": ["speedsizer_linepressure", "linepressure"],
    "Starch_uptake_by_paper_Top_Roll__g/m2_": ["speedsizer_linepressure", "linepressure", "starch2"],
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": ["speedsizer_linepressure", "linepressure", "starch1"],
    "MBS_SCT_CD": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
    "MBS_SCT_MD": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
    "MBS_Burst": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
    "MBS_CMT30": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
}

_ENERGY_BLACK_LIST = [
    "Current_basis_weight",
    "Bentonite_1_mass_flow__g/T_",
    "Bentonite_2_mass_flow__g/T_",
    "DG3_Moisture_content_Outlet_Air",
    "Lip_settings",
    "Conductivity_white_water_B46",
    "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
    "pH_measurement_white_water_B41",
    "CO2_mass_flow__g/T_",
]

_STARCH_BLACK_LIST = [
    "Bentonite_1_mass_flow__g/T_",
    "Bentonite_2_mass_flow__g/T_",
    "DG3_Moisture_content_Outlet_Air",
    "Lip_settings",
    "Conductivity_white_water_B46",
    "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
    "pH_measurement_white_water_B41",
    "CO2_mass_flow__g/T_",
]

_MBS_BLACK_LIST = [
    "Bentonite_1_mass_flow__g/T_",
    "Bentonite_2_mass_flow__g/T_",
    "DG3_Moisture_content_Outlet_Air",
    "Conductivity_white_water_B46",
    "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
    "pH_measurement_white_water_B41",
    "CO2_mass_flow__g/T_",
    "Current_reel_moisture_average(reel)",
]

BLACK_LIST = {
    "Steam__kWh/T_": _ENERGY_BLACK_LIST,
    "Electricity__kWh/T_": _ENERGY_BLACK_LIST,
    "Starch_uptake_by_paper_Top_Roll__g/m2_": _STARCH_BLACK_LIST,
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": _STARCH_BLACK_LIST,
    "MBS_SCT_CD": _MBS_BLACK_LIST,
    "MBS_SCT_MD": _MBS_BLACK_LIST,
    "MBS_Burst": _MBS_BLACK_LIST,
    "MBS_CMT30": _MBS_BLACK_LIST,
}

_MBS_CREATED_VARS = [
    "delta_basis_weight",
    "Starch_uptake__g/m2_",
    "Water_flow_Predryer",
    "Water_flow_Afterdryer",
    "Water_flow",
    "flow_diluted_starch",
    "Fibre__g/m2_",
    "Water_flow_Afterdryer_input",
    "Water_flow_Afterdryer_output",
    "dewatering",
    "fibre_short/long",
    "temperature_starch_working_tank",
]

CREATED_VARIABLE_CANDIDATES = {
    "Steam__kWh/T_": [
        "Water_Predryer", "diluted_starch", "Fibre__g/m2_",
        "Water_Afterdryer_output", "dewatering",
        "Production_Rate__T/h_", "fibre_short/long",
        "Starch_uptake__g/m2_",
    ],
    "Electricity__kWh/T_": [
        "dewatering",
        "Fibre__g/m2_",
    ],
    "Starch_uptake_by_paper_Top_Roll__g/m2_": [
        "delta_basis_weight",
        "Water_flow_Predryer",
        "Water_flow",
        "inv_Rod_pressure_Top_Roll",
        "square_Rod_pressure_Top_Roll",
    ],
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": [
        "delta_basis_weight",
        "Water_flow_Predryer",
        "Water_flow",
        "inv_Rod_Pressure_Bottom_Roll",
        "square_Rod_Pressure_Bottom_Roll",
    ],
    "MBS_SCT_CD": _MBS_CREATED_VARS,
    "MBS_SCT_MD": _MBS_CREATED_VARS,
    "MBS_Burst": _MBS_CREATED_VARS,
    "MBS_CMT30": _MBS_CREATED_VARS,
}

_MBS_FIXED = ["grammage", "conc_starch_1", "delta_basis_weight", "Jet/wire_ratio"]

DEFAULT_FIXED_FEATURES = {
    "Steam__kWh/T_": [
        "linepressure_1",
        "Starch_uptake__g/m2_",
        "grammage",
        # Forced in: the Steam window spans winter->summer, so a large part of
        # what a latent level would absorb is ambient air heating load.
        "ambient_temp_C",
    ],
    "Electricity__kWh/T_": ["Speed", "grammage"],
    "Starch_uptake_by_paper_Top_Roll__g/m2_": [
        "grammage", "Temperature_starch_working_tank_2",
        "starch2_1", "starch2_2", "starch2_3",
    ],
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": [
        "grammage", "Temperature_starch_working_tank_1",
        "starch1_1", "starch1_2", "starch1_3",
    ],
    "MBS_SCT_CD": _MBS_FIXED,
    "MBS_SCT_MD": _MBS_FIXED,
    "MBS_Burst": _MBS_FIXED,
    "MBS_CMT30": _MBS_FIXED,
}


# =============================================================================
# Mediator / tautological variables (setpoint-optimisation mode)
# =============================================================================
# NOT operator levers: either downstream consequences of the process being
# modelled, or near-restatements of the target.
#   water/dewatering/moisture  -> outcomes of forming, pressing, drying
#   exhaust air humidity       -> the direct signature of evaporation
#   Steam_*_for_PM             -> essentially the drying energy input
#   Production_Rate            -> derived from speed x width x basis weight
# Including them inflates fit while producing unactionable recommendations.
MEDIATOR_PATTERNS = {
    "Steam__kWh/T_": [
        r"^Water_Predryer",
        r"^Water_Afterdryer",
        r"^Water_flow",
        r"Moisture_out_of_PreDryer",
        r"Dewatering",
        r"Moisture_content_Outlet_Air",
        r"^Uhle_box_\d+_flow",
        r"^Starch_uptake",
        r"^Production_Rate",
        r"^Steam_(pressure|temperature)_for_PM$",
    ],
    # Deliberately absent for other targets rather than guessed. For the starch
    # targets the target itself is a starch-uptake variable, so the Steam list
    # would be actively wrong.
}

# PLS groups built entirely from mediator sources: dropping the sources without
# dropping the prefix would leave a degenerate PLS step.
MEDIATOR_PLS_PREFIXES = {
    "Steam__kWh/T_": ["exha_mois"],
}

GRAMMAGES = [115, 120, 100, 90, 85, 110]
# The supervised PLS projection is fitted on a leading block rather than on all
# rows, so it is not built with knowledge of the whole record.
PLS_FIT_FRACTION = 0.80
LAGS = 0


def is_mediator(name: str, patterns: list[str]) -> bool:
    """True if `name` matches any mediator pattern (case-insensitive)."""
    return any(re.search(p, name, re.I) for p in patterns)


# =============================================================================
# Target-specific row filtering
# =============================================================================

def _apply_target_filters(turnup_data: pd.DataFrame, y_column: str) -> pd.DataFrame:
    """Date windows and operating-state filters, per target."""
    td = turnup_data

    if y_column == "Steam__kWh/T_":
        td = td[td.index > "2025-11-1"]
        td = td[~((td.index > "2026-01-24 07:00") & (td.index < "2026-01-26 10:00"))]
        td = td[~((td.index > "2026-01-11 12:00") & (td.index < "2026-01-12 11:00"))]
        td = td[~((td.index > "2026-01-17 12:00") & (td.index < "2026-01-19 11:00"))]
        td = td[(td["DG4_Temperature_Inlet_Air"] > 100) & (td["Vacuum_Zone_1_PickUp"] < -0.5)]

    elif y_column == "Electricity__kWh/T_":
        td = td[td.index > "2025-04-01 00:00:00"]
        td = td[~((td.index > "2026-01-24 07:00") & (td.index < "2026-01-26 10:00"))]
        td = td[td["Vacuum_Zone_1_PickUp"] < -0.5]
        td = td[td.index > "2025-11-15"]

    elif y_column in ("Starch_uptake_by_paper_Top_Roll__g/m2_",
                      "Starch_uptake_by_paper_Bottom_Roll__g/m2_"):
        td = td[td.index > "2025-04-01 00:00:00"]
        td = td[~((td.index > "2026-01-24 07:00") & (td.index < "2026-01-26 10:00"))]
        td = td[td["Vacuum_Zone_1_PickUp"] < -0.5]
        td = td[td.index > "2026-3-1"]

    elif y_column in ("MBS_SCT_CD", "MBS_SCT_MD", "MBS_Burst", "MBS_CMT30"):
        td = td[td.index > "2025-04-01 00:00:00"]
        td = td[~((td.index > "2026-01-24 07:00") & (td.index < "2026-01-26 10:00"))]
        td = td[td.index > "2026-02-01"]
        if y_column == "MBS_CMT30":
            td = td[(td["AB_Grade_ID"] == "6010120") | (td["AB_Grade_ID"] == "6010100")]

    return td[td.grammage.isin(GRAMMAGES)]


# =============================================================================
# Result container
# =============================================================================

@dataclass
class PreparedData:
    """Modelling-ready data plus everything needed to interpret or persist it."""
    X: pd.DataFrame
    y: np.ndarray            # modelling target (EWM-filtered if requested)
    y_raw: np.ndarray        # unfiltered target, aligned to X.index
    t_hours: np.ndarray      # float hours since first row, for time smoothing
    t_index: pd.DatetimeIndex
    feature_names: list[str]
    feat_list: list[str]
    pre_estimator: object
    prep_pip: object
    created_vars: list[str]
    fixed_features: list[str]
    turnup_ts: pd.DataFrame
    y_column: str
    apply_ewm_filter: bool
    apply_ewm_filter_y: bool
    exclude_mediators: bool
    mediators_excluded: list[str] = field(default_factory=list)
    mediator_pls_dropped: list[str] = field(default_factory=list)
    pls_fit_rows: int = 0
    row_counts: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.X)

    @property
    def span_days(self) -> float:
        return float((self.t_hours[-1] - self.t_hours[0]) / 24.0)

    def summary(self) -> str:
        return (f"{self.y_column}: {self.n} rows, {len(self.feature_names)} features, "
                f"{self.span_days:.1f} days")


def to_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Timestamps -> float hours since the first observation."""
    t = pd.DatetimeIndex(index).asi8.astype(np.float64)
    return (t - t[0]) / 3.6e12


# =============================================================================
# Main entry point
# =============================================================================

def prepare(
    y_column: str,
    data_path: str | Path,
    *,
    apply_ewm_filter: bool = False,
    apply_ewm_filter_y: bool = False,
    exclude_mediators: bool = False,
    fixed_features: list[str] | None = None,
    verbose: bool = True,
) -> PreparedData:
    """
    Raw parquet -> (X, y, t) ready for modelling.

    apply_ewm_filter    grade-aware EWM on X (removes the noise band)
    apply_ewm_filter_y  grade-aware EWM on y (removes the noise band)
    exclude_mediators   drop downstream/tautological variables so only real
                        operator levers remain. Required for actionable setpoint
                        recommendations; lowers apparent fit.
    """
    def log(*a):
        if verbose:
            print(*a)

    _required = {
        "PIPELINE_PREFIXES": PIPELINE_PREFIXES,
        "BLACK_LIST": BLACK_LIST,
        "CREATED_VARIABLE_CANDIDATES": CREATED_VARIABLE_CANDIDATES,
        "DEFAULT_FIXED_FEATURES": DEFAULT_FIXED_FEATURES,
        "CONTROL_VARS (config.py)": CONTROL_VARS,
    }
    missing = [k for k, d in _required.items() if y_column not in d]
    if missing:
        raise KeyError(
            f"Target '{y_column}' is not configured in: {', '.join(missing)}. "
            f"Add an entry for it before running."
        )

    if fixed_features is None:
        fixed_features = list(DEFAULT_FIXED_FEATURES[y_column])

    mediator_patterns = MEDIATOR_PATTERNS.get(y_column, []) if exclude_mediators else []
    mediator_pls_prefixes = MEDIATOR_PLS_PREFIXES.get(y_column, []) if exclude_mediators else []
    if exclude_mediators and not mediator_patterns:
        log(f"WARNING: exclude_mediators set but no patterns configured for "
            f"'{y_column}'. No variables will be excluded.")

    # Fixed features are forced into every candidate subset, so a mediator left
    # here would defeat the exclusion entirely.
    dropped_fixed = [f for f in fixed_features if is_mediator(f, mediator_patterns)]
    if dropped_fixed:
        log(f"Dropping mediators from fixed_features: {dropped_fixed}")
        fixed_features = [f for f in fixed_features
                          if not is_mediator(f, mediator_patterns)]

    # ---- PLS pipeline -------------------------------------------------------
    pipeline_prefixes = [p for p in PIPELINE_PREFIXES[y_column]
                         if p not in mediator_pls_prefixes]
    dropped_pls = [p for p in PIPELINE_PREFIXES[y_column] if p in mediator_pls_prefixes]
    if dropped_pls:
        log(f"Dropping mediator PLS groups: {dropped_pls}")
    prep_pip, _prep_s_vars = make_prep_pip(prefixes=pipeline_prefixes)

    # ---- Load and filter ----------------------------------------------------
    turnup_data = pd.read_parquet(data_path)

    ctl_vars = unique_in_order(
        v for v in CONTROL_VARS[y_column] if "vacuum" not in v.lower()
    )
    mediators_found: list[str] = []
    if mediator_patterns:
        n_before = len(ctl_vars)
        mediators_found = [v for v in ctl_vars if is_mediator(v, mediator_patterns)]
        ctl_vars = [v for v in ctl_vars if not is_mediator(v, mediator_patterns)]
        log(f"Mediator exclusion: {n_before} -> {len(ctl_vars)} control vars")
        log(f"  Excluded ({len(mediators_found)}): {mediators_found}")

    # created_vars intersects the already-filtered ctl_vars, inheriting exclusion
    created_vars = ordered_intersection(CREATED_VARIABLE_CANDIDATES[y_column], ctl_vars)

    turnup_data = _feature_engineering(turnup_data, setpoint_df, steam_null=False, clip=False)
    turnup_data = turnup_data.set_index("Wedge_Time").sort_index()
    turnup_data = _apply_target_filters(turnup_data, y_column)
    log(f"Filtered data: {turnup_data.shape}")

    # ---- Feature list -------------------------------------------------------
    steam_pressure = [v for v in turnup_data.columns
                      if re.search(r"cylinder.*steam_pressure", v, re.I)]
    steam_diff_pressure = [v for v in turnup_data.columns
                           if re.search(r"cylinder.*differential_pressure", v, re.I)]

    exog_vars_reduced = [
        v for v in ctl_vars
        if (v not in BLACK_LIST[y_column] and v not in created_vars
            and v not in steam_pressure and v not in steam_diff_pressure
            and "vacuum" not in v.lower())
    ]
    exog_vars_reduced = unique_in_order(fixed_features + exog_vars_reduced + ["grammage"])

    for _, step in prep_pip.steps:
        if isinstance(step, GroupwisePLSTransformer):
            transformed = [f"{step.score_prefix}_{i}"
                           for i in range(1, step.n_components + 1)]
            exog_vars_reduced = ordered_difference(
                unique_in_order(exog_vars_reduced + transformed),
                list(step.pls_columns),
            )
    exog_vars_reduced = unique_in_order(fixed_features + exog_vars_reduced)

    pre_estimator, feat_list = build_pre_estimator(
        exog_vars_reduced=exog_vars_reduced,
        prep_pip=prep_pip,
        created_vars=created_vars,
        apply_ewm=apply_ewm_filter,
    )
    log(f"Pipeline input: {len(feat_list)} columns")

    # ---- Rows ---------------------------------------------------------------
    turnup_ts = turnup_data.copy().sort_index()
    n_before = len(turnup_ts)
    turnup_ts = turnup_ts.dropna(subset=[y_column])
    n_after_y = len(turnup_ts)
    turnup_ts = turnup_ts.dropna(subset=feat_list)
    n_after_feat = len(turnup_ts)
    log(f"Rows: {n_before} -> {n_after_y} (target NaN) -> {n_after_feat} (feature NaN)")
    if n_after_feat < 100:
        raise RuntimeError(f"Only {n_after_feat} rows survive filtering; cannot model.")

    # ---- Transform ----------------------------------------------------------
    # Every row that will be dropped has already been dropped, so this leading
    # block is a true fraction of the surviving rows.
    pls_fit_n = int(len(turnup_ts) * PLS_FIT_FRACTION)
    ts_raw = turnup_ts.loc[:, feat_list]
    pre_estimator.fit(ts_raw.iloc[:pls_fit_n], turnup_ts[y_column].iloc[:pls_fit_n])
    ts_transformed = pre_estimator.transform(ts_raw)
    log(f"Transformed shape: {ts_transformed.shape}")

    y_raw_vals = turnup_ts[y_column].values.astype(float).copy()
    ts_transformed[y_column] = y_raw_vals

    if apply_ewm_filter_y:
        from utility import ewm_reset
        grammage_group = pd.Series(turnup_ts["grammage"].values)
        grade_change = grammage_group.ne(grammage_group.shift())
        time_gap = ts_transformed.index.to_series().diff().gt(pd.Timedelta("12h"))
        time_gap.iloc[0] = True
        seg = (grade_change.values | time_gap.values).cumsum()
        y_series = pd.Series(y_raw_vals, index=ts_transformed.index)
        ts_transformed[y_column] = y_series.groupby(seg).transform(ewm_reset).values

    transformed_feature_names = [c for c in ts_transformed.columns if c != y_column]
    X, y = make_design(ts_transformed, y_column, transformed_feature_names, None,
                       y_lags=range(1, 1 + LAGS))
    y = np.asarray(y, dtype=float).ravel()

    y_raw_aligned = pd.Series(y_raw_vals, index=ts_transformed.index).loc[X.index].values

    t_index = pd.DatetimeIndex(X.index)
    if not t_index.is_monotonic_increasing:
        raise RuntimeError("Design matrix index is not sorted in time.")

    prepared = PreparedData(
        X=X,
        y=y,
        y_raw=y_raw_aligned,
        t_hours=to_hours(t_index),
        t_index=t_index,
        feature_names=list(X.columns),
        feat_list=list(feat_list),
        pre_estimator=pre_estimator,
        prep_pip=prep_pip,
        created_vars=list(created_vars),
        fixed_features=list(fixed_features),
        turnup_ts=turnup_ts,
        y_column=y_column,
        apply_ewm_filter=apply_ewm_filter,
        apply_ewm_filter_y=apply_ewm_filter_y,
        exclude_mediators=exclude_mediators,
        mediators_excluded=mediators_found,
        mediator_pls_dropped=dropped_pls,
        pls_fit_rows=int(pls_fit_n),
        row_counts={"initial": int(n_before), "after_target_nan": int(n_after_y),
                    "after_feature_nan": int(n_after_feat)},
    )
    log(f"Design matrix: X={X.shape}, y={y.shape}")
    log(f"Span: {prepared.span_days:.1f} days")
    return prepared


def resolve_fixed_features(prepared: PreparedData) -> list[str]:
    """
    Fixed features as they appear in the TRANSFORMED matrix.

    `grammage` becomes one-hot columns, so it is matched by prefix rather than by
    exact name.
    """
    cols = prepared.feature_names
    resolved = [v for v in prepared.fixed_features
                if "grammage" not in v.lower() and v in cols]
    resolved += [c for c in cols if "grammage" in c.lower()]
    return unique_in_order(resolved)


def pls_feature_groups(prepared: PreparedData) -> dict[str, list[str]]:
    """PLS score groups present in X, for all-or-nothing feature selection."""
    groups = {}
    for _, step in prepared.prep_pip.steps:
        if isinstance(step, GroupwisePLSTransformer):
            cols = [f"{step.score_prefix}_{i}" for i in range(1, step.n_components + 1)]
            present = [c for c in cols if c in prepared.feature_names]
            if present:
                groups[step.score_prefix] = present
    return groups
