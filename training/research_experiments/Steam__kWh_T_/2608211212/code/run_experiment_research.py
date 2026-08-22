"""
Setpoint-optimisation research: identify f in  y = f(X) + s(t) + eps
by partialling the latent state out of BOTH sides before fitting anything.

WHY NOT THE PREVIOUS APPROACHES
-------------------------------
  y = f(X)                  fails: the unobserved slowly-varying machine state
                            s(t) is not in the model at all
  y = f(X) + s, jointly     fails to identify: a random-walk level has ~1 dof
                            per observation and the f/s split is decided by a
                            variance ratio estimated from the same likelihood
                            that fits f, so the level absorbs any slow covariate
                            variation
  y = f(X) + s, backfitting  only bounds the absorption by stopping early. That
                            is an accidental regulariser with no stopping rule,
                            and it forces Ridge-only, forces feature selection,
                            and rules out flexible models
  delta_y = g(delta_X)      fails because differencing is a pure HIGH-pass, and
                            grade-aware EWM already showed the high band is noise

THIS APPROACH
-------------
Remove E[.|t] from y and from every column of X with the SAME fixed operator,
then fit f on the residualised data. For linear f this is exact (a linear filter
commutes with a linear model), so coefficients are preserved while the trend is
annihilated. The bandwidth is chosen by validation, not by MLE on the fit, which
is what breaks the circularity.

Frequency view: EWM removes the noise band, time-partialling removes the drift
band, and f is identified from the mid band in between.

Consequences: any learner may be used (the nuisance is gone before fitting, so
nothing competes with it), feature selection becomes optional, and there is no
iteration or convergence question. For setpoint optimisation s cancels anyway,
since argmax_X [f(X) + s] == argmax_X f(X).

Usage:
    python run_experiment_research.py --y_column "Steam__kWh/T_" \
        --data_path ../data/costimier_turnup.parquet \
        --apply_ewm_filter --apply_ewm_filter_y --exclude_mediators

Key parameters:
    --exclude_mediators   drop downstream/tautological variables (required for
                          actionable setpoint recommendations)
    --bandwidths          smoother bandwidth grid, in DAYS
    --learners            ridge,hist_gbr,realmlp
    --variant_b           also run flexible-f-with-small-time-basis
    --nuisance_audit      re-estimate the nuisance per fold to check the
                          coefficients do not depend on full-sample partialling
"""

import sys
from pathlib import Path

# Add parent directory (where utility.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re
from datetime import datetime

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CONTROL_VARS
from data_cleaning import (
    unique_in_order, ordered_difference, ordered_intersection, make_design,
)
from preprocessing import make_prep_pip, build_pre_estimator
from utility import GroupwisePLSTransformer, _feature_engineering, setpoint_df

import partialling_research as P


# =============================================================================
# Default configuration
# =============================================================================

PIPELINE_PREFIXES = {
    "Steam__kWh/T_": ["exha_mois", "inlet_temp", "linepressure", "fabric_tension", "gas_decu"],
     "Electricity__kWh/T_": ["speedsizer_linepressure", "linepressure"],
     "Starch_uptake_by_paper_Top_Roll__g/m2_": ["speedsizer_linepressure", "linepressure","starch2"],
     "Starch_uptake_by_paper_Bottom_Roll__g/m2_": ["speedsizer_linepressure", "linepressure","starch1"],
     "MBS_SCT_CD": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
     "MBS_SCT_MD": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
     "MBS_Burst": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],
     "MBS_CMT30": ["draw", "speedsizer_linepressure", "linepressure", "conc_starch"],

}

BLACK_LIST = {
    "Steam__kWh/T_": [
        "Current_basis_weight", 
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", 
        "CO2_mass_flow__g/T_",
    ],
    "Electricity__kWh/T_": [
        "Current_basis_weight", 
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", 
        "CO2_mass_flow__g/T_",
    ],
    "Starch_uptake_by_paper_Top_Roll__g/m2_": [
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", "CO2_mass_flow__g/T_",
    ],
    "Starch_uptake_by_paper_Bottom_Roll__g/m2_": [
        "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", 
        "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", 
        "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", "CO2_mass_flow__g/T_",
    ],
    "MBS_SCT_CD": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
    "MBS_SCT_MD": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
     "MBS_Burst": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
     "MBS_CMT30": [
        'Bentonite_1_mass_flow__g/T_',
        'Bentonite_2_mass_flow__g/T_',
        'DG3_Moisture_content_Outlet_Air',               
        'Conductivity_white_water_B46',
        'pH-Messung_Verdünnungswasser__2..12_pH_', 
        'pH_measurement_white_water_B41',
        'CO2_mass_flow__g/T_',
        "Current_reel_moisture_average(reel)"
     ],
}

# All four MBS strength targets share the same blacklist, PLS prefixes and
# fixed features, so they share the same created-variable candidates.
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
        "Starch_uptake__g/m2_"
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
     "Starch_uptake_by_paper_Bottom_Roll__g/m2_":[
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

DEFAULT_FIXED_FEATURES = {
    "Steam__kWh/T_": [ 
        "linepressure_1",
        "Starch_uptake__g/m2_",
        "grammage",
        # Forced in: the Steam window spans winter->summer, so a large part of
        # what the latent level was absorbing is very likely ambient air heating
        # load. It was selectable before but not guaranteed to be picked.
        "ambient_temp_C",
    ],
    "Electricity__kWh/T_": [
        "Speed",
        "grammage"
    ],
     "Starch_uptake_by_paper_Top_Roll__g/m2_": [
        'grammage', 
        'Temperature_starch_working_tank_2', 
        'starch2_1', 
        'starch2_2', 
        'starch2_3'
     ],
     "Starch_uptake_by_paper_Bottom_Roll__g/m2_":[
        'grammage', 
        'Temperature_starch_working_tank_1', 
        'starch1_1', 
        'starch1_2', 
        'starch1_3'
     ],
     "MBS_SCT_CD": [
        "grammage",'conc_starch_1',"delta_basis_weight","Jet/wire_ratio"
     ],
     "MBS_SCT_MD": [
        "grammage",'conc_starch_1',"delta_basis_weight","Jet/wire_ratio"
     ],
     "MBS_Burst": [
        "grammage",'conc_starch_1',"delta_basis_weight","Jet/wire_ratio"
     ],
     "MBS_CMT30": [
        "grammage",'conc_starch_1',"delta_basis_weight","Jet/wire_ratio"
     ],
}

# =============================================================================
# Mediator / tautological variables (setpoint-optimisation mode)
# =============================================================================
# These are NOT operator levers. They are either downstream consequences of the
# process being modelled, or near-restatements of the target:
#
#   * water/dewatering/moisture families  -> outcomes of forming, pressing and
#     drying, so "reduce water to evaporate" is true but not a setpoint
#   * exhaust air humidity                -> the direct signature of evaporation
#   * Steam_*_for_PM                      -> essentially the drying energy input
#   * Production_Rate                     -> derived from speed x width x weight
#
# Including them inflates fit while producing recommendations that cannot be
# acted on. Enable exclusion with --exclude_mediators.
#
# Matched as case-insensitive regex against feature names.
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
    # Not characterised yet for the other targets: the flag is a deliberate
    # no-op there rather than a guess. For the starch targets in particular the
    # target itself is a starch-uptake variable, so the Steam list would be
    # actively wrong.
}

# PLS groups that are built entirely from mediator source columns. Dropping the
# sources without dropping the prefix would leave a degenerate PLS step.
MEDIATOR_PLS_PREFIXES = {
    "Steam__kWh/T_": ["exha_mois"],
}

# Fraction of the series used to fit the supervised PLS projection. Kept as a
# leading block (not all rows) so the projection is not built with knowledge of
# the whole record. See the note where it is used.
PLS_FIT_FRACTION = 0.80
LAGS = 0
# Smoother bandwidth grid in DAYS. Spans "a shift" to "two months" so the sweep
# brackets the plausible separation between actionable variation and drift.
DEFAULT_BANDWIDTHS_DAYS = [0.5, 1, 2, 4, 7, 14, 30, 60]
N_FOLDS = 5


def _is_mediator(name: str, patterns: list[str]) -> bool:
    """True if `name` matches any mediator pattern (case-insensitive)."""
    return any(re.search(p, name, re.I) for p in patterns)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Time-partialling identification of f for setpoint optimisation")
    parser.add_argument("--y_column", required=True, help="Target column name")
    parser.add_argument("--data_path", required=True, help="Path to parquet data")
    parser.add_argument("--apply_ewm_filter", action="store_true",
                        help="Grade-aware EWM on X (removes the noise band)")
    parser.add_argument("--apply_ewm_filter_y", action="store_true",
                        help="Grade-aware EWM on y (removes the noise band)")
    parser.add_argument("--fixed_features", type=str, default=None,
                        help="Comma-separated fixed feature names")
    parser.add_argument("--exclude_mediators", action="store_true",
                        help="Drop downstream/tautological variables (water, dewatering, "
                             "exhaust humidity, Steam_*_for_PM, ...) so the model only "
                             "uses actual operator levers. Required for setpoint "
                             "optimisation; will lower R2.")
    parser.add_argument("--bandwidths", type=str, default=None,
                        help="Comma-separated smoother bandwidths in DAYS")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS,
                        help="Contiguous CV blocks")
    parser.add_argument("--learners", type=str, default="ridge,hist_gbr",
                        help="Comma-separated: ridge,hist_gbr,realmlp")
    parser.add_argument("--variant_b", action="store_true",
                        help="Also run Variant B (flexible f + small fixed time basis)")
    parser.add_argument("--time_basis_columns", type=int, default=5,
                        help="Columns in the Variant B time basis (keep small)")
    parser.add_argument("--nuisance_audit", action="store_true",
                        help="Re-estimate the nuisance per fold to verify coefficients "
                             "do not depend on full-sample partialling")
    args = parser.parse_args()

    Y_COLUMN = args.y_column
    DATA_PATH = args.data_path
    APPLY_EWM_FILTER = args.apply_ewm_filter
    FILTER_Y = args.apply_ewm_filter_y
    EXCLUDE_MEDIATORS = args.exclude_mediators
    N_FOLDS_RUN = args.n_folds

    bandwidths_days = (
        [float(b) for b in args.bandwidths.split(",")]
        if args.bandwidths else list(DEFAULT_BANDWIDTHS_DAYS)
    )
    bandwidths_hours = [b * 24.0 for b in bandwidths_days]
    learners = [s.strip() for s in args.learners.split(",") if s.strip()]
    unknown = [l for l in learners if l not in P.LEARNERS]
    if unknown:
        raise ValueError(f"Unknown learner(s) {unknown}; available: {list(P.LEARNERS)}")

    # Fail fast with a readable message rather than a KeyError deep in the run.
    _required = {
        "PIPELINE_PREFIXES": PIPELINE_PREFIXES,
        "BLACK_LIST": BLACK_LIST,
        "CREATED_VARIABLE_CANDIDATES": CREATED_VARIABLE_CANDIDATES,
        "DEFAULT_FIXED_FEATURES": DEFAULT_FIXED_FEATURES,
        "CONTROL_VARS (config.py)": CONTROL_VARS,
    }
    _missing = [name for name, d in _required.items() if Y_COLUMN not in d]
    if _missing:
        raise KeyError(
            f"Target '{Y_COLUMN}' is not configured in: {', '.join(_missing)}. "
            f"Add an entry for it before running."
        )

    if args.fixed_features:
        fixed_features = [f.strip() for f in args.fixed_features.split(",")]
    else:
        fixed_features = DEFAULT_FIXED_FEATURES[Y_COLUMN]

    # Resolve mediator config for this target
    mediator_patterns = MEDIATOR_PATTERNS.get(Y_COLUMN, []) if EXCLUDE_MEDIATORS else []
    mediator_pls_prefixes = (
        MEDIATOR_PLS_PREFIXES.get(Y_COLUMN, []) if EXCLUDE_MEDIATORS else []
    )
    if EXCLUDE_MEDIATORS and not mediator_patterns:
        print(f"WARNING: --exclude_mediators set but no patterns configured for "
              f"'{Y_COLUMN}'. No variables will be excluded.")

    if mediator_patterns:
        dropped_fixed = [f for f in fixed_features if _is_mediator(f, mediator_patterns)]
        if dropped_fixed:
            print(f"Dropping mediators from fixed_features: {dropped_fixed}")
            fixed_features = [
                f for f in fixed_features if not _is_mediator(f, mediator_patterns)
            ]

    start_time = datetime.now()
    timestamp = start_time.strftime("%y%m%d%H%M")
    out_dir = Path("research_experiments") / Y_COLUMN.replace("/", "_") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # =========================================================================
    # Data Loading & Filtering
    # =========================================================================
    print("\n--- Data Loading & Filtering ---")

    pipeline_prefixes = [
        p for p in PIPELINE_PREFIXES[Y_COLUMN] if p not in mediator_pls_prefixes
    ]
    if mediator_pls_prefixes:
        removed = [p for p in PIPELINE_PREFIXES[Y_COLUMN] if p in mediator_pls_prefixes]
        if removed:
            print(f"Dropping mediator PLS groups: {removed}")
    prep_pip, prep_s_vars = make_prep_pip(prefixes=pipeline_prefixes)

    turnup_data = pd.read_parquet(DATA_PATH)

    ctl_vars = unique_in_order(
        v for v in CONTROL_VARS[Y_COLUMN] if "vacuum" not in v.lower()
    )

    if mediator_patterns:
        n_before_med = len(ctl_vars)
        mediators_found = [v for v in ctl_vars if _is_mediator(v, mediator_patterns)]
        ctl_vars = [v for v in ctl_vars if not _is_mediator(v, mediator_patterns)]
        print(f"Mediator exclusion: {n_before_med} -> {len(ctl_vars)} control vars")
        print(f"  Excluded ({len(mediators_found)}): {mediators_found}")

    # created_vars is intersected with the already-filtered ctl_vars, so it
    # inherits the mediator exclusion.
    created_vars = ordered_intersection(CREATED_VARIABLE_CANDIDATES[Y_COLUMN], ctl_vars)

    turnup_data = _feature_engineering(turnup_data, setpoint_df, steam_null=False, clip=False)
    turnup_data = turnup_data.set_index("Wedge_Time").sort_index()

    if Y_COLUMN== "Steam__kWh/T_":
        #turnup_data = turnup_data[turnup_data.index >   "2026-3-1"]
        turnup_data = turnup_data[turnup_data.index > "2025-11-1"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-11 12:00") & (turnup_data.index < "2026-01-12 11:00"))]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-17 12:00") & (turnup_data.index < "2026-01-19 11:00"))]
        #turnup_data = turnup_data[turnup_data.index < "2026-07-5"]
        turnup_data = turnup_data[
            #(turnup_data["Condensate_energy_from_paper_plant_to_power_plant"].between(5, 10))
            #& 
            (turnup_data["DG4_Temperature_Inlet_Air"] > 100)
            & 
            (turnup_data["Vacuum_Zone_1_PickUp"] < -0.5)
        ]
    if Y_COLUMN== "Electricity__kWh/T_":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data["Vacuum_Zone_1_PickUp"]<-0.5]
        turnup_data = turnup_data[turnup_data.index>"2025-11-15"]

    if Y_COLUMN== "Starch_uptake_by_paper_Top_Roll__g/m2_":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data["Vacuum_Zone_1_PickUp"]<-0.5]
        turnup_data = turnup_data[turnup_data.index>"2026-3-1"]
    
    if Y_COLUMN== "Starch_uptake_by_paper_Bottom_Roll__g/m2_":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data["Vacuum_Zone_1_PickUp"]<-0.5]
        turnup_data = turnup_data[turnup_data.index>"2026-3-1"]

    if Y_COLUMN== "MBS_SCT_CD":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]

    if Y_COLUMN== "MBS_SCT_MD":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]    
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]
    
    if Y_COLUMN== "MBS_Burst":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]

    if Y_COLUMN== "MBS_CMT30":
        turnup_data = turnup_data[turnup_data.index>"2025-04-01 00:00:00"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]  
        turnup_data = turnup_data[turnup_data.index>"2026-02-01"]
        turnup_data = turnup_data[(turnup_data["AB_Grade_ID"]=="6010120") |  (turnup_data["AB_Grade_ID"]=="6010100")]

    turnup_data = turnup_data[turnup_data.grammage.isin([115, 120, 100,  90,  85, 110])]

    print(f"Filtered data: {turnup_data.shape}")

    # =========================================================================
    # Build Preprocessing Pipeline
    # =========================================================================
    print("\n--- Build Preprocessing Pipeline ---")

    steam_pressure = [v for v in turnup_data.columns if re.search(r"cylinder.*steam_pressure", v, re.I)]
    steam_diff_pressure = [v for v in turnup_data.columns if re.search(r"cylinder.*differential_pressure", v, re.I)]

    exog_vars_reduced = [
        v for v in ctl_vars
        if (v not in BLACK_LIST[Y_COLUMN] and v not in created_vars
            and v not in steam_pressure and v not in steam_diff_pressure
            and "vacuum" not in v.lower())
    ]
    exog_vars_reduced = unique_in_order(fixed_features + exog_vars_reduced + ["grammage"])

    for _, step in prep_pip.steps:
        if isinstance(step, GroupwisePLSTransformer):
            transformed_names = [f"{step.score_prefix}_{i}" for i in range(1, step.n_components + 1)]
            exog_vars_reduced = ordered_difference(
                unique_in_order(exog_vars_reduced + transformed_names),
                list(step.pls_columns),
            )
    exog_vars_reduced = unique_in_order(fixed_features + exog_vars_reduced)

    pre_estimator, feat_list = build_pre_estimator(
        exog_vars_reduced=exog_vars_reduced,
        prep_pip=prep_pip,
        created_vars=created_vars,
        apply_ewm=APPLY_EWM_FILTER,
    )
    print(f"Pipeline input: {len(feat_list)} columns")

    # =========================================================================
    # Transform & Build Design Matrix
    # =========================================================================
    print("\n--- Transform & Build Design Matrix ---")

    turnup_ts = turnup_data.copy().sort_index()

    # Drop every row we are going to drop BEFORE computing any split, so the
    # split is a true fraction of the surviving rows.
    n_before = len(turnup_ts)
    turnup_ts = turnup_ts.dropna(subset=[Y_COLUMN])
    n_after_y = len(turnup_ts)
    turnup_ts = turnup_ts.dropna(subset=feat_list)
    n_after_feat = len(turnup_ts)
    print(f"Rows: {n_before} -> {n_after_y} (target NaN) -> {n_after_feat} (feature NaN)")
    if n_after_feat < 100:
        raise RuntimeError(f"Only {n_after_feat} rows survive filtering; cannot model.")

    pls_fit_n = int(len(turnup_ts) * PLS_FIT_FRACTION)
    ts_raw = turnup_ts.loc[:, feat_list]

    # NOTE ON THE PLS PROJECTION
    # The prep pipeline contains supervised PLS, so it is fitted on a leading
    # block rather than on everything. It is a FIXED low-dimensional projection,
    # not the model under study, but folds whose validation window falls inside
    # that leading block do see a basis built with their targets. Recorded in
    # results.json so the caveat travels with the numbers.
    pre_estimator.fit(ts_raw.iloc[:pls_fit_n], turnup_ts[Y_COLUMN].iloc[:pls_fit_n])
    ts_transformed = pre_estimator.transform(ts_raw)
    print(f"Transformed shape: {ts_transformed.shape}")

    y_raw_vals = turnup_ts[Y_COLUMN].values.astype(float).copy()
    ts_transformed[Y_COLUMN] = y_raw_vals

    # Grade-aware EWM on the target: removes the noise band, which is the other
    # half of the band-pass (time-partialling removes the drift band).
    if FILTER_Y:
        from utility import ewm_reset
        grammage_group = pd.Series(turnup_ts["grammage"].values)
        grade_change = grammage_group.ne(grammage_group.shift())
        time_gap = ts_transformed.index.to_series().diff().gt(pd.Timedelta("12h"))
        time_gap.iloc[0] = True
        seg = (grade_change.values | time_gap.values).cumsum()
        y_series = pd.Series(y_raw_vals, index=ts_transformed.index)
        ts_transformed[Y_COLUMN] = y_series.groupby(seg).transform(ewm_reset).values

    transformed_feature_names = [c for c in ts_transformed.columns if c != Y_COLUMN]
    X, y = make_design(ts_transformed, Y_COLUMN, transformed_feature_names, None,
                       y_lags=range(1, 1 + LAGS))
    y = np.asarray(y, dtype=float).ravel()
    print(f"Design matrix: X={X.shape}, y={y.shape}")

    y_raw_aligned = pd.Series(y_raw_vals, index=ts_transformed.index).loc[X.index].values

    t_index = pd.DatetimeIndex(X.index)
    if not t_index.is_monotonic_increasing:
        raise RuntimeError("Design matrix index is not sorted in time.")
    t_hours = P._to_hours(t_index)
    feature_names = list(X.columns)
    span_days = (t_hours[-1] - t_hours[0]) / 24.0
    print(f"Span: {span_days:.1f} days, {len(X)} rows, {len(feature_names)} features")

    # =========================================================================
    # Variance decomposition by frequency band
    # =========================================================================
    # Where does the target's variance actually live? This sizes the prize before
    # any modelling: only the mid band is both real and actionable.
    print("\n--- Variance decomposition ---")
    var_raw = float(np.var(y_raw_aligned))
    noise_var = float(np.var(y_raw_aligned - y)) if FILTER_Y else 0.0

    band_rows = []
    for bw_h, bw_d in zip(bandwidths_hours, bandwidths_days):
        slow = P.gaussian_time_smooth(t_hours, y, bw_h)
        mid = y - slow
        band_rows.append({
            "bandwidth_days": bw_d,
            "var_slow": float(np.var(slow)),
            "var_mid": float(np.var(mid)),
            "share_slow": float(np.var(slow) / var_raw) if var_raw else 0.0,
            "share_mid": float(np.var(mid) / var_raw) if var_raw else 0.0,
            "std_mid": float(np.std(mid)),
        })
    band_df = pd.DataFrame(band_rows)
    print(f"  var(y_raw)={var_raw:.1f}  std={np.sqrt(var_raw):.2f}")
    if FILTER_Y:
        print(f"  noise band (y_raw - EWM(y)): var={noise_var:.1f} "
              f"({noise_var/var_raw:.1%} of total), std={np.sqrt(noise_var):.2f}")
    print(band_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # =========================================================================
    # Bandwidth sweep (Variant A, ridge)
    # =========================================================================
    # The central diagnostic. Coefficients that hold steady across bandwidths are
    # identified from mid-band variation; coefficients that swing were borrowing
    # from the trend. R2 here answers "at which timescale is X most informative
    # about y" - the target changes with bandwidth, so it is not a model contest.
    print("\n--- Bandwidth sweep (Variant A, ridge) ---")
    sweep = P.bandwidth_sweep(
        X, y, t_hours, bandwidths_hours,
        learner="ridge", variant="A", n_folds=N_FOLDS_RUN, verbose=True,
    )
    sweep_df = P.sweep_frame(sweep)
    paths_df = P.coefficient_paths(sweep)

    best = max(sweep, key=lambda r: r.r2_mean)
    best_bw_h = best.bandwidth_hours
    print(f"\n  Best bandwidth: {best_bw_h/24:.2f} days "
          f"(R2(y~)={best.r2_mean:+.4f}, smoother dof={best.smoother_dof:.1f})")

    # =========================================================================
    # Learner comparison at the best bandwidth
    # =========================================================================
    # Now a fair contest: identical target, identical folds, nuisance already
    # removed so nothing competes with the learner for the slow variance. This is
    # the comparison that was never possible under backfitting.
    print("\n--- Learner comparison ---")
    comparisons: list[P.EvalResult] = []
    variants = ["A"] + (["B"] if args.variant_b else [])
    for variant in variants:
        for learner in learners:
            try:
                res = P.evaluate(
                    X, y, t_hours, bandwidth_hours=best_bw_h,
                    learner=learner, variant=variant, n_folds=N_FOLDS_RUN,
                    time_basis_columns=args.time_basis_columns,
                )
            except Exception as exc:
                print(f"  variant {variant} / {learner:9s} FAILED: {exc}")
                continue
            comparisons.append(res)
            print(f"  variant {variant} / {learner:9s} "
                  f"R2(y~)={res.r2_mean:+.4f} +-{res.r2_std:.4f}  "
                  f"rmse={res.rmse_mean:7.3f}")

    comparison_df = P.sweep_frame(comparisons)

    # =========================================================================
    # Coefficients at the best bandwidth
    # =========================================================================
    print("\n--- Coefficients (Variant A, ridge, standardised units) ---")
    best_ridge = next(
        (r for r in comparisons if r.variant == "A" and r.learner == "ridge"), best
    )
    coef_df = best_ridge.coef_frame()
    if not coef_df.empty:
        print(coef_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        unstable = coef_df.loc[coef_df["sign_flips"] > 0, "feature"].tolist()
        if unstable:
            print(f"\n  NOT identified (sign flips across folds): {unstable}")
        else:
            print("\n  No sign flips across folds.")

    # Optional audit: does the nuisance having seen all rows change anything?
    audit_df = pd.DataFrame()
    if args.nuisance_audit:
        print("\n--- Nuisance audit (per-fold partialling) ---")
        print("  Read the COEFFICIENTS, not the R2: a validation block is wide")
        print("  relative to the kernel, so y_tilde there retains drift.")
        audit = P.evaluate(
            X, y, t_hours, bandwidth_hours=best_bw_h,
            learner="ridge", variant="A", n_folds=N_FOLDS_RUN,
            nuisance_fit="fold",
        )
        audit_df = audit.coef_frame()
        if not audit_df.empty and not coef_df.empty:
            merged = coef_df[["feature", "coef_mean"]].merge(
                audit_df[["feature", "coef_mean"]], on="feature",
                suffixes=("_full", "_fold"),
            )
            merged["abs_diff"] = (merged["coef_mean_full"] - merged["coef_mean_fold"]).abs()
            print(merged.sort_values("abs_diff", ascending=False)
                  .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
            print(f"  max |difference| = {merged['abs_diff'].max():.4f}  "
                  f"(small => full-sample partialling is not distorting f)")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\n--- Plots ---")

    # (1) Bandwidth sweep
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.errorbar(sweep_df["bandwidth_days"], sweep_df["r2_mean"],
                 yerr=sweep_df["r2_std"], marker="o", capsize=3, label="R2(y~)")
    ax1.axvline(best_bw_h / 24, ls="--", color="C3", alpha=0.7,
                label=f"best = {best_bw_h/24:.2f}d")
    ax1.set_xscale("log")
    ax1.set_xlabel("Smoother bandwidth (days, log scale)")
    ax1.set_ylabel("R2 on partialled target")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(sweep_df["bandwidth_days"], sweep_df["share_of_y_removed"],
             marker="s", color="C2", alpha=0.6, label="share of y removed as trend")
    ax2.set_ylabel("Share of y variance assigned to trend")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    ax1.set_title(f"{Y_COLUMN} - bandwidth sweep (Variant A, ridge)")
    plt.tight_layout()
    plt.savefig(out_dir / "bandwidth_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (2) Coefficient paths for the most important features
    if not paths_df.empty and not coef_df.empty:
        top = coef_df.head(12)["feature"].tolist()
        fig, ax = plt.subplots(figsize=(11, 6))
        for feat in top:
            sub = paths_df[paths_df["feature"] == feat].sort_values("bandwidth_days")
            ax.plot(sub["bandwidth_days"], sub["coef_mean"], marker="o", ms=3, label=feat)
        ax.axhline(0, color="gray", ls="--", alpha=0.6)
        ax.axvline(best_bw_h / 24, ls="--", color="C3", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Smoother bandwidth (days, log scale)")
        ax.set_ylabel("Coefficient (standardised)")
        ax.set_title(f"{Y_COLUMN} - coefficient paths (flat = identified)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "coefficient_paths.png", dpi=150, bbox_inches="tight")
        plt.close()

    # (3) Band decomposition of the target at the best bandwidth
    slow_best = P.gaussian_time_smooth(t_hours, y, best_bw_h)
    mid_best = y - slow_best
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t_index, y_raw_aligned, lw=0.4, alpha=0.45, label="y raw")
    if FILTER_Y:
        axes[0].plot(t_index, y, lw=0.7, alpha=0.8, label="EWM(y) (noise band removed)")
    axes[0].plot(t_index, slow_best, lw=2.0, color="C3",
                 label=f"slow s(t), bw={best_bw_h/24:.2f}d")
    axes[0].set_ylabel(Y_COLUMN)
    axes[0].set_title(f"{Y_COLUMN} - band decomposition")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(t_index, mid_best, lw=0.6, color="C0")
    axes[1].axhline(0, color="gray", ls="--", alpha=0.6)
    axes[1].set_ylabel("mid band (target for f)")
    axes[1].set_title(f"Mid band: std={np.std(mid_best):.2f} "
                      f"({np.var(mid_best)/var_raw:.1%} of raw variance) - this is what f can explain")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "band_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()

    # (4) Learner comparison
    if not comparison_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [f"{r.variant}/{r.learner}" for r in comparisons]
        vals = [r.r2_mean for r in comparisons]
        errs = [r.r2_std for r in comparisons]
        ax.bar(labels, vals, yerr=errs, capsize=4, alpha=0.8,
               color=["C0" if r.variant == "A" else "C1" for r in comparisons])
        ax.set_ylabel("R2 on partialled target")
        ax.set_title(f"{Y_COLUMN} - learner comparison at bw={best_bw_h/24:.2f}d")
        ax.grid(alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "learner_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # (5) Observed vs predicted on the partialled target (best ridge, held-out folds)
    pdat_best = P.partial_out_time(X, y, t_hours, best_bw_h)
    oof = np.full(len(y), np.nan)
    make_ridge = P.LEARNERS["ridge"]()
    for tr, va in P.contiguous_blocks(len(y), N_FOLDS_RUN):
        m = make_ridge()
        m.fit(pdat_best.X_tilde[tr], pdat_best.y_tilde[tr])
        oof[va] = m.predict(pdat_best.X_tilde[va])
    ok = ~np.isnan(oof)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(t_index[ok], pdat_best.y_tilde[ok], lw=0.6, label="y~ actual")
    axes[0].plot(t_index[ok], oof[ok], lw=0.8, alpha=0.8, label="y~ predicted (out-of-fold)")
    axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_title("Partialled target, out-of-fold prediction")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].scatter(pdat_best.y_tilde[ok], oof[ok], s=8, alpha=0.3)
    lim = [min(pdat_best.y_tilde[ok].min(), oof[ok].min()),
           max(pdat_best.y_tilde[ok].max(), oof[ok].max())]
    axes[1].plot(lim, lim, "r--", lw=1.5)
    axes[1].set_xlabel("y~ actual"); axes[1].set_ylabel("y~ predicted")
    from sklearn.metrics import r2_score as _r2
    axes[1].set_title(f"Out-of-fold R2 = {_r2(pdat_best.y_tilde[ok], oof[ok]):.4f}")
    axes[1].set_aspect("equal", adjustable="box"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "partialled_prediction.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Save
    # =========================================================================
    print("--- Saving ---")
    import shutil

    sweep_df.to_csv(out_dir / "bandwidth_sweep.csv", index=False)
    paths_df.to_csv(out_dir / "coefficient_paths.csv", index=False)
    band_df.to_csv(out_dir / "band_decomposition.csv", index=False)
    if not coef_df.empty:
        coef_df.to_csv(out_dir / "coefficients.csv", index=False)
    if not comparison_df.empty:
        comparison_df.to_csv(out_dir / "learner_comparison.csv", index=False)
    if not audit_df.empty:
        audit_df.to_csv(out_dir / "nuisance_audit_coefficients.csv", index=False)
    turnup_ts.to_parquet(out_dir / "turnup_ts.parquet")

    code_dir = out_dir / "code"
    code_dir.mkdir(exist_ok=True)
    for src_file in [Path(__file__).name, "partialling_research.py", "config.py",
                     "data_cleaning.py", "preprocessing.py"]:
        src_path = Path(__file__).parent / src_file
        if src_path.exists():
            shutil.copy2(src_path, code_dir / src_file)

    end_time = datetime.now()
    results_json = {
        "approach": "time-partialling (Robinson/DML): remove E[.|t] from y and X "
                    "with the same fixed smoother, then fit f",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "data_start": str(t_index[0]),
        "data_end": str(t_index[-1]),
        "n_rows": int(len(X)),
        "span_days": float(span_days),
        "parameters": {
            "y_column": Y_COLUMN,
            "data_path": DATA_PATH,
            "apply_ewm_filter": APPLY_EWM_FILTER,
            "apply_ewm_filter_y": FILTER_Y,
            "exclude_mediators": EXCLUDE_MEDIATORS,
            "fixed_features": fixed_features,
            "bandwidths_days": bandwidths_days,
            "n_folds": N_FOLDS_RUN,
            "learners": learners,
            "variant_b": args.variant_b,
            "time_basis_columns": args.time_basis_columns,
            "pls_fit_fraction": PLS_FIT_FRACTION,
            "pls_fit_rows": int(pls_fit_n),
        },
        "caveats": [
            "The supervised PLS projection is fitted on the leading "
            f"{PLS_FIT_FRACTION:.0%} of rows, so folds inside that block see a "
            "basis built with their own targets. It is a fixed low-dimensional "
            "projection, not the model under study.",
            "R2 is measured on the partialled target y~, which CHANGES with "
            "bandwidth. Compare learners at fixed bandwidth; read the sweep as "
            "'at which timescale is X informative', not as a model contest.",
        ],
        "variance_decomposition": {
            "var_y_raw": var_raw,
            "std_y_raw": float(np.sqrt(var_raw)),
            "var_noise_band": noise_var,
            "share_noise_band": float(noise_var / var_raw) if var_raw else 0.0,
            "by_bandwidth": band_rows,
        },
        "best_bandwidth_days": float(best_bw_h / 24.0),
        "best_bandwidth_smoother_dof": float(best.smoother_dof),
        "sweep": sweep_df.to_dict(orient="records"),
        "learner_comparison": comparison_df.to_dict(orient="records"),
        "coefficients": coef_df.to_dict(orient="records") if not coef_df.empty else [],
        "all_features": feature_names,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=float)

    artifact = {
        "approach": "time_partialling",
        "pre_estimator": pre_estimator,
        "feat_list": feat_list,
        "feature_names": feature_names,
        "y_column": Y_COLUMN,
        "best_bandwidth_hours": float(best_bw_h),
        "coefficients": coef_df,
        "sweep": sweep_df,
        "band_decomposition": band_df,
    }
    with open(out_dir / "model.pkl", "wb") as f:
        cloudpickle.dump(artifact, f)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
