"""
Run iterative backfitting experiment (CMA-ES + Ridge + Local Level).

Usage:
    python run_experiment.py --y_column "Steam__kWh/T_" --data_path ./data/costimier_turnup.parquet

Parameters:
    --y_column          Target column name
    --data_path         Path to parquet data
    --apply_ewm_filter  Enable EWM filtering (flag)
    --n_iterations      Max backfit iterations (default 10)
    --gamma             Level damping (default 1.0)
    --splines           Enable splines after first iteration (flag)
    --fixed_features    Comma-separated fixed feature names (optional)
"""

import sys
from pathlib import Path

# Add parent directory (where utility.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import re
from datetime import datetime

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from config import CONTROL_VARS
from data_cleaning import (
    unique_in_order, ordered_difference, ordered_intersection,
    outlier, make_design,
)
from preprocessing import make_prep_pip, build_pre_estimator, build_final_pipeline
from feature_selection import cmaes_feature_selection, fit_all_features
from state_estimation import iterative_backfit
from utility import GroupwisePLSTransformer, _feature_engineering, setpoint_df


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

TEST_SIZE = 0.20
# Fraction of the TRAIN block reserved for feature selection scoring, so the
# real test block is never used to choose features.
VAL_SIZE = 0.25
LAGS = 0
# Contiguous train blocks used for the coefficient-stability sweep.
COEF_STABILITY_BLOCKS = 5


def _is_mediator(name: str, patterns: list[str]) -> bool:
    """True if `name` matches any mediator pattern (case-insensitive)."""
    return any(re.search(p, name, re.I) for p in patterns)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run backfitting experiment")
    parser.add_argument("--y_column", required=True, help="Target column name")
    parser.add_argument("--data_path", required=True, help="Path to parquet data")
    parser.add_argument("--apply_ewm_filter", action="store_true", help="Enable EWM filtering")
    parser.add_argument("--apply_ewm_filter_y", action="store_true", help="Apply EWM filter to target (train/fit only, metrics use raw)")
    parser.add_argument("--n_iterations", type=int, default=10, help="Max backfit iterations")
    parser.add_argument("--gamma", type=float, default=1.0, help="Level damping factor")
    parser.add_argument("--splines", action="store_true", help="Enable splines after first iteration")
    parser.add_argument("--fixed_features", type=str, default=None,
                        help="Comma-separated fixed feature names")
    parser.add_argument("--exclude_mediators", action="store_true",
                        help="Drop downstream/tautological variables (water, dewatering, "
                             "exhaust humidity, Steam_*_for_PM, ...) so the model only "
                             "uses actual operator levers. Required for setpoint "
                             "optimisation; will lower R2.")
    args = parser.parse_args()

    Y_COLUMN = args.y_column
    DATA_PATH = args.data_path
    APPLY_EWM_FILTER = args.apply_ewm_filter
    FILTER_Y = args.apply_ewm_filter_y
    N_ITERATIONS = args.n_iterations
    GAMMA = args.gamma
    SPLINES = args.splines

    EXCLUDE_MEDIATORS = args.exclude_mediators

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

    # Fixed features are forced into every candidate subset, so a mediator left
    # in here would defeat the exclusion entirely.
    if mediator_patterns:
        dropped_fixed = [f for f in fixed_features if _is_mediator(f, mediator_patterns)]
        if dropped_fixed:
            print(f"Dropping mediators from fixed_features: {dropped_fixed}")
            fixed_features = [
                f for f in fixed_features if not _is_mediator(f, mediator_patterns)
            ]

    # Timing
    start_time = datetime.now()

    # Output directory
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

    # IMPORTANT: drop every row we are going to drop BEFORE computing the split.
    # Computing `split` first and applying it after a dropna makes `split` a
    # larger-than-intended fraction of the surviving rows (and can exceed the
    # frame length entirely), which fits the supervised PLS inside
    # `pre_estimator` on test-period targets. That is target leakage.
    print("feat_list", feat_list)
    n_before = len(turnup_ts)
    turnup_ts = turnup_ts.dropna(subset=[Y_COLUMN])
    n_after_y = len(turnup_ts)
    turnup_ts = turnup_ts.dropna(subset=feat_list)
    n_after_feat = len(turnup_ts)
    print(f"Rows: {n_before} -> {n_after_y} (target NaN) -> {n_after_feat} (feature NaN)")

    # Split is derived from the FINAL row count only.
    n_samples = len(turnup_ts)
    split = int(n_samples * (1.0 - TEST_SIZE))

    ts_raw = turnup_ts.loc[:, feat_list]

    pre_estimator.fit(ts_raw.iloc[:split], turnup_ts[Y_COLUMN].iloc[:split])
    ts_transformed = pre_estimator.transform(ts_raw)

    print(f"Transformed shape: {ts_transformed.shape}")
    print(f"Transformed columns: {list(ts_transformed.columns)}")

    ts_transformed[Y_COLUMN] = turnup_ts[Y_COLUMN].values

    # Optionally EWM-filter the target (for training/fitting only)
    if FILTER_Y:
        from utility import ewm_reset
        grammage_group = turnup_ts["grammage"].values[:len(ts_transformed)]
        grade_change = pd.Series(grammage_group).ne(pd.Series(grammage_group).shift())
        time_gap = ts_transformed.index.to_series().diff().gt(pd.Timedelta("12h"))
        time_gap.iloc[0] = True
        seg = (grade_change.values | time_gap.values).cumsum()
        y_series = pd.Series(turnup_ts[Y_COLUMN].values[:len(ts_transformed)], index=ts_transformed.index)
        ts_transformed[Y_COLUMN] = y_series.groupby(seg).transform(ewm_reset).values

    # Keep raw y for metrics
    y_raw = turnup_ts[Y_COLUMN].values[:len(ts_transformed)]

    transformed_feature_names = [c for c in ts_transformed.columns if c != Y_COLUMN]
    X, y = make_design(ts_transformed, Y_COLUMN, transformed_feature_names, None, y_lags=range(1, 1 + LAGS))
    print(f"Design matrix: X={X.shape}, y={y.shape}")

    # =========================================================================
    # Train/Test Split
    # =========================================================================
    n_samples = len(X)
    split_design = int(n_samples * (1.0 - TEST_SIZE))

    # The preprocessing pipeline was fitted on rows [:split] of ts_transformed.
    # If make_design dropped rows (lags), the design-matrix boundary would no
    # longer coincide with it and the pipeline would have seen test rows.
    if split_design != split:
        raise RuntimeError(
            f"Split mismatch: pipeline fitted on {split} rows but design matrix "
            f"split is {split_design} (X has {n_samples} rows vs "
            f"{len(ts_transformed)} transformed). Rows were dropped after the "
            f"pipeline was fitted - this would leak test data into the PLS fit."
        )
    split = split_design

    Xtr, Xte = X.iloc[:split].copy(), X.iloc[split:].copy()
    ytr, yte = y.iloc[:split].copy(), y.iloc[split:].copy()

    # Feature selection must NOT see the test block. Carve an inner validation
    # split out of the training block instead. `outer_cv` previously pointed at
    # the real test set, so every candidate subset was ranked by test RMSE and
    # the reported metrics were optimistically biased.
    inner_split = int(split * (1.0 - VAL_SIZE))
    inner_cv = [(np.arange(inner_split), np.arange(inner_split, split))]

    # Raw y for metrics (unfiltered, aligned with design matrix index)
    y_raw_series = pd.Series(y_raw, index=ts_transformed.index, name=Y_COLUMN)
    y_raw_aligned = y_raw_series.loc[X.index]
    yte_raw = y_raw_aligned.iloc[split:].values
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")
    print(f"Inner selection split: fit[:{inner_split}] / score[{inner_split}:{split}] "
          f"(test block {split}: held out)")

    # =========================================================================
    # Feature Selection Setup
    # =========================================================================
    fixed_features_resolved = (
        [v for v in fixed_features if "grammage" not in v.lower()]
        + [v for v in X.columns if "grammage" in v.lower()]
    )

    feature_groups = {
        step.score_prefix: [f"{step.score_prefix}_{i}" for i in range(1, step.n_components + 1)]
        for _, step in prep_pip.steps
        if isinstance(step, GroupwisePLSTransformer)
    }

    def run_feature_selection(X_full, y_full, iteration, splines=False):
        return cmaes_feature_selection(
            X_full, y_full,
            k_range=(3, 15),
            fixed_features=fixed_features_resolved,
            feature_groups=feature_groups,
            cv_splits=inner_cv,
            selection="topk",
            max_evals=3000,
            sigma0=1.0,
            seed=42,
            popsize=24,
            penalty_fn=None,
            splines=splines,
            verbose=False,
        )

    # =========================================================================
    # Iterative Backfitting
    # =========================================================================
    print("\n--- Iterative Backfitting ---")

    backfit_result = iterative_backfit(
        X=X, y=y,
        train_idx=np.arange(split),
        test_idx=np.arange(split, n_samples),
        feature_selection_fn=run_feature_selection,
        n_iterations=N_ITERATIONS,
        patience=2,
        state_estimator_kwargs={"level": True},
        splines=SPLINES,
        gamma=GAMMA,
        verbose=True,
    )

    history = backfit_result["iteration_history"]

    # =========================================================================
    # Metrics (re-estimate level on raw residuals if y was filtered)
    # =========================================================================
    y_actual = yte_raw  # always use raw y for metrics
    y_pred_ridge = backfit_result["y_test_pred"].values

    # Re-estimate level using raw residuals for honest combined metric
    if FILTER_Y:
        from state_estimation import ResidualStateEstimator

        # Raw train residuals
        y_raw_train = y_raw_series.loc[X.index].iloc[:split].values
        raw_residuals_train = y_raw_train - backfit_result["y_train_pred"].values
        raw_residuals_train_series = pd.Series(raw_residuals_train, index=Xtr.index)

        # Fit level on raw train residuals
        state_est_raw = ResidualStateEstimator(level=True)
        state_est_raw.fit(raw_residuals_train_series)

        # Raw test residuals -> update level
        raw_residuals_test = y_actual - y_pred_ridge
        raw_residuals_test_series = pd.Series(raw_residuals_test, index=Xte.index)
        state_result_raw = state_est_raw.update(raw_residuals_test_series)

        n_test = len(Xte)
        level_test_raw = state_result_raw.level.iloc[-n_test:].values
        y_pred_combined = y_pred_ridge + level_test_raw
        print("(Level re-estimated on raw residuals for metrics)")
    else:
        y_pred_combined = backfit_result["y_test_combined"].values

    metrics_ridge = {
        "rmse": float(np.sqrt(mean_squared_error(y_actual, y_pred_ridge))),
        "mae": float(mean_absolute_error(y_actual, y_pred_ridge)),
        "r2": float(r2_score(y_actual, y_pred_ridge)),
    }
    metrics_combined = {
        "rmse": float(np.sqrt(mean_squared_error(y_actual, y_pred_combined))),
        "mae": float(mean_absolute_error(y_actual, y_pred_combined)),
        "r2": float(r2_score(y_actual, y_pred_combined)),
    }

    print(f"\nRidge only:  RMSE={metrics_ridge['rmse']:.2f}, MAE={metrics_ridge['mae']:.2f}, R2={metrics_ridge['r2']:.4f}")
    print(f"Ridge+Level: RMSE={metrics_combined['rmse']:.2f}, MAE={metrics_combined['mae']:.2f}, R2={metrics_combined['r2']:.4f}")

    # =========================================================================
    # Diagnostics for setpoint optimisation
    # =========================================================================
    # For optimisation the deliverable is the GRADIENT of f, not RMSE. These
    # diagnostics ask whether each lever's coefficient is a real effect or an
    # artefact of the latent trend.
    print("\n--- Diagnostics ---")

    selected = backfit_result["selected_features"]
    level_train_vals = backfit_result["level_train"].values
    level_test_vals = backfit_result["level_test"].values
    y_train_pred_vals = backfit_result["y_train_pred"].values
    y_raw_train = y_raw_aligned.iloc[:split].values

    # --- (a) Component attribution -----------------------------------------
    # If the regression component and the level component are strongly
    # correlated, the level is absorbing regression signal (or vice versa).
    # That correlation is the most direct evidence of the identifiability
    # problem, and it should be near zero if the split is clean.
    corr_train = float(np.corrcoef(y_train_pred_vals, level_train_vals)[0, 1])
    corr_test = float(np.corrcoef(y_pred_ridge, level_test_vals)[0, 1])

    component_attribution = {
        "std_target_test": float(np.std(y_actual)),
        "std_regression_test": float(np.std(y_pred_ridge)),
        "std_level_test": float(np.std(level_test_vals)),
        "std_residual_test": float(np.std(y_actual - y_pred_combined)),
        "corr_regression_level_train": corr_train,
        "corr_regression_level_test": corr_test,
        "r2_regression_only": metrics_ridge["r2"],
        "r2_with_level": metrics_combined["r2"],
        "r2_gain_from_level": metrics_combined["r2"] - metrics_ridge["r2"],
    }
    print(f"  Component std (test): target={component_attribution['std_target_test']:.2f}, "
          f"regression={component_attribution['std_regression_test']:.2f}, "
          f"level={component_attribution['std_level_test']:.2f}, "
          f"residual={component_attribution['std_residual_test']:.2f}")
    print(f"  corr(regression, level): train={corr_train:+.3f}, test={corr_test:+.3f}"
          f"   <- large |corr| means the components are competing")

    # --- (b) Interpretable linear coefficients ------------------------------
    # Fitted on standardised features against the DE-TRENDED train target, i.e.
    # exactly the target the regression component is responsible for. This is a
    # plain linear proxy even when the main model used splines, because spline
    # basis coefficients do not map one-to-one onto levers.
    y_detrended_train = y_raw_train - level_train_vals

    scaler_diag = StandardScaler()
    Xtr_sel_std = scaler_diag.fit_transform(Xtr[selected].values)
    ridge_diag = RidgeCV(alphas=np.logspace(0, 3, 20))
    ridge_diag.fit(Xtr_sel_std, y_detrended_train)
    coefs_full = ridge_diag.coef_

    # --- (c) Coefficient stability across contiguous train blocks ----------
    # Refit on each block. A coefficient that changes sign or swings widely is
    # not identified: it is picking up whatever the trend was doing locally.
    n_blocks = COEF_STABILITY_BLOCKS
    block_edges = np.linspace(0, split, n_blocks + 1).astype(int)
    block_coefs = []
    for b in range(n_blocks):
        lo, hi = block_edges[b], block_edges[b + 1]
        if hi - lo < max(10, 2 * len(selected)):
            continue  # block too small to fit meaningfully
        sc_b = StandardScaler()
        Xb = sc_b.fit_transform(Xtr[selected].values[lo:hi])
        rb = RidgeCV(alphas=np.logspace(0, 3, 20))
        rb.fit(Xb, y_detrended_train[lo:hi])
        block_coefs.append(rb.coef_)

    coef_table = []
    if block_coefs:
        block_coefs_arr = np.vstack(block_coefs)
        for i, feat in enumerate(selected):
            per_block = block_coefs_arr[:, i]
            full = float(coefs_full[i])
            sign_flips = int(np.sum(np.sign(per_block) != np.sign(full)))
            coef_table.append({
                "feature": feat,
                "coef_full_train": full,
                "coef_block_mean": float(np.mean(per_block)),
                "coef_block_std": float(np.std(per_block)),
                "coef_block_min": float(np.min(per_block)),
                "coef_block_max": float(np.max(per_block)),
                "sign_flips": sign_flips,
                "n_blocks": int(len(block_coefs)),
                # |mean| / std across blocks: high = stable, <1 = unreliable
                "stability_ratio": (
                    float(abs(np.mean(per_block)) / np.std(per_block))
                    if np.std(per_block) > 0 else float("inf")
                ),
            })
    else:
        for i, feat in enumerate(selected):
            coef_table.append({
                "feature": feat,
                "coef_full_train": float(coefs_full[i]),
                "n_blocks": 0,
            })

    # --- (d) Collinearity of each lever with the estimated trend ------------
    # High |corr| with the level means this coefficient is the one most at risk
    # of having been absorbed, and the one to distrust most.
    for row in coef_table:
        col = Xtr[row["feature"]].values
        if np.std(col) > 0 and np.std(level_train_vals) > 0:
            row["corr_with_level"] = float(
                np.corrcoef(col, level_train_vals)[0, 1]
            )
        else:
            row["corr_with_level"] = 0.0

    coef_df = pd.DataFrame(coef_table)
    if "stability_ratio" in coef_df.columns:
        coef_df = coef_df.sort_values("stability_ratio", ascending=False)

    print(f"\n  Coefficient stability across {len(block_coefs)} contiguous train blocks")
    print(f"  (standardised units, target = y - level; sorted most to least stable)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(coef_df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    if "sign_flips" in coef_df.columns:
        unstable = coef_df[coef_df["sign_flips"] > 0]["feature"].tolist()
        if unstable:
            print(f"\n  NOT SAFE to optimise over (sign flips across blocks): {unstable}")
        else:
            print("\n  No sign flips across blocks.")

    coef_df.to_csv(out_dir / "coefficient_stability.csv", index=False)

    diagnostics = {
        "component_attribution": component_attribution,
        "coefficient_stability": coef_table,
        "n_stability_blocks": int(len(block_coefs)),
        "diagnostic_alpha": float(ridge_diag.alpha_),
    }

    # =========================================================================
    # Plots
    # =========================================================================
    iters = [h["iteration"] for h in history]

    # Convergence plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(iters, [h["rmse_ridge"] for h in history], "o-", label="Ridge")
    axes[0].plot(iters, [h["rmse_combined"] for h in history], "s-", label="Ridge+Level")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("RMSE")
    axes[0].set_title("Test RMSE per Iteration"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(iters, [h["level_scale"] for h in history], "o-", color="C2",
                 label="level scale")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Level scale")
    axes[1].set_title("Latent Level Flexibility per Iteration")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(iters, [h["r2_ridge"] for h in history], "o-", label="Ridge")
    axes[2].plot(iters, [h["r2_combined"] for h in history], "s-", label="Ridge+Level")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("R2")
    axes[2].set_title("Test R2 per Iteration"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Time series comparison
    test_index = yte.index
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_index, y_actual, label="Actual", lw=1.5, ls="--")
    ax.plot(test_index, y_pred_ridge, label="Ridge", lw=1.0, alpha=0.7)
    ax.plot(test_index, y_pred_combined, label="Ridge+Level", lw=1.2)
    ax.set_title(f"{Y_COLUMN} - Best Iteration")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Local level
    fig, ax = plt.subplots(figsize=(14, 4))
    level_test = backfit_result["level_test"]
    ax.plot(level_test.index, level_test.values, lw=1.3)
    ax.set_title(f"{Y_COLUMN} - Local Level (test)")
    ax.set_ylabel("Level"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "local_level.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_actual, y_pred_ridge, alpha=0.4, s=15)
    lims = [min(y_actual.min(), y_pred_ridge.min()), max(y_actual.max(), y_pred_ridge.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5)
    axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Ridge: RMSE={metrics_ridge['rmse']:.2f} MAE={metrics_ridge['mae']:.2f} R2={metrics_ridge['r2']:.4f}")
    axes[0].set_aspect("equal", adjustable="box"); axes[0].grid(alpha=0.3)

    axes[1].scatter(y_actual, y_pred_combined, alpha=0.4, s=15)
    lims_c = [min(y_actual.min(), y_pred_combined.min()), max(y_actual.max(), y_pred_combined.max())]
    axes[1].plot(lims_c, lims_c, "r--", lw=1.5)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Ridge+Level: RMSE={metrics_combined['rmse']:.2f} MAE={metrics_combined['mae']:.2f} R2={metrics_combined['r2']:.4f}")
    axes[1].set_aspect("equal", adjustable="box"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Save results
    # =========================================================================
    import shutil

    end_time = datetime.now()

    # Save train/test datasets
    Xtr.to_parquet(out_dir / "X_train.parquet")
    Xte.to_parquet(out_dir / "X_test.parquet")
    ytr.to_frame().to_parquet(out_dir / "y_train.parquet")
    yte.to_frame().to_parquet(out_dir / "y_test.parquet")
    turnup_ts.to_parquet(out_dir / "turnup_ts.parquet")

    # Copy source code
    code_dir = out_dir / "code"
    code_dir.mkdir(exist_ok=True)
    source_files = [
        Path(__file__).name, "config.py", "data_cleaning.py",
        "feature_selection.py", "preprocessing.py", "state_estimation.py",
    ]
    for src_file in source_files:
        src_path = Path(__file__).parent / src_file
        if src_path.exists():
            shutil.copy2(src_path, code_dir / src_file)

    # Parameters + metrics JSON
    results_json = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "data_start": str(Xtr.index[0]),
        "data_end": str(Xte.index[-1]),
        "train_start": str(Xtr.index[0]),
        "train_end": str(Xtr.index[-1]),
        "test_start": str(Xte.index[0]),
        "test_end": str(Xte.index[-1]),
        "parameters": {
            "y_column": Y_COLUMN,
            "data_path": DATA_PATH,
            "apply_ewm_filter": APPLY_EWM_FILTER,
            "apply_ewm_filter_y": FILTER_Y,
            "n_iterations": N_ITERATIONS,
            "gamma": GAMMA,
            "splines": SPLINES,
            "fixed_features": fixed_features,
            "best_iteration": history[-1]["iteration"] if history else 0,
            "exclude_mediators": EXCLUDE_MEDIATORS,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "inner_selection_split": int(inner_split),
        },
        "metrics": {
            "ridge_only": metrics_ridge,
            "ridge_level": metrics_combined,
        },
        "diagnostics": diagnostics,
        "selected_features": backfit_result["selected_features"],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Backfit result
    with open(out_dir / "backfit_result.pkl", "wb") as f:
        cloudpickle.dump(backfit_result, f)

    # Production model: preprocessing + Ridge fitted on ALL data
    print("\n--- Building Production Model ---")
    selected_features = backfit_result["selected_features"]
    best_alpha = backfit_result["feature_selection_result"].best_alpha

    production_pipeline, production_feat_list = build_final_pipeline(
        selected_features=selected_features,
        prep_pip=prep_pip,
        created_vars=created_vars,
        ridge_alpha=best_alpha,
        apply_ewm=APPLY_EWM_FILTER,
    )

    # Fit on all data with level-adjusted target
    level_full = pd.concat([backfit_result["level_train"], backfit_result["level_test"]])
    y_adjusted_full = turnup_ts[Y_COLUMN].iloc[:len(level_full)] - level_full.values

    production_pipeline.fit(
        turnup_ts[production_feat_list].iloc[:len(level_full)],
        y_adjusted_full,
    )

    production_artifact = {
        "pipeline": production_pipeline,
        "feat_list": production_feat_list,
        "selected_features": selected_features,
        "state_estimator": backfit_result["state_estimator"],
        "target_column": Y_COLUMN,
        "best_alpha": best_alpha,
    }
    with open(out_dir / "production_model.pkl", "wb") as f:
        cloudpickle.dump(production_artifact, f)

    print(f"\nDone. Results saved to: {out_dir}")
    print(f"Files: {[f.name for f in out_dir.iterdir()]}")


if __name__ == "__main__":
    main()
