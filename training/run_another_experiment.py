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
from datetime import datetime

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from config import CONTROL_VARS
from data_cleaning import (
    unique_in_order, ordered_difference, ordered_intersection,
    outlier, make_design,
)
from preprocessing import make_prep_pip, build_pre_estimator, build_final_pipeline
from feature_selection import cmaes_feature_selection, fit_all_features
from state_estimation import ResidualStateEstimator
from utility import GroupwisePLSTransformer, _feature_engineering, setpoint_df


# =============================================================================
# Default configuration
# =============================================================================

PIPELINE_PREFIXES = {
    "Steam__kWh/T_": ["exha_mois", "inlet_temp", "linepressure", "fabric_tension", "gas_decu"],
     "Electricity__kWh/T_": ["speedsizer_linepressure", "linepressure"],
     "Starch_uptake_by_paper_Top_Roll__g/m2_": ["speedsizer_linepressure", "linepressure", "starch2"]
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
    ]
}

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
        "inv_Rod_Pressure_Bottom_Roll",
        "inv_Rod_pressure_Top_Roll",
        "square_Rod_Pressure_Bottom_Roll",
        "square_Rod_pressure_Top_Roll",
     ]
}

DEFAULT_FIXED_FEATURES = {
    "Steam__kWh/T_": [ 
        "linepressure_1",
        "Starch_uptake__g/m2_",
        "grammage",
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
     ]
}

TEST_SIZE = 0.20
LAGS = 0


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
    args = parser.parse_args()

    Y_COLUMN = args.y_column
    DATA_PATH = args.data_path
    APPLY_EWM_FILTER = args.apply_ewm_filter
    FILTER_Y = args.apply_ewm_filter_y
    N_ITERATIONS = args.n_iterations
    GAMMA = args.gamma
    SPLINES = args.splines

    if args.fixed_features:
        fixed_features = [f.strip() for f in args.fixed_features.split(",")]
    else:
        fixed_features = DEFAULT_FIXED_FEATURES[Y_COLUMN]

    # Timing
    start_time = datetime.now()

    # Output directory
    timestamp = start_time.strftime("%y%m%d%H%M")
    out_dir = Path("other_experiments") / Y_COLUMN.replace("/", "_") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # =========================================================================
    # Data Loading & Filtering
    # =========================================================================
    print("\n--- Data Loading & Filtering ---")

    prep_pip, prep_s_vars = make_prep_pip(prefixes=PIPELINE_PREFIXES[Y_COLUMN])

    turnup_data = pd.read_parquet(DATA_PATH)

    ctl_vars = unique_in_order(
        v for v in CONTROL_VARS[Y_COLUMN] if "vacuum" not in v.lower()
    )
    created_vars = ordered_intersection(CREATED_VARIABLE_CANDIDATES[Y_COLUMN], ctl_vars)

    turnup_data = _feature_engineering(turnup_data, setpoint_df, steam_null=False, clip=False)
    turnup_data = turnup_data.set_index("Wedge_Time").sort_index()

    if Y_COLUMN== "Steam__kWh/T_":
        turnup_data = turnup_data[turnup_data.index >   "2026-3-1"]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-24 07:00") & (turnup_data.index < "2026-01-26 10:00"))]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-11 12:00") & (turnup_data.index < "2026-01-12 11:00"))]
        turnup_data = turnup_data[~((turnup_data.index > "2026-01-17 12:00") & (turnup_data.index < "2026-01-19 11:00"))]
        turnup_data = turnup_data[turnup_data.index < "2026-07-5"]
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

    turnup_data = turnup_data[turnup_data.grammage.isin([115, 120, 100,  90,  85, 110])]

    
    print(f"Filtered data: {turnup_data.shape}")

    # =========================================================================
    # Build Preprocessing Pipeline
    # =========================================================================
    print("\n--- Build Preprocessing Pipeline ---")
    import re

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
    turnup_ts = turnup_ts.dropna(subset=[Y_COLUMN])

    n_samples = len(turnup_ts)
    split = int(n_samples * (1.0 - TEST_SIZE))

    ts_raw = turnup_ts.loc[:, feat_list]
    pre_estimator.fit(ts_raw.iloc[:split], turnup_ts[Y_COLUMN].iloc[:split])
    ts_transformed = pre_estimator.transform(ts_raw)

    print(f"Transformed shape: {ts_transformed.shape}")
    print(f"Transformed columns: {list(ts_transformed.columns)}")

    ts_transformed[Y_COLUMN] = turnup_ts[Y_COLUMN].values

    grade_change = turnup_ts["AB_Grade_ID"].ne(turnup_ts["AB_Grade_ID"].shift())
    gap12 = turnup_ts.index.to_series().diff().gt(pd.Timedelta("12h")).fillna(True)
    ts_transformed["run"] = (grade_change | gap12).cumsum()

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

    # Exclude "run" column from features (used for aggregation, not regression)
    transformed_feature_names = [c for c in ts_transformed.columns if c not in (Y_COLUMN, "run")]
    X, y = make_design(ts_transformed, Y_COLUMN, transformed_feature_names, None, y_lags=range(1, 1 + LAGS))

    # Keep run labels aligned with the design matrix (make_design may drop rows)
    run_labels = ts_transformed["run"].loc[X.index]
    print(f"Design matrix: X={X.shape}, y={y.shape}")

    # =========================================================================
    # Train/Test Split
    # =========================================================================
    n_samples = len(X)
    split = int(n_samples * (1.0 - TEST_SIZE))

    Xtr, Xte = X.iloc[:split].copy(), X.iloc[split:].copy()
    ytr, yte = y.iloc[:split].copy(), y.iloc[split:].copy()
    outer_cv = [(np.arange(split), np.arange(split, n_samples))]

    # Raw y for metrics (unfiltered, aligned with design matrix index)
    y_raw_series = pd.Series(y_raw, index=ts_transformed.index, name=Y_COLUMN)
    y_raw_aligned = y_raw_series.loc[X.index]
    yte_raw = y_raw_aligned.iloc[split:].values
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

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
            cv_splits=outer_cv,
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
    # Iterative Backfitting with Run-Level Local Level
    # =========================================================================
    print("\n--- Iterative Backfitting (run-level local level) ---")

    # Helper: aggregate residuals to run level and compute midpoint timestamps
    def _aggregate_to_runs(residuals_series, run_labels_series):
        """Mean residual per run, indexed by midpoint timestamp.
        Returns (aggregated_series, run_ids_in_order)."""
        resid_run = residuals_series.groupby(run_labels_series).mean()
        run_ids = resid_run.index.copy()  # run IDs that survived groupby
        mid_ts = residuals_series.index.to_series().groupby(run_labels_series).apply(
            lambda g: g.iloc[len(g) // 2]
        )
        resid_run.index = mid_ts.loc[run_ids].values
        return resid_run, run_ids

    # Helper: broadcast run-level values back to timesteps
    def _broadcast_to_timesteps(run_level_values, run_labels_series, run_ids):
        """Map run-level series back to timestep index via run labels."""
        run_map = pd.Series(run_level_values.values, index=run_ids)
        return run_labels_series.map(run_map).values

    # Identify run labels for train/test
    run_labels_train = run_labels.iloc[:split]
    run_labels_test = run_labels.iloc[split:]

    # State for backfitting loop
    y_adjusted_full = y.astype(np.float64).copy()
    level_ts_train = pd.Series(0.0, index=Xtr.index)  # timestep-level level (broadcast)

    history = []
    best_rmse_ridge = np.inf
    best_iteration = -1
    best_state = None
    no_improve_count = 0

    for iteration in range(N_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{N_ITERATIONS}")
        print(f"{'='*60}")

        # --- Step A: Feature selection on adjusted target ---
        use_splines = SPLINES and iteration > 0
        fs_result = run_feature_selection(
            X, np.array(y_adjusted_full).ravel(), iteration, splines=use_splines,
        )
        selected = fs_result.selected_features
        estimator = fs_result.final_estimator

        y_train_pred = estimator.predict(Xtr[selected].values)
        y_test_pred = estimator.predict(Xte[selected].values)

        # --- Step B: Residuals from ORIGINAL target (not adjusted) ---
        residuals_train = pd.Series(
            y_raw_aligned.iloc[:split].values - y_train_pred, index=Xtr.index
        )
        residuals_test = pd.Series(
            yte_raw - y_test_pred, index=Xte.index
        )

        # --- Step C: Aggregate residuals to run level ---
        resid_run_train, train_run_ids_agg = _aggregate_to_runs(residuals_train, run_labels_train)
        resid_run_test, test_run_ids_agg = _aggregate_to_runs(residuals_test, run_labels_test)

        # --- Step D: Fit LocalLevel on run-level train residuals ---
        state_est = ResidualStateEstimator(level=True)
        state_result_train = state_est.fit(resid_run_train)
        level_run_train = state_result_train.level

        # --- Step E: Causal one-step-ahead level for test runs ---
        level_run_test = state_est.one_step_ahead_level(resid_run_test)

        # --- Step F: Broadcast run-level level back to timesteps ---
        level_ts_train_vals = _broadcast_to_timesteps(
            level_run_train, run_labels_train, train_run_ids_agg
        )
        level_ts_test_vals = _broadcast_to_timesteps(
            level_run_test, run_labels_test, test_run_ids_agg
        )

        # Fill NaN (unmapped runs) with 0 — no level adjustment for those
        level_ts_train_vals = np.nan_to_num(level_ts_train_vals, nan=0.0)
        level_ts_test_vals = np.nan_to_num(level_ts_test_vals, nan=0.0)

        level_ts_train = pd.Series(level_ts_train_vals, index=Xtr.index)
        level_ts_test = pd.Series(level_ts_test_vals, index=Xte.index)

        # --- Step G: Adjust target for next iteration (train only) ---
        y_adjusted_full = y.astype(np.float64).copy()
        y_adjusted_full.iloc[:split] = (
            y_raw_aligned.iloc[:split].values - GAMMA * level_ts_train_vals
        )

        # --- Metrics ---
        y_test_combined = y_test_pred + level_ts_test_vals

        rmse_ridge = float(np.sqrt(mean_squared_error(yte_raw, y_test_pred)))
        rmse_combined = float(np.sqrt(mean_squared_error(yte_raw, y_test_combined)))
        r2_ridge = float(r2_score(yte_raw, y_test_pred))
        r2_combined = float(r2_score(yte_raw, y_test_combined))

        iter_info = {
            "iteration": iteration + 1,
            "n_features": fs_result.n_features,
            "selected_features": selected,
            "ridge_alpha": fs_result.best_alpha,
            "rmse_ridge": rmse_ridge,
            "rmse_combined": rmse_combined,
            "r2_ridge": r2_ridge,
            "r2_combined": r2_combined,
            "level_scale": state_result_train.level_scale,
            "obs_noise_scale": state_result_train.observation_noise_scale,
        }
        history.append(iter_info)

        # Track best by ridge RMSE
        if rmse_ridge < best_rmse_ridge:
            best_rmse_ridge = rmse_ridge
            best_iteration = iteration
            no_improve_count = 0
            best_state = {
                "feature_selection_result": fs_result,
                "selected_features": selected,
                "state_estimator": state_est,
                "y_train_pred": pd.Series(y_train_pred, index=Xtr.index),
                "y_test_pred": pd.Series(y_test_pred, index=Xte.index),
                "level_train": level_ts_train.copy(),
                "level_test": level_ts_test.copy(),
                "level_run_train": level_run_train.copy(),
                "level_run_test": level_run_test.copy(),
                "y_test_combined": pd.Series(y_test_combined, index=Xte.index),
            }
        else:
            no_improve_count += 1

        print(f"  Features: {len(selected)}")
        print(f"  Ridge RMSE: {rmse_ridge:.3f}, R2: {r2_ridge:.4f}")
        print(f"  Combined RMSE: {rmse_combined:.3f}, R2: {r2_combined:.4f}")
        print(f"  Level scale: {state_result_train.level_scale:.4f}")
        if iteration == best_iteration:
            print(f"  *** New best ***")

        # Early stopping
        if no_improve_count >= 2:
            print(f"\n  Early stopping at iteration {iteration + 1} (no improvement for 2 iters)")
            break

    print(f"\nBest iteration: {best_iteration + 1}")

    # =========================================================================
    # Metrics (from best iteration)
    # =========================================================================
    y_actual = yte_raw
    y_pred_ridge = best_state["y_test_pred"].values
    y_pred_combined = best_state["y_test_combined"].values
    level_ts_test = best_state["level_test"]
    level_run_train = best_state["level_run_train"]
    level_run_test = best_state["level_run_test"]

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
    # Plots
    # =========================================================================
    iters = [h["iteration"] for h in history]

    # Convergence plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(iters, [h["rmse_ridge"] for h in history], "o-", label="Ridge")
    axes[0].plot(iters, [h["rmse_combined"] for h in history], "s-", label="Ridge+Level")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("RMSE")
    axes[0].set_title("Test RMSE per Iteration"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(iters, [h["rmse_ridge"] for h in history], "o-", label="Ridge")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("MAE")
    axes[1].set_title("Test RMSE per Iteration (Ridge)"); axes[1].grid(alpha=0.3)

    axes[2].plot(iters, [h["r2_ridge"] for h in history], "o-", label="Ridge")
    axes[2].plot(iters, [h["r2_combined"] for h in history], "s-", label="Ridge+Level")
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("R2")
    axes[2].set_title("Test R2 per Iteration"); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Time series comparison
    test_index = Xte.index
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_index, y_actual, label="Actual", lw=1.5, ls="--")
    ax.plot(test_index, y_pred_ridge, label="Ridge", lw=1.0, alpha=0.7)
    ax.plot(test_index, y_pred_combined, label="Ridge+Level", lw=1.2)
    ax.set_title(f"{Y_COLUMN} - Best Iteration")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Local level (run-level: both train and test)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(level_run_train.index, level_run_train.values, "o-", ms=3, lw=1.3)
    axes[0].axhline(0, ls="--", color="gray", alpha=0.5)
    axes[0].set_title(f"{Y_COLUMN} - Run-Level Local Level (train)")
    axes[0].set_ylabel("Level"); axes[0].grid(alpha=0.3)

    axes[1].plot(level_run_test.index, level_run_test.values, "s-", ms=3, lw=1.3, color="C1")
    axes[1].axhline(0, ls="--", color="gray", alpha=0.5)
    axes[1].set_title(f"{Y_COLUMN} - Run-Level Local Level (test, causal)")
    axes[1].set_ylabel("Level"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "local_level_run.png", dpi=150, bbox_inches="tight")
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
        "run_experiment.py", "config.py", "data_cleaning.py",
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
            "best_iteration": best_iteration + 1,
            "level_estimation": "run-level",
        },
        "metrics": {
            "ridge_only": metrics_ridge,
            "ridge_level": metrics_combined,
        },
        "selected_features": best_state["selected_features"],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Backfit result
    with open(out_dir / "backfit_result.pkl", "wb") as f:
        cloudpickle.dump(best_state, f)

    # Production model: preprocessing + Ridge fitted on ALL data
    print("\n--- Building Production Model ---")
    selected_features = best_state["selected_features"]
    best_alpha = best_state["feature_selection_result"].best_alpha

    production_pipeline, production_feat_list = build_final_pipeline(
        selected_features=selected_features,
        prep_pip=prep_pip,
        created_vars=created_vars,
        ridge_alpha=best_alpha,
        apply_ewm=APPLY_EWM_FILTER,
    )

    # Fit on all data with level-adjusted target (broadcast run-level level)
    level_full = pd.concat([best_state["level_train"], best_state["level_test"]])
    y_adjusted_production = turnup_ts[Y_COLUMN].iloc[:len(level_full)] - level_full.values

    production_pipeline.fit(
        turnup_ts[production_feat_list].iloc[:len(level_full)],
        y_adjusted_production,
    )

    production_artifact = {
        "pipeline": production_pipeline,
        "feat_list": production_feat_list,
        "selected_features": selected_features,
        "state_estimator": best_state["state_estimator"],
        "target_column": Y_COLUMN,
        "best_alpha": best_alpha,
    }
    with open(out_dir / "production_model.pkl", "wb") as f:
        cloudpickle.dump(production_artifact, f)

    print(f"\nDone. Results saved to: {out_dir}")
    print(f"Files: {[f.name for f in out_dir.iterdir()]}")


if __name__ == "__main__":
    main()
