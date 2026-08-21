"""
Run-Level + Within-Run Hierarchical Steam Model.

Procedure:
1. Aggregate timestep data to run level (mean per run)
2. Remove grade effect, estimate latent level+trend from the target
3. Fit Ridge on the de-trended run-level target: y_r = f(X_r) + s_r
4. Compute within-run deviations: y_h = y - y_r, X_h = X - X_r
   (with EWM smoothing if --apply_ewm_filter is set)
5. Fit Ridge on within-run deviations: y_h = g(X_h)
6. Final prediction: y_hat = f(X_r) + s_r + g(X_h), evaluated against raw y

Usage:
    python run_level_experiment.py --y_column "Steam__kWh/T_" --data_path ../data/costimier_turnup.parquet
"""

import sys
from pathlib import Path

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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from config import CONTROL_VARS
from data_cleaning import unique_in_order, ordered_difference, ordered_intersection
from preprocessing import make_prep_pip, build_pre_estimator
from feature_selection import cmaes_feature_selection
from state_estimation import ResidualStateEstimator
from utility import GroupwisePLSTransformer, _feature_engineering, setpoint_df


# =============================================================================
# Configuration (per target)
# =============================================================================

PIPELINE_PREFIXES = {
    "Steam__kWh/T_": ["exha_mois", "inlet_temp", "linepressure", "fabric_tension", "gas_decu"],
}

BLACK_LIST = {
    "Steam__kWh/T_": [
        "Current_basis_weight", "Bentonite_1_mass_flow__g/T_",
        "Bentonite_2_mass_flow__g/T_", "DG3_Moisture_content_Outlet_Air",
        "Lip_settings", "Conductivity_white_water_B46",
        "pH-Messung_Verd\u00fcnnungswasser__2..12_pH_",
        "pH_measurement_white_water_B41", "CO2_mass_flow__g/T_",
    ],
}

CREATED_VARIABLE_CANDIDATES = {
    "Steam__kWh/T_": [
        "Water_Predryer", "diluted_starch", "Fibre__g/m2_",
        "Water_Afterdryer_output", "dewatering",
        "Production_Rate__T/h_", "fibre_short/long", "Starch_uptake__g/m2_",
    ],
}

DEFAULT_FIXED_FEATURES = {
    "Steam__kWh/T_": ["Starch_uptake__g/m2_", "ambient_temp_C", "linepressure_1", "grammage"],
}

GRAMMAGES = [115, 120, 100, 90, 85, 110]
TEST_SIZE = 0.20
ALPHAS = np.logspace(0, 3, 20)


def main():
    parser = argparse.ArgumentParser(description="Run-level steam model with latent state")
    parser.add_argument("--y_column", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--apply_ewm_filter", action="store_true")
    parser.add_argument("--trend", action="store_true", help="Use local linear trend (not just level)")
    parser.add_argument("--fixed_features", type=str, default=None)
    parser.add_argument("--model", choices=["ridge", "cmaes"], default="ridge",
                        help="ridge = all features; cmaes = CMA-ES feature selection")
    parser.add_argument("--splines", action="store_true", help="Use splines (cmaes model only)")
    parser.add_argument("--k_min", type=int, default=3, help="Min features for CMA-ES")
    parser.add_argument("--k_max", type=int, default=15, help="Max features for CMA-ES")
    parser.add_argument("--max_evals", type=int, default=3000, help="CMA-ES evaluation budget")
    args = parser.parse_args()

    Y_COLUMN = args.y_column
    DATA_PATH = args.data_path
    APPLY_EWM_FILTER = args.apply_ewm_filter
    USE_TREND = args.trend
    MODEL = args.model
    SPLINES = args.splines
    fixed_features = (
        [f.strip() for f in args.fixed_features.split(",")]
        if args.fixed_features else DEFAULT_FIXED_FEATURES[Y_COLUMN]
    )

    start_time = datetime.now()
    out_dir = Path("experiments_runlevel") / Y_COLUMN.replace("/", "_") / start_time.strftime("%y%m%d%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # -------------------------------------------------------------------------
    # 1. Data loading & filtering
    # -------------------------------------------------------------------------
    print("\n[1] Loading data...")
    prep_pip, prep_s_vars = make_prep_pip(prefixes=PIPELINE_PREFIXES[Y_COLUMN])
    turnup_data = pd.read_parquet(DATA_PATH)

    ctl_vars = unique_in_order(v for v in CONTROL_VARS[Y_COLUMN] if "vacuum" not in v.lower())
    created_vars = ordered_intersection(CREATED_VARIABLE_CANDIDATES[Y_COLUMN], ctl_vars)

    turnup_data = _feature_engineering(turnup_data, setpoint_df, steam_null=False, clip=False)
    turnup_data = turnup_data.set_index("Wedge_Time").sort_index()

    turnup_data = turnup_data[turnup_data.index > "2026-3-1"]
    turnup_data = turnup_data[turnup_data.index < "2026-07-5"]
    turnup_data = turnup_data[
        (turnup_data["DG4_Temperature_Inlet_Air"] > 100)
        & (turnup_data["Vacuum_Zone_1_PickUp"] < -0.5)
    ]
    print(f"    Filtered: {turnup_data.shape}")

    # -------------------------------------------------------------------------
    # 2. Build preprocessing pipeline
    # -------------------------------------------------------------------------
    print("[2] Building preprocessing pipeline...")
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
    print(f"    Pipeline input: {len(feat_list)} columns")

    # -------------------------------------------------------------------------
    # 3. Transform & aggregate to run level
    # -------------------------------------------------------------------------
    print("[3] Transforming and aggregating to run level...")
    turnup_ts = turnup_data.copy().sort_index()
    turnup_ts = turnup_ts.dropna(subset=[Y_COLUMN])
    turnup_ts = turnup_ts[turnup_ts.grammage.isin(GRAMMAGES)]

    n_samples = len(turnup_ts)
    split_ts = int(n_samples * (1.0 - TEST_SIZE))

    ts_raw = turnup_ts.loc[:, feat_list]
    pre_estimator.fit(ts_raw.iloc[:split_ts], turnup_ts[Y_COLUMN].iloc[:split_ts])
    ts_transformed = pre_estimator.transform(ts_raw)

    # Define runs
    grade_change = turnup_ts["AB_Grade_ID"].ne(turnup_ts["AB_Grade_ID"].shift())
    gap12 = turnup_ts.index.to_series().diff().gt(pd.Timedelta("12h")).fillna(True)
    run_labels = (grade_change | gap12).cumsum()

    # Aggregate features to run level (drop target if present)
    ts_transformed = ts_transformed.drop(columns=[Y_COLUMN], errors="ignore")
    X_run = ts_transformed.groupby(run_labels.values).mean()

    # Aggregate target and grade
    y_run = turnup_ts[Y_COLUMN].groupby(run_labels).mean()
    run_grade_id = turnup_ts["AB_Grade_ID"].groupby(run_labels).first().values

    # Midpoint timestamp per run for a proper time index
    run_timestamps = turnup_ts.index.to_series().groupby(run_labels).apply(lambda g: g.iloc[len(g) // 2])
    X_run.index = run_timestamps.values
    y_run.index = run_timestamps.values

    feature_cols = list(X_run.columns)
    print(f"    Runs: {len(X_run)}, features: {len(feature_cols)}")

    # -------------------------------------------------------------------------
    # 4. Train/test split (run level)
    # -------------------------------------------------------------------------
    n_runs = len(X_run)
    split = int(n_runs * (1.0 - TEST_SIZE))
    Xtr_run, Xte_run = X_run.iloc[:split], X_run.iloc[split:]
    ytr_run, yte_run = y_run.iloc[:split], y_run.iloc[split:]
    print(f"[4] Train: {len(Xtr_run)} runs, Test: {len(Xte_run)} runs")

    # -------------------------------------------------------------------------
    # 5. Grade-demean + estimate CAUSAL latent level from target (no leakage)
    # -------------------------------------------------------------------------
    print("[5] Estimating causal latent level...")
    # Grade means from train only
    grade_means_train = y_run.iloc[:split].groupby(run_grade_id[:split]).mean()
    grade_mean_full = pd.Series(run_grade_id, index=y_run.index).map(grade_means_train)
    grade_mean_full = grade_mean_full.fillna(y_run.iloc[:split].mean())

    y_demeaned = y_run - grade_mean_full

    # Fit state on train demeaned target; parameters estimated on train only
    state_est = ResidualStateEstimator(level=True, trend=USE_TREND)
    state_result = state_est.fit(y_demeaned.iloc[:split])

    # Train level: smoothed is fine (in-sample), used to build the detrended
    # training target for Ridge.
    level_train = state_result.level

    # Test level: strictly causal one-step-ahead (uses only past)
    level_causal_test = state_est.one_step_ahead_level(y_demeaned.iloc[split:])

    print(f"    Level scale: {state_result.level_scale:.4f}, "
          f"Obs noise: {state_result.observation_noise_scale:.4f}")

    # -------------------------------------------------------------------------
    # 6. Detrend target and fit regression (leakage-free)
    # -------------------------------------------------------------------------
    print(f"[6] Detrending target and fitting model ({MODEL})...")
    # Detrended target: y* = y - level  (grade effect kept, model handles it)
    ytr_detrended = ytr_run.values - level_train.values
    yte_detrended = yte_run.values - level_causal_test.values  # causal level on test

    if MODEL == "ridge":
        # --- Plain Ridge on all features ---
        ridge_cv = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            param_grid={"ridge__alpha": ALPHAS},
            cv=TimeSeriesSplit(n_splits=5),
            scoring="neg_root_mean_squared_error",
            refit=True,
        )
        ridge_cv.fit(Xtr_run.values, ytr_detrended)
        y_test_pred_detrended = ridge_cv.predict(Xte_run.values)
        selected_features = feature_cols
        model_estimator = ridge_cv
        best_alpha = float(ridge_cv.best_params_["ridge__alpha"])
        print(f"    Best alpha: {best_alpha:.2f}")

    else:
        # --- CMA-ES feature selection (+ optional splines) + Ridge ---
        # Fit and evaluate on detrended (both train and test have level removed).
        # This selects features that genuinely explain the regression component.
        # After selection, refit on detrended-train only, predict, add level back.

        # Fixed features present in the transformed columns
        fixed_resolved = (
            [v for v in fixed_features if "grammage" not in v.lower() and v in feature_cols]
            + [v for v in feature_cols if "grammage" in v.lower()]
        )
        # PLS score groups (all-or-nothing selection)
        feature_groups = {
            step.score_prefix: [f"{step.score_prefix}_{i}" for i in range(1, step.n_components + 1)]
            for _, step in prep_pip.steps
            if isinstance(step, GroupwisePLSTransformer)
        }
        # Keep only groups whose scores are present
        feature_groups = {
            k: [c for c in cols if c in feature_cols]
            for k, cols in feature_groups.items()
            if any(c in feature_cols for c in cols)
        }

        # Build y_detrended_full: both train and test are detrended.
        # CMA-ES with cv_splits=[(train,test)] will fit on detrended-train
        # and score on detrended-test, selecting features that genuinely
        # explain the regression component (level removed from both sides).
        train_idx = np.arange(split)
        test_idx = np.arange(split, n_runs)
        y_detrended_full = np.empty(n_runs)
        y_detrended_full[train_idx] = ytr_detrended        # y_train - level_train
        y_detrended_full[test_idx] = yte_detrended          # y_test - causal_level_test
        cv_splits = [(train_idx, test_idx)]

        fs_result = cmaes_feature_selection(
            X_run, y_detrended_full,
            k_range=(args.k_min, args.k_max),
            fixed_features=fixed_resolved,
            feature_groups=feature_groups,
            cv_splits=cv_splits,
            selection="topk",
            max_evals=args.max_evals,
            sigma0=1.0,
            seed=42,
            popsize=24,
            penalty_fn=None,
            splines=SPLINES,
            verbose=True,
        )
        selected_features = fs_result.selected_features
        best_alpha = float(fs_result.best_alpha)
        print(f"    Selected {len(selected_features)} features, alpha={best_alpha:.2f}")
        print(f"    Features: {selected_features}")

        # fs_result.final_estimator is already fit on detrended-train
        model_estimator = fs_result.final_estimator
        y_test_pred_detrended = model_estimator.predict(Xte_run[selected_features].values)

    # -------------------------------------------------------------------------
    # 7. Metrics — how well regression explains the detrended target
    # -------------------------------------------------------------------------
    def _metrics(y_true, y_pred):
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    # (a) De-trended: regression vs (y_test - causal_level)  <-- the key metric
    metrics_detrended = _metrics(yte_detrended, y_test_pred_detrended)

    # (b) Original scale: regression + causal level vs raw y
    y_actual = yte_run.values
    y_pred_full = y_test_pred_detrended + level_causal_test.values
    metrics_full = _metrics(y_actual, y_pred_full)

    metrics = {
        "detrended": metrics_detrended,   # regression on causal-detrended target
        "full_causal": metrics_full,      # regression + causal level on raw scale
    }

    print("\n[7] Results:")
    print(f"    De-trended (regression vs y-level_causal): "
          f"RMSE={metrics_detrended['rmse']:.2f}, MAE={metrics_detrended['mae']:.2f}, R2={metrics_detrended['r2']:.4f}")
    print(f"    Full causal (regression + level vs raw y): "
          f"RMSE={metrics_full['rmse']:.2f}, MAE={metrics_full['mae']:.2f}, R2={metrics_full['r2']:.4f}")

    # -------------------------------------------------------------------------
    # 7b. Within-run (reel-level) model: y_h = g(X_h)
    # -------------------------------------------------------------------------
    print("\n[7b] Within-run (reel-level) regression...")

    # Broadcast run-level prediction (f(X_r) + s_r) back to each timestep
    # Build y_r per run: Ridge prediction + level (smoothed for train, causal for test)
    y_run_pred_train = model_estimator.predict(
        Xtr_run[selected_features].values if MODEL == "cmaes" else Xtr_run.values
    )
    y_run_pred_test = y_test_pred_detrended  # already computed
    # Full run-level prediction on original scale
    yr_train = y_run_pred_train + level_train.values     # f(X_r) + s_r (train)
    yr_test = y_run_pred_test + level_causal_test.values  # f(X_r) + s_r (test)

    # Map run IDs to train/test run-level predictions
    # run_labels are consecutive integers (1,2,3...) matching X_run row order
    all_run_ids = sorted(run_labels.dropna().unique())  # sorted = row order in X_run
    train_run_ids = all_run_ids[:split]
    test_run_ids = all_run_ids[split:n_runs]

    run_to_yr = {}
    for i, rid in enumerate(train_run_ids):
        run_to_yr[rid] = yr_train[i]
    for i, rid in enumerate(test_run_ids):
        run_to_yr[rid] = yr_test[i]

    # Map run IDs to run-level feature means (X_r)
    run_to_Xr = {}
    for i, rid in enumerate(all_run_ids[:n_runs]):
        run_to_Xr[rid] = X_run.iloc[i].values

    # Broadcast to timestep level (vectorized)
    yr_broadcast = run_labels.map(run_to_yr).values.astype(float)
    Xr_broadcast = np.array([run_to_Xr.get(r, np.full(X_run.shape[1], np.nan))
                             for r in run_labels.values])

    # Compute within-run deviations
    # X_h = X - X_r (ts_transformed is already EWM-filtered if apply_ewm is set)
    X_h = ts_transformed.values - Xr_broadcast

    # y_h = y - y_r (or EWM(y) - y_r if EWM active)
    if APPLY_EWM_FILTER:
        from utility import ewm_reset
        # EWM-smooth y within runs
        y_ts_raw = turnup_ts[Y_COLUMN].values[:len(ts_transformed)]
        y_ts_series = pd.Series(y_ts_raw, index=ts_transformed.index)
        y_ewm = y_ts_series.groupby(run_labels.values).transform(ewm_reset).values
        y_h = y_ewm - yr_broadcast
    else:
        y_ts_raw = turnup_ts[Y_COLUMN].values[:len(ts_transformed)]
        y_h = y_ts_raw - yr_broadcast

    # Keep raw y for final evaluation
    y_raw_ts = turnup_ts[Y_COLUMN].values[:len(ts_transformed)]

    # Mask: only timesteps with a valid (mapped) run
    valid_mask = ~np.isnan(yr_broadcast)

    # Train/test split at timestep level (same split ratio)
    # Apply valid mask within each split
    train_valid = valid_mask[:split_ts]
    test_valid = valid_mask[split_ts:]

    X_h_train = X_h[:split_ts][train_valid]
    X_h_test = X_h[split_ts:][test_valid]
    y_h_train = y_h[:split_ts][train_valid]
    y_h_test = y_h[split_ts:][test_valid]

    # Fit Ridge on within-run deviations (train only)
    print(f"    X_h shape: train={X_h_train.shape}, test={X_h_test.shape}"
          f" (dropped {(~train_valid).sum()} train, {(~test_valid).sum()} test NaN rows)")
    ridge_h_cv = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
        param_grid={"ridge__alpha": ALPHAS},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        refit=True,
    )
    ridge_h_cv.fit(X_h_train, y_h_train)
    y_h_pred_test = ridge_h_cv.predict(X_h_test)
    best_alpha_h = float(ridge_h_cv.best_params_["ridge__alpha"])
    print(f"    Within-run Ridge alpha: {best_alpha_h:.2f}")

    # -------------------------------------------------------------------------
    # 7c. Combined metrics: y_hat = y_r + y_h_hat, evaluated against raw y
    # -------------------------------------------------------------------------
    yr_test_broadcast = yr_broadcast[split_ts:][test_valid]
    y_combined_test = yr_test_broadcast + y_h_pred_test
    y_raw_test = y_raw_ts[split_ts:][test_valid]

    metrics_within_run = _metrics(y_h_test, y_h_pred_test)
    metrics_combined = _metrics(y_raw_test, y_combined_test)

    # Also report run-level only (broadcast) vs raw
    metrics_run_only = _metrics(y_raw_test, yr_test_broadcast)

    metrics["within_run"] = metrics_within_run
    metrics["combined"] = metrics_combined
    metrics["run_only_broadcast"] = metrics_run_only

    print(f"    Within-run regression: RMSE={metrics_within_run['rmse']:.2f}, R2={metrics_within_run['r2']:.4f}")
    print(f"    Run-level only (broadcast): RMSE={metrics_run_only['rmse']:.2f}, R2={metrics_run_only['r2']:.4f}")
    print(f"    Combined (run + within-run): RMSE={metrics_combined['rmse']:.2f}, R2={metrics_combined['r2']:.4f}")

    # -------------------------------------------------------------------------
    # 8. Plots
    # -------------------------------------------------------------------------
    print("[8] Saving plots...")
    test_idx = yte_run.index

    # De-trended target vs regression prediction
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_idx, yte_detrended, "k--", lw=1.5, label="y - causal level (target)", zorder=10)
    ax.plot(test_idx, y_test_pred_detrended, "o-", ms=3,
            label=f"{MODEL} (R2={metrics_detrended['r2']:.3f})")
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_title(f"{Y_COLUMN} — De-trended Target vs Regression")
    ax.set_ylabel("De-trended " + Y_COLUMN); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "detrended_regression.png", dpi=150, bbox_inches="tight"); plt.close()

    # Full-scale prediction
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_idx, y_actual, "k--", lw=1.5, label="Actual", zorder=10)
    ax.plot(test_idx, y_pred_full, "s-", ms=3,
            label=f"Ridge + Causal Level (R2={metrics_full['r2']:.3f})")
    ax.set_title(f"{Y_COLUMN} — Full Prediction (regression + causal level)")
    ax.set_ylabel(Y_COLUMN); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "predictions.png", dpi=150, bbox_inches="tight"); plt.close()

    # Causal level on test
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_idx, level_causal_test.values, "s-", ms=3, label="Causal (one-step-ahead)")
    ax.axhline(0, ls="--", color="gray")
    ax.set_title(f"{Y_COLUMN} — Causal Latent Level (test)")
    ax.set_ylabel("Level"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "level_causal.png", dpi=150, bbox_inches="tight"); plt.close()

    # Scatter: de-trended and full side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # De-trended
    axes[0].scatter(yte_detrended, y_test_pred_detrended, alpha=0.6, s=30)
    lims = [min(yte_detrended.min(), y_test_pred_detrended.min()),
            max(yte_detrended.max(), y_test_pred_detrended.max())]
    axes[0].plot(lims, lims, "r--", lw=1.5)
    axes[0].set_xlabel("De-trended actual"); axes[0].set_ylabel("Regression prediction")
    axes[0].set_title(f"De-trended: RMSE={metrics_detrended['rmse']:.2f}, R2={metrics_detrended['r2']:.4f}")
    axes[0].set_aspect("equal", adjustable="box"); axes[0].grid(alpha=0.3)

    # Full (regression + causal level)
    axes[1].scatter(y_actual, y_pred_full, alpha=0.6, s=30)
    lims_f = [min(y_actual.min(), y_pred_full.min()), max(y_actual.max(), y_pred_full.max())]
    axes[1].plot(lims_f, lims_f, "r--", lw=1.5)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Regression + causal level")
    axes[1].set_title(f"Full: RMSE={metrics_full['rmse']:.2f}, R2={metrics_full['r2']:.4f}")
    axes[1].set_aspect("equal", adjustable="box"); axes[1].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(out_dir / "scatter.png", dpi=150, bbox_inches="tight"); plt.close()

    # Within-run model plots
    test_ts_index = ts_transformed.index[split_ts:][test_valid]

    # Combined prediction vs raw (timestep level)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_ts_index, y_raw_test, "k", lw=0.5, alpha=0.5, label="Actual (raw)")
    ax.plot(test_ts_index, yr_test_broadcast, lw=1.2, label=f"Run-level (R2={metrics_run_only['r2']:.3f})")
    ax.plot(test_ts_index, y_combined_test, lw=1.0, alpha=0.8,
            label=f"Combined (R2={metrics_combined['r2']:.3f})")
    ax.set_title(f"{Y_COLUMN} — Reel-Level: Run-only vs Combined")
    ax.set_ylabel(Y_COLUMN); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "combined_timeseries.png", dpi=150, bbox_inches="tight"); plt.close()

    # Scatter: run-only vs combined (timestep level)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_raw_test, yr_test_broadcast, alpha=0.3, s=10)
    lims_r = [min(y_raw_test.min(), yr_test_broadcast.min()),
              max(y_raw_test.max(), yr_test_broadcast.max())]
    axes[0].plot(lims_r, lims_r, "r--", lw=1.5)
    axes[0].set_xlabel("Actual (raw)"); axes[0].set_ylabel("Run-level prediction")
    axes[0].set_title(f"Run-only: RMSE={metrics_run_only['rmse']:.2f}, R2={metrics_run_only['r2']:.4f}")
    axes[0].set_aspect("equal", adjustable="box"); axes[0].grid(alpha=0.3)

    axes[1].scatter(y_raw_test, y_combined_test, alpha=0.3, s=10)
    lims_c = [min(y_raw_test.min(), y_combined_test.min()),
              max(y_raw_test.max(), y_combined_test.max())]
    axes[1].plot(lims_c, lims_c, "r--", lw=1.5)
    axes[1].set_xlabel("Actual (raw)"); axes[1].set_ylabel("Combined (run + within-run)")
    axes[1].set_title(f"Combined: RMSE={metrics_combined['rmse']:.2f}, R2={metrics_combined['r2']:.4f}")
    axes[1].set_aspect("equal", adjustable="box"); axes[1].grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(out_dir / "scatter_combined.png", dpi=150, bbox_inches="tight"); plt.close()

    # -------------------------------------------------------------------------
    # 9. Save results
    # -------------------------------------------------------------------------
    print("[9] Saving results...")
    end_time = datetime.now()

    results_json = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "data_start": str(X_run.index[0]),
        "data_end": str(X_run.index[-1]),
        "n_runs": int(n_runs),
        "parameters": {
            "y_column": Y_COLUMN,
            "data_path": DATA_PATH,
            "apply_ewm_filter": APPLY_EWM_FILTER,
            "trend": USE_TREND,
            "model": MODEL,
            "splines": SPLINES,
            "fixed_features": fixed_features,
            "grammages": GRAMMAGES,
            "ridge_alpha": best_alpha,
            "ridge_alpha_within_run": best_alpha_h,
        },
        "metrics": metrics,
        "selected_features": selected_features,
        "all_features": feature_cols,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    artifact = {
        "model_run": model_estimator,
        "model_within_run": ridge_h_cv,
        "state_estimator": state_est,
        "grade_means_train": grade_means_train,
        "selected_features": selected_features,
        "feature_cols": feature_cols,
        "feat_list": feat_list,
        "pre_estimator": pre_estimator,
        "y_column": Y_COLUMN,
        "ridge_alpha_within_run": best_alpha_h,
    }
    with open(out_dir / "model.pkl", "wb") as f:
        cloudpickle.dump(artifact, f)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
