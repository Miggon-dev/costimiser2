# Training Pipeline: Iterative Backfitting (CMA-ES + Ridge + Local Level)

## Setup

Activate the virtual environment:

```
.venv\Scripts\activate.bat
```

## Run Experiment

```
python run_experiment.py --y_column "Steam__kWh/T_" --data_path ./data/costimier_turnup.parquet
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--y_column` | str | *required* | Target column name (e.g. `"Steam__kWh/T_"`) |
| `--data_path` | str | *required* | Path to the parquet data file |
| `--apply_ewm_filter` | flag | off | Enable segment-aware EWM smoothing on features |
| `--filter_y` | flag | off | Apply EWM filter to target (for training only; metrics use raw y) |
| `--n_iterations` | int | 10 | Maximum backfitting iterations |
| `--gamma` | float | 1.0 | Level damping factor (0 = no subtraction, 1 = full) |
| `--splines` | flag | off | Enable SplineTransformer after first iteration |
| `--fixed_features` | str | see below | Comma-separated feature names always included |

Default fixed features: `ambient_temp_C, exha_mois_1, exha_mois_2, inlet_temp_1, inlet_temp_2, linepressure_1, fabric_tension_1, gas_decu_1, gas_decu_2, gas_decu_3, grammage`

### Examples

```bash
# Minimal
python run_experiment.py --y_column "Steam__kWh/T_" --data_path ./data/costimier_turnup.parquet

# With splines and damping
python run_experiment.py --y_column "Steam__kWh/T_" --data_path ./data/costimier_turnup.parquet --splines --gamma 0.8

# With EWM on features and target
python run_experiment.py --y_column "Steam__kWh/T_" --data_path ./data/costimier_turnup.parquet --apply_ewm_filter --filter_y

# Custom fixed features
python run_experiment.py --y_column "Steam__kWh/T_" --data_path ./data/costimier_turnup.parquet --fixed_features "ambient_temp_C,retention,grammage"
```

## Output

Results are saved to `experiments/<y_column>/<YYMMDDHHmm>/`:

| File | Description |
|------|-------------|
| `results.json` | Parameters, metrics, timing, data ranges, selected features |
| `convergence.png` | Test RMSE, MAE, R2 per iteration |
| `prediction_timeseries.png` | Actual vs Ridge vs Ridge+Level (best iteration) |
| `local_level.png` | Estimated local level on test period |
| `scatter.png` | Actual vs Predicted scatter (Ridge and Ridge+Level with metrics) |
| `backfit_result.pkl` | Full backfit result dictionary (cloudpickle) |
| `production_model.pkl` | Final pipeline (preprocessing + Ridge) fitted on all data (cloudpickle) |
| `X_train.parquet` | Training features |
| `X_test.parquet` | Test features |
| `y_train.parquet` | Training target |
| `y_test.parquet` | Test target |
| `code/` | Copy of all source files used in the experiment |

### results.json structure

```json
{
  "start_time": "2025-08-11T14:30:00",
  "end_time": "2025-08-11T14:35:00",
  "duration_seconds": 300.0,
  "data_start": "2025-11-15 ...",
  "data_end": "2026-07-10 ...",
  "train_start": "...",
  "train_end": "...",
  "test_start": "...",
  "test_end": "...",
  "parameters": { ... },
  "metrics": {
    "ridge_only": { "rmse": ..., "mae": ..., "r2": ... },
    "ridge_level": { "rmse": ..., "mae": ..., "r2": ... }
  },
  "selected_features": [ ... ]
}
```

## Algorithm

Iterative backfitting decomposes the target into regression + latent state:

```
y_t = X_t'*beta + s_t + epsilon_t
```

Each iteration:
1. Fit Ridge on adjusted target: `beta = Ridge(X, y - gamma * level)`
2. Compute residuals: `r = y - X*beta`
3. Fit Local Level (Kalman smoother) on residuals: `level = LocalLevel(r)`
4. Update adjusted target for next iteration

Early stopping (patience=2) returns the best iteration based on Ridge RMSE. The production model is the final preprocessing + Ridge pipeline fitted on all data with the level-adjusted target.

### EWM Filtering

When `--apply_ewm_filter` is enabled, features are smoothed using an exponentially-weighted mean that resets at segment boundaries (grade changes or time gaps > 12h).

When `--filter_y` is additionally enabled, the target is also EWM-filtered for training/fitting purposes. However, all evaluation metrics are always computed against the **raw (unfiltered) target** to provide honest performance estimates.

## Explore Experiments

Use `explore_experiments.ipynb` to compare results across runs:
- Table of all experiments sorted by Ridge RMSE
- Best experiment details (parameters, features, plots)
- Iteration history

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Target-to-feature mappings, variable group definitions |
| `data_cleaning.py` | Outlier detection, design matrix construction, helpers |
| `feature_selection.py` | CMA-ES feature selection, Ridge with alpha CV |
| `preprocessing.py` | PLS pipeline, pre-estimator builder, final pipeline builder |
| `state_estimation.py` | Local Level estimation, iterative backfitting, orthogonal variants |
| `run_experiment.py` | CLI script for running experiments |
| `explore_experiments.ipynb` | Notebook for comparing experiment results |
