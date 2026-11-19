# Experiment Framework

How differencing experiments are run, what they write, and how to compare them.
Scripts: `run_differencing_experiment.py` (one run), `run_all_differencing.py`
(batch grid), `explore_differencing_experiments.ipynb` (comparison, diagnostics,
deployment). Context: [`model_building_overview.md`](model_building_overview.md);
the math of the model is in
[`differencing_and_absolute_reconstruction.md`](differencing_and_absolute_reconstruction.md).

## 1. Running one experiment

```powershell
# from training/, with the .venv11 environment (README.md: py -3.11 -m venv .venv11)
python run_differencing_experiment.py --y_column "Steam__kWh/T_" `
    --data_path ../data/costimiser_turnup.parquet `
    --model ridge --lag_days 0.1 --ewm_halflife 6 `
    --conditional_independence --alpha 0.06
```

`utility.py` lives in the parent directory; the script inserts
`Path(__file__).parent.parent` into `sys.path`.

### 1.1 Arguments (`argparse` in `main()`)

| Argument | Default | Meaning |
| --- | --- | --- |
| `--y_column` | required | Target (must exist in `PIPELINE_PREFIXES`, `BLACK_LIST`, `CREATED_VARIABLE_CANDIDATES`, `CONTROL_VARS`) |
| `--data_path` | required | Parquet file |
| `--model` | `ridge` | `ridge`, `splines_ridge`, `realmlp`, `gp`, `gam`, `hgbr`, `fs_ridge` |
| `--lag_days` | `0.25` | Differencing lag in days (6 h) |
| `--ewm_halflife` | `0` | **Integer hours.** `0` means `lag_hours / 3`; otherwise the number of hours (stored as e.g. `"6h"`) |
| `--spline_knots`, `--spline_degree` | `6`, `3` | Only for `splines_ridge` |
| `--allow_cross_grammage` | off | Allow pairs that straddle a grammage change (and keep `grammage` as one-hot feature) |
| `--fs_k_features` | `15` | Top-k for `fs_ridge` (mutual information) |
| `--conditional_independence` | off | Select predictors with the CI search over `CONSIDERED_VARS`; replaces `CONTROL_VARS[target]` |
| `--max_states` | `5000` | CI: max subsets visited |
| `--max_tests` | `500_000` | CI: max partial-correlation tests |
| `--max_k` | `3` | CI: max conditioning-set size |
| `--alpha` | `0.05` | CI: conditional significance level |

Module constants that act as knobs (edit the file): `GRAMMAGES`, `TEST_SIZE = 0.20`,
`ALPHAS = np.logspace(0, 4, 30)`, `DEFAULT_LAG_SWEEP = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 7, 14]`,
the RealMLP settings (`REALMLP_HIDDEN_SIZES = [64, 64]`, `REALMLP_LR = 0.0001`,
`REALMLP_BATCH_SIZE = 32`, `REALMLP_N_EPOCHS = 256`, `REALMLP_N_REFIT = 5`,
`REALMLP_VAL_FRACTION = 0.20`), and the per-target dictionaries
`PIPELINE_PREFIXES`, `CI_PIPELINE_PREFIXES`, `BLACK_LIST`, `CREATED_VARIABLE_CANDIDATES`.
The CI marginal level (`alpha_marginal=0.70`) and the pair tolerance (`0.35`) are
hard-coded.

### 1.2 Stages printed by the script

1. Load, `_feature_engineering(clip=True)`, per-target row filters.
2. Optional CI selection (`build_ci_differenced_frame`, `select_subset_survivors`,
   `independence_diagnostics`).
3. Preprocessing: feature list, `build_pre_estimator(apply_ewm=False, include_grammage=args.allow_cross_grammage)`.
4. EWM filtering of `feat_list` and the target per segment.
5. `pre_estimator.fit(first 80%)`, transform all rows.
6. Informational lag sweep on train rows (`lag_sweep.csv`).
7. Pairing at `--lag_days`, fit, metrics, plots.
8. Grade baseline, save `model.pkl`, refit and save `model_full.pkl`.

---

## 2. Output layout

Root: `experiments/differencing/<y_column with "/" -> "_">/<yymmddHHMMSS>/`
(relative to the current directory when the script is started). Example for
`Steam__kWh/T_`: `experiments/differencing/Steam__kWh_T_/261118214520/`. The run
id is the start time (`%y%m%d%H%M%S`) and is also the suffix of the bundle's
`model_version`.

| File | Content |
| --- | --- |
| `results.json` | Parameters, data sizes, metrics, CI diagnostics (below) |
| `model.pkl` | cloudpickle dict fitted on **train** contrasts: `model`, `pre_estimator`, `feat_list`, `grade_baseline`; plus `selected_features` for `fs_ridge`; plus `scaler` for `realmlp` and `gam` |
| `model_full.pkl` | Same keys, regression refit on **all** contrasts (`grade_baseline_full`). Intended for deployment, not evaluation. The pre-estimator is the same fitted-on-80% object |
| `lag_sweep.csv` | `lag_days, cv_rmse, persistence_rmse, cv_r2, n_contrasts` for each lag in `DEFAULT_LAG_SWEEP` with at least 200 train contrasts (informational only) |
| `mi_scores.csv` | `fs_ridge` only: `feature, mi_score, selected` |
| `dX_train.parquet`, `dX_test.parquet` | Differenced features (columns `feature_cols`) plus `AB_Grade_ID` of the later row |
| `dy_train.parquet`, `dy_test.parquet` | Differenced target (`dy_train`, `dy_test`) |
| `turnup_ts.parquet` | The EWM-filtered frame used for modelling |
| `turnup_ts_raw.parquet` | Same rows before EWM (after feature engineering) |
| `timeseries.png`, `scatter.png` | Test-contrast plots vs EWM `y` and raw `y` (gaps larger than 12 h are broken) |
| `code/` | Copies of `run_differencing_experiment.py`, `config.py`, `data_cleaning.py`, `feature_selection.py`, `preprocessing.py`, `state_estimation.py` (not `utility.py` and not `ci_feature_selection.py`) |

### 2.1 `results.json`

```json
{
  "start_time": "...", "end_time": "...", "duration_seconds": 89.5,
  "parameters": {
    "y_column": "...", "data_path": "...", "model": "ridge", "lag_days": 0.1,
    "apply_ewm_filter": true, "apply_ewm_filter_y": true, "ewm_halflife": "6h",
    "allow_cross_grammage": false, "conditional_independence": true,
    "ci_max_states": 5000, "ci_max_tests": 500000, "ci_max_k": 3, "ci_alpha": 0.06,
    "spline_knots": null, "spline_degree": null, "ridge_alpha": 1487.35,
    "fs_k_features": null, "realmlp_params": null
  },
  "data": {"n_samples": 3214, "split": 2571, "n_features": 21, "feature_cols": ["..."],
           "n_train_contrasts": 2168, "n_test_contrasts": 512},
  "metrics": {"dy": {"rmse":..,"mae":..,"r2":..},
              "ewm_persistence": {}, "ewm_model": {}, "raw_persistence": {}, "raw_model": {},
              "gain_over_persistence_ewm": 0.016, "gain_over_persistence_raw": 0.157},
  "ci_selection": { "metadata": {}, "selected_feature_independence": [], "omitted_separators": [], "fixed_audit": [] }
}
```

(`apply_ewm_filter*` are constants `true`. `ci_selection` is `null` without CI;
its structure is described in [`ci_feature_selection.md`](ci_feature_selection.md)
section 10. Numbers above are from run `261118214520`.)

Metric definitions (all on **test** contrasts):

| Key | Prediction | Compared with |
| --- | --- | --- |
| `dy` | `pred_dy` | `dy` |
| `ewm_persistence` | `y_ewm(t-k)` | `y_ewm(t)` |
| `ewm_model` | `y_ewm(t-k) + pred_dy` | `y_ewm(t)` |
| `raw_persistence` | `y_raw(t-k)` | `y_raw(t)` |
| `raw_model` | `y_ewm(t-k) + pred_dy` | `y_raw(t)` |
| `gain_over_persistence_*` | R2(model) - R2(persistence) | |

In run `261118214520`: `R2(dy) = 0.234`, `ewm_model R2 = 0.949` against
`ewm_persistence R2 = 0.933`, `raw_model R2 = 0.599` against
`raw_persistence R2 = 0.442`. The high EWM R2 values are mostly the persistence
of the level; `R2(dy)` and the gains are the meaningful comparison numbers.

---

## 3. Batch runs: `run_all_differencing.py`

Edit the constants and run `python run_all_differencing.py` (sequential
`subprocess.run` of `run_differencing_experiment.py`; progress and a summary of
successes/failures are printed; no parallelism, output is not captured).

| Constant | Current value | Notes |
| --- | --- | --- |
| `DATA_PATH` | `"../data/costimiser_turnup.parquet"` | |
| `Y_COLUMNS` | `["MBS_SCT_CD"]` (others commented out) | targets |
| `LAG_DAYS` | `[0.1, 0.25]` | |
| `EWM_HOURS` | `[6]` (earlier assignment `[0, 4, 6, 8]` is overridden) | passed as `--ewm_halflife` |
| `MODELS` | `["ridge"]` | non-CI only; others commented |
| `FS_K_FEATURES` | `[5, 10, ..., 35]` | only when `fs_ridge` is in `MODELS` |
| `CONDITIONAL_INDEPENDENCE` | `[True]` | grid dimension: `[False]`, `[True]` or `[False, True]` |
| `CI_MAX_STATES`, `CI_MAX_TESTS`, `CI_MAX_K` | `5000`, `500_000`, `3` | forwarded |
| `CI_ALPHA` | `[0.01, 0.02, ..., 0.10]` | one CI run per alpha |

`build_commands()` iterates target x lag x EWM hour x variant. For each CI
variant it emits exactly one `--model ridge` run (CI is the selection, so no model
or `k` sweep); non-CI variants sweep `MODELS` (and `FS_K_FEATURES` for
`fs_ridge`). `--allow_cross_grammage` is never passed. With the current constants
the batch is `1 x 2 x 1 x 10 = 20` runs.

---

## 4. Comparing runs

`explore_differencing_experiments.ipynb` (cells in order):

| Cells | What |
| --- | --- |
| 1, 3 | Set `TARGET` (one line active) and `EWM_HALFLIFE = "6h"`; load every `experiments/differencing/<TARGET>/*/results.json` into a DataFrame (`id, model, lag_days, ewm_halflife, ridge_alpha, n_features, n_samples, n_train, n_test, fs_k_features, r2_dy, rmse_dy, r2_ewm_model, r2_ewm_persist, r2_raw_model, r2_raw_persist, gain_ewm, gain_raw`, and `ci_alpha` when present) and show a table sorted by `r2_ewm_model` |
| 5 | Bar chart per run |
| 7 | **Best run**: among runs whose `ewm_halflife == EWM_HALFLIFE`, sort by `['r2_dy', 'r2_raw_model']` descending and take the first; prints metrics and plots `lag_sweep.csv` |
| 9 | Test-period plots rebuilt from `dX_test`, `dy_test`, `turnup_ts*.parquet`, `model.pkl` |
| 21, 23, 24 | State/absolute model from the stored beta (`c(t) + beta'X(t)`), no refit |
| 25 to 28 | SHAP on `model_full` (`calculate_manual_shap`, `plotly_shap_beeswarm`) and on raw features |
| 29 to 35 | Ad hoc diagnostics for `exha_mois_1` and steam (specific steam vs exhaust moisture) |
| 36 to 42 | Deployment: `TurnupFilter` + `StateBuilder` + `ProductionModel` demo (38), bundle writer (40), equivalence tests (41), engine usage (42) |

What to compare:

- `r2_dy` and `gain_*` over the lag and alpha grid, same target, same
  `ewm_halflife`. R2 values against `ewm_*` are dominated by persistence; read
  `gain_ewm` and `gain_raw`.
- `n_features` versus `ci_alpha` (CI runs): the feature count rises with alpha
  and plateaus.
- Stability: `ci_selection.selected_feature_independence[].borderline` and
  `max_p`; `lag_sweep.csv` curves.

Illustration from the runs stored in `experiments/differencing/` when this was
written (best `r2_dy` per target and lag; all `ridge`, all `ewm_halflife = "6h"`, all CI):

| Target | Lag 0.1 d | Lag 0.25 d |
| --- | --- | --- |
| `Electricity__kWh/T_` | not run | 0.115 (alpha 0.05 to 0.10; 16 features) |
| `MBS_SCT_CD` | 0.369 (alpha 0.05 to 0.10; 21 to 22 features) | 0.403 (alpha 0.02; only alpha 0.01, 0.02 present) |
| `Starch_..._Bottom_Roll__g/m2_` | 0.393 (alpha 0.01; 13 features) | 0.392 (alpha 0.10; 15 features) |
| `Starch_..._Top_Roll__g/m2_` | 0.298 (alpha 0.10; 13 features) | 0.180 (alpha 0.10; 14 features) |
| `Steam__kWh/T_` | 0.234 (alpha 0.05/0.06; 21 features) | 0.200 (alpha 0.04 to 0.06; 20 features) |
| `Steam_power_corrected` | 0.212 (alpha 0.03/0.04; 26 features) | negative for all alphas (-0.154 to -0.369) |

(Run counts per target folder: Electricity 10, `MBS_SCT_CD` 13, Starch bottom 20,
Starch top 20, `Steam_power_corrected` 20, `Steam__kWh/T_` 16.)

### 4.1 From a chosen run to a deployed bundle

1. Pick `best_path` (cell 7 or manually).
2. Cell 40: `load_state_and_prod(best_path, use_full_model=True, ..., max_tolerance=1.0, fill='ffill')`,
   set `ENGINE_TARGET`, `UNITS`, `SCALE`, `MULTIPLIER`, `DIVISOR` if the engine
   target differs from the training target; the cell writes the bundle, the
   turnup filter and the merged manifest into `deploy/`.
3. Cell 41: equivalence checks, expected `~1e-6` or smaller.
4. Check the printed warning if the model's halflife differs from `EWM_HALFLIFE`.

---

## 5. Other experiment folders (not part of the differencing flow)

| Folder | Written by | Layout / notes |
| --- | --- | --- |
| `experiments/<target>/<yymmddHHMM>/` | `run_experiment.py` | Backfit level model: `results.json`, `convergence.png`, `prediction_timeseries.png`, `local_level.png`, `scatter.png`, `backfit_result.pkl`, `production_model.pkl`, `X_train/X_test/y_train/y_test.parquet`, `code/` (see `README.md`) |
| `other_experiments/<target>/<yymmddHHMM>/` | `run_another_experiment.py` | Variant of the backfit script |
| `experiments_runlevel/<target>/<yymmddHHMM>/` | `run_level_experiment.py` | Run-level + within-run model |
| `research_experiments/` (`ablation/`, `differencing/`, `predictions/`, `<target>/`) | `run_ablation_research.py`, `run_differencing_research.py`, `run_prediction_plots_research.py`, `run_experiment_research.py` | Research on leakage attribution, lag sweep vs bandwidth sweep, out-of-fold prediction plots. `run_experiment_research.py` writes to `research_experiments/<target>/<timestamp>`; the others' output paths were not checked |
| `experiments/differencing_13102026/`, `differencing_all/`, `differencing_copy/` | not written by the current script (names suggest snapshots/copies) | Contain per-target run folders (`differencing_copy` also `Electrical_power_MW`, `Steam_power`); not analysed |
| `deploy/`, `deploy_old/` | notebook cell 40 / `_regen_final.py` | Deployment bundles (overview section 7) |

---

## 6. Caveats / open questions

1. **Run ranking uses test metrics.** Cell 7 selects the best run by `r2_dy` on
   the test contrasts, and cell 40 then refits that configuration on all data
   (`model_full`). The reported test numbers of the chosen run are therefore
   optimistic with respect to the choice of alpha/lag/CI settings. There is no
   separate untouched hold-out in this flow.
2. **CI alpha plateau.** Many neighbouring `ci_alpha` values give identical
   feature sets and metrics (for example Electricity alpha 0.05 to 0.10 all give
   16 features, `r2_dy = 0.115`), so the sweep mostly explores a few distinct
   models. Runs are about 1 to 4 minutes each (`duration_seconds`; for example
   89 s for run `261118214520`).
3. **`ridge_alpha` differs between `results.json` and the bundle.** The run
   stores the alpha of the train-only fit (`1487.35` for `261118214520`); the
   bundle estimator is `model_full`, which repeats the alpha search on all
   contrasts (`1082.6`).
4. **Reproducibility gaps.** `results.json` does not store the active PLS prefix
   list, `alpha_marginal`, the pair tolerance, the row filters or a data
   hash; `code/` copies do not include `utility.py` or `ci_feature_selection.py`.
   The run id has one-second resolution; two runs started in the same second would
   share a folder.
5. **Partial grids.** When this was written `experiments/differencing/MBS_SCT_CD`
   had 13 runs while the current grid produces 20, with only alpha 0.01 and 0.02
   for lag 0.25. New run folders were appearing in that directory while the
   documents were being prepared (a batch was apparently running), so the counts
   and the table in section 4 are a snapshot.
6. **`model.pkl` for `fs_ridge`** stores `selected_features` and a fitted
   estimator that was trained on the selected columns only; consumers must
   subset the transformed columns by name (the notebook and `load_state_and_prod`
   do this through `selected_features`).
7. **`--ewm_halflife` is an integer** (hours); sub-hour halflives can only be
   obtained through the `0` default (`lag/3`), for example lag `0.1` gives
   `"48min"`.
