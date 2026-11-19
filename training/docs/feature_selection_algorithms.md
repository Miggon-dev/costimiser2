# Feature Selection Algorithms

This document lists every feature-selection mechanism present in the training
code, how each works, its parameters and defaults, and where it is used. The
conditional-independence (CI) search is described in detail in
[`ci_feature_selection.md`](ci_feature_selection.md); here it is only placed in
context and its wiring/defaults are listed. Overall context:
[`model_building_overview.md`](model_building_overview.md).

## Summary

| # | Mechanism | Where | Used by | Selects on |
| --- | --- | --- | --- | --- |
| 1 | Curated lists: `CONTROL_VARS`, `BLACK_LIST`, `CREATED_VARIABLE_CANDIDATES`, `vacuum` rule | `config.py`, `run_differencing_experiment.py` | Default differencing runs | Process knowledge |
| 2 | Conditional-independence subset search | `ci_feature_selection.select_subset_survivors` | `--conditional_independence`, `run_all_differencing.py` (current grid), standalone `run_analysis` | Partial-correlation tests on differenced data |
| 3 | Fixed variables (`FIXED_VARIABLES`) | `config.py` | CI runs | Forced in |
| 4 | Mutual-information top-k (`fs_ridge`) | `run_differencing_experiment.py` | `--model fs_ridge` | Mutual information with `dy` |
| 5 | CMA-ES subset search with Ridge CV | `feature_selection.cmaes_feature_selection` | Legacy level/backfit scripts and research scripts | Time-series CV RMSE |
| 6 | Mediator exclusion | `data_prep_research.py` | Research (`exclude_mediators`) | Process knowledge (regex) |
| 7 | PLS grouping as dimension reduction | `preprocessing._PLS_REGISTRY` | All models | Supervised latent projection (see [`feature_transformations_pls_ewm.md`](feature_transformations_pls_ewm.md)) |
| 8 | Implicit selection by the learner | Ridge shrinkage, `hgbr`, `gam` | `--model` | Regularisation |

---

## 1. Curated lists (default for non-CI runs)

`main()` in `run_differencing_experiment.py` builds the feature list like this:

1. `control_source = ci_selected_vars if CI else CONTROL_VARS[Y_COLUMN]`.
2. `ctl_vars` drops every name containing `vacuum` (case-insensitive).
3. `created_vars = ordered_intersection(CREATED_VARIABLE_CANDIDATES[Y], ctl_vars)`.
   These are columns built by `utility.FeatureCreator`
   (see the transformations document), not read from the data.
4. `exog_vars_reduced` keeps names that are not in `BLACK_LIST[Y]`, not created,
   not matching `cylinder.*steam_pressure` or `cylinder.*differential_pressure`
   (regex on the data columns), and not containing `vacuum`.
5. PLS group scores are appended and their source columns removed
   (see the PLS section).

`CONTROL_VARS[target]` lives in `config.py`. Several targets are aliased:
`MBS_SCT_MD`, `MBS_Burst`, `MBS_CMT30` use the `MBS_SCT_CD` list and
`Steam_power_corrected` uses the `Steam_power` list. The lists are grouped by
process section in comments (forming wire, press section, headbox, approach flow).

Note that excluded `vacuum` variables and cylinder pressures still reach the
model through PLS groups whose sources include them (`vacuum`,
`steam_pressure`; see PLS document).

### 1.1 `BLACK_LIST`

Per-target removal list in `run_differencing_experiment.py`. Energy targets
(`Steam_power`, `Steam__kWh/T_`, `Electrical_power_MW`, `Electricity__kWh/T_`)
remove `Current_basis_weight`, `Bentonite_1/2_mass_flow__g/T_`,
`DG3_Moisture_content_Outlet_Air`, `Lip_settings`,
`Conductivity_white_water_B46`, `pH-Messung_Verdünnungswasser__2..12_pH_`,
`pH_measurement_white_water_B41`, `CO2_mass_flow__g/T_`. Starch targets use the
same list without `Current_basis_weight`. `MBS_*` use a shorter list
(`Bentonite_*`, `DG3_Moisture_content_Outlet_Air`, `pH-Messung_...`,
`CO2_mass_flow__g/T_`; `MBS_CMT30` additionally removes
`Conductivity_white_water_B46` and `Current_reel_moisture_average(reel)`).
`data_prep_research.py` carries a similar but separate `BLACK_LIST`.

---

## 2. Conditional-independence (CI) subset search

See [`ci_feature_selection.md`](ci_feature_selection.md) for the algorithm
(`_partial_test`, `inspect`, `_search_subset_states`, diagnostics, per-score PLS
honouring). Summary of how it is configured and wired:

| Parameter | Where set | Default | Meaning |
| --- | --- | --- | --- |
| `--conditional_independence` | `run_differencing_experiment.py` | off | Enable CI selection |
| `--alpha` | CLI | `0.05` | Conditional significance level |
| `alpha_marginal` | hard-coded in `main()` (`alpha_marginal=0.70`) | `0.70` | Level for the empty conditioning set |
| `--max_k` | CLI | `3` | Maximum conditioning-set size |
| `--max_states` | CLI | `5000` | Maximum subsets visited |
| `--max_tests` | CLI | `500_000` | Maximum partial-correlation tests |
| candidate pool | `config.CONSIDERED_VARS[target]` | falls back to `_BASE_CONSIDERED_VARS` | Variables searched |
| forced variables | `config.FIXED_VARIABLES[target]` | `_DEFAULT_FIXED` or per-target override | Always kept |
| PLS structure | `CI_PIPELINE_PREFIXES[target]`, falling back to `PIPELINE_PREFIXES` | per target | Groups present in both the CI frame and the model |

`FIXED_VARIABLES` per target: `Steam__kWh/T_`: `["Starch_uptake__g/m2_"]`;
`Electricity__kWh/T_`, both starch rolls: `[]`; `Electrical_power_MW`,
`Starch_uptake__g/m2_`, `MBS_*`, `Steam_power`, `Steam_power_corrected`:
`["Starch_uptake__g/m2_", "Jet/wire_ratio"]`.

Flow in `main()`:

1. `build_ci_differenced_frame(...)` builds the differenced candidate frame
   (EWM, pre-estimator, same-grammage lag pairs) over `CONSIDERED_VARS[target]`.
   Constant and non-finite columns are dropped.
2. Fixed variables are mapped to the frame: passthrough name, or the PLS score
   names when the fixed variable is a PLS source; anything else raises.
3. `select_subset_survivors(frame, Y, alpha, alpha_marginal=0.70, max_k, max_states, max_tests, fixed_vars=ci_fixed)`.
4. `independence_diagnostics(...)` reports robustness (stored in
   `results.json["ci_selection"]`, no effect on the selection).
5. Selected scores are mapped back to source variables; with the fixed
   variables they form `ci_selected_vars`, which replaces `CONTROL_VARS[Y]`.
   `ci_selected_scores` records exactly which PLS scores were kept so the model
   uses only those.

`run_all_differencing.py` sweeps `CI_ALPHA = [0.01, ..., 0.10]` with
`CI_MAX_STATES=5000`, `CI_MAX_TESTS=500_000`, `CI_MAX_K=3`. In the stored runs
the number of model features grows with alpha and then plateaus (for example
`MBS_SCT_CD`, lag 0.1 d: 14 features at alpha 0.01, 21 at 0.05, 22 at 0.10;
`Steam__kWh/T_`, lag 0.25 d: 14 at 0.01, 20 at 0.04 to 0.06, 23 at 0.10).
That follows from the test direction: a larger alpha makes
separation (declaring independence) harder, so more variables survive.

### 2.1 Standalone version: `run_analysis`

`ci_feature_selection.py` also contains a self-contained pipeline that is
independent of `run_differencing_experiment.py`:

- `run_analysis(data_path, target="MBS_SCT_CD", considered_vars=None, fixed_vars=None, ..., lag_days=.25, ewm_halflife="6h", pair_tolerance=.35, max_gap="12h", alpha_conditional=.05, alpha_marginal=.70, max_k=3, max_states=5000, max_tests=500_000, output_dir=None)`
- `prepare_differencing_data(...)` builds the differenced frame; `filter_training_period(...)` holds its own date/vacuum/DG4 filters and an end date `2026-07-01`.
- `DEFAULT_CONSIDERED_VARS` and `DEFAULT_FIXED_VARS = ["Jet/wire_ratio", "Starch_uptake__g/m2_"]` are embedded constants.

Differences from the in-script version, as written:

| Aspect | `run_differencing_experiment.build_ci_differenced_frame` | `ci_feature_selection.prepare_differencing_data` |
| --- | --- | --- |
| Pair constraint | same `grammage` (unless `--allow_cross_grammage`) | same EWM segment (`same_segment_only=True`: grammage change or gap > `max_gap`) |
| Raw rows | NaN dropped on the feature-list columns | NaN dropped on all selected raw columns |
| PLS | prefixes from `CI_PIPELINE_PREFIXES`; sources not restricted | sources restricted to considered, non-fixed variables; groups with fewer than two sources stay explicit; `n_components = min(n, len(sources))` |
| EWM halflife | derived from lag (default `lag/3`) or `--ewm_halflife` | argument `ewm_halflife`, default `"6h"` |

`.venv11/temp/fixed_subset_selection.py` is an earlier snapshot of
`ci_feature_selection.py`. Comparing the files line by line: it contains no line
that is absent from `ci_feature_selection.py`, and lacks only
`independence_diagnostics` (and a few adjacent lines). `run_fixed_subset_selection.ipynb`
in the same folder is its companion notebook and was not analysed.

---

## 3. Fixed variables

`FIXED_VARIABLES[target]` (see above) are enforced only inside the CI search
(initial state of the search; exempt from removal; audited in
`fixed_audit_df`). Additionally, `run_differencing_experiment.py` always appends
the raw fixed variables to `ci_selected_vars`. In non-CI differencing runs there
is no separate fixed list: the curated `CONTROL_VARS` fully determine the
inputs.

In the research data prep (`data_prep_research.DEFAULT_FIXED_FEATURES`) fixed
features are given in **transformed** names (for example `linepressure_1`,
`grammage`, `ambient_temp_C` for `Steam__kWh/T_`) and are forced into every
candidate subset; `resolve_fixed_features` matches `grammage` one-hot columns by
substring.

---

## 4. Mutual-information top-k (`--model fs_ridge`)

In `run_differencing_experiment.py`:

1. `k = min(--fs_k_features, number of differenced features)`; default
   `--fs_k_features 15`.
2. `SelectKBest(score_func=mutual_info_regression, k=k)` is fitted on the **train
   contrasts only** (`dX[tr_sel]`, `dy[tr_sel]`).
3. A Ridge (`GridSearchCV`, `ALPHAS`, `TimeSeriesSplit(5)`) is fitted on the
   selected columns.
4. `mi_scores.csv` (feature, `mi_score`, `selected`) is written to the run folder.
5. `preprocessing.prune_pre_estimator(...)` back-propagates the selection: PLS
   groups whose scores are all unselected and created features that do not
   reach the model are removed, `feat_list` shrinks to the minimal raw inputs,
   and the pruned pipeline is verified to reproduce the selected columns to
   `tol=1e-6`; otherwise the full pipeline is kept (`[prune] kept full pipeline`).
6. `model.pkl` stores `selected_features` so the deployment code subsets the
   transformed columns by name.

Mutual information is scored feature by feature; it does not account for
redundancy between features or for conditioning (unlike the CI search).
`run_all_differencing.py` sweeps `FS_K_FEATURES = [5, 10, 15, 20, 25, 30, 35]`
when `"fs_ridge"` is in `MODELS` (currently commented out).

---

## 5. CMA-ES subset search (`feature_selection.py`)

Used by the level/backfit family (`run_experiment.py`, `run_another_experiment.py`,
`run_level_experiment.py --model cmaes`, `backfit_research.py`,
`run_ablation_research.py`). It is not used by `run_differencing_experiment.py`.

`cmaes_feature_selection(X, y, *, k_target=None, k_range=None, fixed_features=None, feature_groups=None, cv_splits=None, selection="topk", max_evals=3000, sigma0=1.0, popsize=None, seed=42, penalty_fn=None, alphas=None, splines=False, spline_n_knots=4, spline_degree=2, n_jobs=1, verbose=True)`

How it works:

1. **Atoms.** `_AtomMap.build` turns each entry of `feature_groups` (typically a
   PLS group, e.g. `{"gas_decu": ["gas_decu_1", "gas_decu_2", "gas_decu_3"]}`)
   into one atom that is selected all-or-nothing; every other free feature is
   its own atom. `fixed_features` are always included. See
   `data_prep_research.pls_feature_groups`.
2. **Search vector.** CMA-ES (`cma.CMAEvolutionStrategy`, start `0`, `sigma0`)
   optimises a real vector `z` with one coordinate per atom. If `k_range`
   spans more than one value one extra coordinate encodes `k`
   (`_z_to_k`: sigmoid mapped into `[k_min, k_max]`, rounded). With `k_target`
   `k` is fixed.
3. **Decoding.** `selection="topk"` keeps the `k` atoms with the largest
   `sigmoid(z)` (`_topk_mask`); `"stochastic"` samples `k` atoms with
   probabilities proportional to `sigmoid(z)` (`_stochastic_mask`).
4. **Objective.** `_evaluate_subset`: for each `(train, test)` pair in
   `cv_splits` fit `_fit_ridge` (Ridge with alpha chosen by inner
   `TimeSeriesSplit(5)` over `_DEFAULT_ALPHAS = logspace(0, 3, 20)`, optionally
   with a `SplineTransformer` on non-grammage columns) and average test RMSE.
   Loss = mean RMSE + `penalty_fn(...)`.
5. **Penalties.** `collinearity_penalty(weight=0.1)` (weight times the log
   condition number of the standardised selected columns),
   `parsimony_penalty(weight=0.5)` (weight times the feature count),
   `combined_penalty(*penalties)`.
6. **Bookkeeping.** `popsize` defaults to `max(int(4 + 3*ln(n_dims)), 8)`; a
   cache keyed by the selected index set avoids refitting repeated subsets; the
   loop stops at `max_evals` or when CMA-ES stops. The final Ridge is fitted on
   the train part of the first `cv_splits` entry (default one split, first 80%).
7. Returns `FeatureSelectionResult` (selected features, atoms, `best_rmse`,
   `best_loss`, `best_alpha`, `best_k`, `n_evals`, `n_cache_hits`,
   `final_estimator`). `fit_all_features(...)` is the no-selection drop-in.

`README.md` documents the default fixed features of `run_experiment.py`:
`ambient_temp_C, exha_mois_1, exha_mois_2, inlet_temp_1, inlet_temp_2, linepressure_1, fabric_tension_1, gas_decu_1, gas_decu_2, gas_decu_3, grammage`.

**Leakage note from the code itself.** `backfit_research.py` explains that in
`state_estimation.iterative_backfit` the caller passes
`cv_splits=[(train, test)]`, so subsets were ranked by test RMSE, and the best
iteration was also chosen on test RMSE. `backfit_research.backfit` separates
`fit_idx`, `select_idx` and `test_idx` so both decisions can be pointed at a
validation block; `run_ablation_research.py` quantifies the contribution of
each. This is why selection quality was reassessed and why the differencing
pipeline with train-only selection (or the CI search) replaced it. Note that
the CI search in the differencing script is not train-only (see overview
caveat 7).

---

## 6. Mediator exclusion (research only)

`data_prep_research.MEDIATOR_PATTERNS` (only defined for `Steam__kWh/T_`) is a
list of regexes (`^Water_Predryer`, `^Water_Afterdryer`, `^Water_flow`,
`Moisture_out_of_PreDryer`, `Dewatering`, `Moisture_content_Outlet_Air`,
`^Uhle_box_\d+_flow`, `^Starch_uptake`, `^Production_Rate`,
`^Steam_(pressure|temperature)_for_PM$`) of variables that are downstream
consequences or near-restatements of the target. With `exclude_mediators=True`
`prepare()` removes them from the control variables and from fixed features,
and drops PLS groups listed in `MEDIATOR_PLS_PREFIXES` (`exha_mois` for
`Steam__kWh/T_`). Rationale in the code: they inflate fit but give
unactionable setpoint recommendations. The differencing script has no
equivalent switch. `CI_PIPELINE_PREFIXES["Steam__kWh/T_"]` omits `exha_mois`
(the group the research code treats as a mediator), but
`CONSIDERED_VARS["Steam__kWh/T_"]` (checked by import) still contains
mediator-pattern names such as `DG1_Moisture_content_Outlet_Air`,
`Dewatering_Shoe_press`, `Production_Rate__T/h_`, `Moisture_out_of_PreDryer`,
`Starch_uptake__g/m2_` (also a fixed variable), `Steam_pressure_for_PM` and
`Steam_temperature_for_PM`. So the CI Steam model can still use mediators.

---

## 7. Implicit selection and other learners

- **Ridge** (`build_ridge_model`): `StandardScaler` + `Ridge`, alpha from
  `logspace(0, 4, 30)` by `GridSearchCV(TimeSeriesSplit(5))`. Shrinkage, not
  selection.
- `splines_ridge` (`SplineTransformer(n_knots=6, degree=3, include_bias=False)` then Ridge),
  `hgbr` (`HistGradientBoostingRegressor(max_depth=1, max_iter=500, learning_rate=0.05, min_samples_leaf=20, l2_regularization=1.0, early_stopping=True)`; additive stumps),
  `gam` (`pygam.LinearGAM`, one `s(i, n_splines=20)` per feature), `gp`
  (`ConstantKernel * RBF + WhiteKernel`, subsampled to 2000 contrasts) and
  `realmlp` (`pytabkit.RealMLP_TD_Regressor`, `hidden_sizes=[64, 64]`, lr `1e-4`,
  256 epochs). Only the linear models (`ridge`, `fs_ridge`) get a grade baseline
  and are meaningful for the `beta'X` interpretation used in deployment (the
  anchor formula itself works for any `f`, see the differencing document).
  All deployed bundles are Ridge.

---

## 8. When each mechanism is used

| Situation | Mechanism |
| --- | --- |
| Reproduce a model with expert-chosen inputs | Curated `CONTROL_VARS` run (no flag) |
| Let the data pick predictors from a large pool, with must-have variables | CI search (`--conditional_independence`), sweep `--alpha` |
| Quick data-driven top-k without conditioning logic | `--model fs_ridge --fs_k_features k` |
| Level (non-differenced) model with subset optimised by CV RMSE | `cmaes_feature_selection` through `run_experiment.py` |
| Actionable setpoint research | `data_prep_research.prepare(exclude_mediators=True)` |

---

## 9. Caveats / open questions

1. **CI uses test-period contrasts** and a larger row set than the model
   (overview caveat 7).
2. **`CONSIDERED_VARS` aliasing** changes the candidate pool of other targets
   when one is edited (overview caveat 1). In particular the two starch rolls
   share one list from which both rolls' own rod pressure and working-tank
   variables have been removed.
3. **`alpha_marginal = 0.70` is hard-coded** in `main()`; it is not exposed as a
   CLI argument and is not stored in `results.json["parameters"]` (only
   `ci_alpha`, `ci_max_k`, `ci_max_states`, `ci_max_tests` are; it is present in
   `ci_selection.metadata`).
4. **Three differently-named PLS prefix dictionaries** exist
   (`PIPELINE_PREFIXES` in `run_differencing_experiment.py`,
   `data_prep_research.py` and `ci_feature_selection.py`) with different
   contents for the same targets (for example `MBS_SCT_CD`:
   `["speedsizer_linepressure", "conc_starch"]` in the differencing script,
   `["draw", "speedsizer_linepressure", "linepressure", "conc_starch"]` in
   `data_prep_research.py`, `["speedsizer_linepressure"]` in
   `ci_feature_selection.py`). Check which one a script imports.
5. **`vacuum` handling is inconsistent with `CONTROL_VARS`.** Vacuum columns
   are listed in `CONTROL_VARS` but are always removed as direct features; they
   only reach the model through the `vacuum` PLS group, which is configured
   only for `Steam_power` and `Steam_power_corrected`.
6. **Test-period use in `fs_ridge`.** The MI selector is fitted on train
   contrasts only, but the pre-estimator's PLS and scaler are fitted on the first
   80% of rows, which is consistent; the lag sweep uses train rows only.
   No leakage issue was found for this path.
