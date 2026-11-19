# Conditional-Independence Feature Selection

This document describes, in detail, the conditional-independence (CI) feature
selection algorithm implemented in `ci_feature_selection.py`. The algorithm
chooses a subset of predictors for a given target by keeping only those
variables whose statistical relationship with the target **survives
conditioning on the other variables in the current subset**. Variables that can
be "explained away" by some subset of the already-selected variables are
dropped.

The entry point for the search is `select_subset_survivors(...)`, which
standardises the data, resolves duplicate columns, builds a correlation matrix,
and then delegates the state-space search to `_search_subset_states(...)`. The
conditional-independence test itself is `_partial_test(...)`.

In the differencing pipeline the search is invoked from
`run_differencing_experiment.py` (the `--conditional_independence` path) on a
differenced frame built over the per-target `config.CONSIDERED_VARS[target]`
candidate pool, with `config.FIXED_VARIABLES[target]` forced to remain.

---

## 1. Core idea: survival under conditioning

For a candidate predictor `X` and a target `Y`, we ask whether `X` still carries
information about `Y` once we account for a conditioning set `Z` drawn from the
other currently selected predictors.

- If there exists **any** subset `Z` of the other selected variables such that
  `X` and `Y` become conditionally independent given `Z` (i.e. the partial
  correlation is not significantly different from zero), then `Z` is a
  **separating set** for `X`. The relationship between `X` and `Y` does not
  survive conditioning — `X` is considered redundant and is removed.
- If **no** such separating subset exists (up to the search depth `max_k`), the
  relationship survives — `X` is retained.

The target of the search is a **self-consistent subset** `S` in which:

1. **Every member survives** — no selected variable has a separating set drawn
   from the other members of `S` (the "survivor" / backward condition).
2. **Every omitted variable is explained** — each variable outside `S` has a
   separating set drawn from `S` itself (the "no eligible additions" / forward
   condition).

A subset that satisfies both conditions up to depth `max_k` is reported as
`complete`. Completion is bounded subset-test consistency, not a guarantee of
global optimality or causal identification (see §8).

---

## 2. The conditional-independence test (`_partial_test`)

The test measures the **partial correlation** between `X` and `Y` given a
conditioning set `Z`, then converts it to a p-value.

Inputs:

- `corr` — the correlation matrix of the standardised data. Row/column `0` is
  the target `Y`; the remaining rows/columns are the candidate representatives.
- `n` — the number of observations (paired, differenced rows).
- `x` — index of the candidate variable.
- `z` — tuple of indices of the conditioning variables.

Procedure:

1. Assemble the sub-matrix `c` for the ordered index list `[Y, X, Z...]`.
2. If `Z` is non-empty, compute the **residual (Schur complement)** that removes
   the linear effect of `Z` from the `(Y, X)` block:

   `residual = c[YX, YX] - c[YX, Z] · zz⁻¹ · c[Z, YX]`

   where `zz = c[Z, Z]`. The solve is done with `np.linalg.solve`.
   If `Z` is empty, `residual` is just the `(Y, X)` correlation block — i.e.
   the ordinary (marginal) correlation.
3. **Numerical guards:**
   - If the smallest eigenvalue of `zz` is `<= 1e-10 · largest eigenvalue`, the
     conditioning block is effectively rank-deficient (collinear conditioners);
     the test returns `(nan, nan)` — an *invalid* result.
   - If either residual variance on the diagonal is `<= 1e-10`, the test also
     returns `(nan, nan)`.
4. The **partial correlation** is

   `r = residual[Y,X] / sqrt(residual[Y,Y] · residual[X,X])`, clipped to `[-1, 1]`.
5. Convert to a two-sided p-value with a Student-t statistic:
   - degrees of freedom `df = n - |Z| - 2`
   - `t = |r| · sqrt(df / (1 - r²))`
   - `p = 2 · student_t.sf(t, df)`

Interpretation: a **large p-value means conditional independence** (the partial
correlation is not distinguishable from zero given `Z`), so `Z` separates `X`
from `Y`. A small p-value means the relationship survives conditioning on `Z`.

`(nan, nan)` results are not treated as separation; they are tracked as
*invalid* tests and make any "complete" claim provisional (`unknown` flag).

---

## 3. Two significance thresholds: marginal vs conditional

The search uses two alphas:

- `alpha_marginal` — applied when the conditioning set is **empty** (`Z = ()`),
  i.e. the plain marginal correlation of `X` with `Y`.
- `alpha` (the conditional alpha) — applied when `Z` is **non-empty**.

A conditioning set `Z` counts as a separator for `X` when:

- `p > alpha_marginal` if `Z` is empty, or
- `p > alpha` if `Z` is non-empty.

Using a **looser** `alpha_marginal` (e.g. `0.70` in the differencing run, vs
`alpha = 0.05`) makes it easy to drop variables that have essentially no
marginal relationship with the target at all, while still requiring strong
evidence before declaring a conditional separation. These values are passed from
`run_differencing_experiment.py` as `alpha=0.05, alpha_marginal=0.70`.

---

## 4. Finding a separating set for one variable (`inspect`)

`inspect(x, others)` decides whether variable `x` can be separated from the
target using conditioning subsets drawn from `others` (the other selected
variables, or the whole selected set when testing an omitted candidate).

Steps:

1. **Separator cache reuse.** A per-variable cache `known[x]` stores separating
   sets discovered earlier. If any cached separator `z` is still a subset of
   `others`, it is reused immediately — no test is run. This is valid because a
   set that separated `x` before still separates it as long as all its members
   are present.
2. **Subset enumeration by increasing size.** Otherwise, enumerate conditioning
   sets `z` of size `k = 0, 1, ..., min(max_k, |others|)`, in that order
   (smallest first), using `itertools.combinations` over the sorted `others`.
   - Run the CI test `ci(x, z)` (a cached wrapper around `_partial_test`).
   - If the result is invalid (`nan`), mark `invalid = True` and skip it.
   - Track the **weakest** evidence seen so far — the `(z, r, p)` with the
     smallest `|r|` — which later serves as the best single piece of evidence
     if no separator is found.
   - As soon as a `z` passes the separation threshold (`p > alpha` rules of §3),
     record it in `known[x]` and return it as the separating set.
3. **Return values:**
   - a separating set `(z, r, p)` if one was found (the relationship does **not**
     survive), otherwise `None`;
   - the `weakest` evidence tuple (used for diagnostics and search ordering);
   - the `invalid` flag indicating whether any test along the way was `nan`.

Because sizes are tried smallest-first, the algorithm prefers **parsimonious**
separating sets — the minimal conditioning needed to explain a variable away.

---

## 5. State-space search over subsets (`_search_subset_states`)

The subset `S` is found by a best-first search over candidate subsets, starting
from the forced `fixed` variables and adding/removing one variable at a time.

Setup:

- `fixed` — indices of `fixed_vars`, always kept (exempt from removal).
- A memoised CI wrapper `ci(x, z)` enforces the global budget `max_tests`; once
  the budget is hit it raises `Limit` and the search stops with
  `stop_reason = 'test_limit'`.
- A priority queue (`heapq`) ordered to **prefer larger admissible subsets**,
  with a monotonic serial number breaking ties deterministically.

The search begins by pushing the `fixed` set as the initial state.

For each popped state `S` (until `max_states` states are visited):

1. **Survivor (backward) check.** For every non-fixed member `x` of `S`, call
   `inspect(x, S - {x})`.
   - If `x` has a separating set within the other members, it is **blocked** —
     it should not be in `S`. Record it.
   - Otherwise record its weakest-evidence strength.
2. **Remove blocked members.** If any member is blocked, for each such `x` push
   the successor state `S - {x}` (action `remove`) and move on — this state is
   not self-consistent, so it is not a candidate solution.
3. **Forward (eligibility) check.** If no member is blocked, examine every
   variable `x` **outside** `S` with `inspect(x, S)`:
   - If `x` has a separating set within `S`, it is correctly **rejected** (an
     omitted separator) — it belongs outside `S`.
   - If `x` has **no** separating set within `S`, it is **eligible**: it still
     relates to the target given everything in `S` and arguably should be added.
4. **Record the state.** Track the best state seen so far as the one with the
   **fewest eligible omissions** (fewest variables that "want in" but are not
   there). This is the fallback answer if no complete state is found.
5. **Termination on completeness.** If there are **no eligible omissions and no
   invalid tests** (`unknown` is false), `S` satisfies both conditions and the
   search stops with `stop_reason = 'complete'`.
6. **Expansion.** Otherwise, sort the eligible candidates by **strongest
   evidence first** (largest `|r|`) and push each successor `S ∪ {x}`
   (action `add`). Combined with the "prefer larger subsets" queue ordering,
   the search drives toward larger, self-consistent subsets while still being
   able to backtrack by removing blocked members.

Stop reasons recorded in metadata:

- `complete` — a fully self-consistent subset was found.
- `test_limit` — the `max_tests` CI-test budget was exhausted.
- `state_limit` — `max_states` states were visited with the queue non-empty.
- `no_verified_solution_in_reachable_states` — the queue emptied without a
  complete state.

When the search does not complete, the **best** provisional state (fewest
eligible omissions) is returned and flagged as provisional.

---

## 6. Preprocessing inside the selector

Before the search runs, `select_subset_survivors` does several things that
affect correctness:

- **Column ordering.** Data is reordered as `[target] + fixed_vars + others`.
- **Validation.** The target must be present, column names unique, and the data
  finite and non-constant (`std > 1e-12`). There must be more than `max_k + 3`
  observations.
- **Standardisation.** Every column is centred and scaled to unit variance, so
  correlations and partial correlations are computed on standardised data.
- **Alias resolution.** Columns that are **numerically identical** after
  standardisation (equal up to sign, within `1e-7`) are collapsed to a single
  representative; the duplicates are recorded in `aliases_df`. If a *fixed*
  variable is found to be a duplicate of another column, the function raises
  (fixed variables may not be numerically redundant).
- **Correlation matrix.** A single correlation matrix over
  `[target] + representatives` is computed once and reused for every CI test via
  sub-matrix indexing — this is what makes thousands of tests cheap.

---

## 7. Fixed variables

Variables listed in `fixed_vars` (from `config.FIXED_VARIABLES[target]`) are:

- always included in the initial state and never removed by the survivor check;
- excluded from the "blocked member" logic (they cannot be pruned);
- still **audited** at the end. The audit records, for each fixed variable,
  whether it would have survived (`passes_tested_subsets`), would have been
  blocked (`blocked_but_fixed`), had invalid tests (`invalid_tests`), or could
  not be fully audited within budget (`audit_budget_exhausted`). This audit is
  returned as `fixed_audit_df` so the exemption is transparent.

In the differencing pipeline, fixed variables are mapped through the PLS
structure (passthrough name or score names) before being handed to the selector,
so the "always keep" guarantee holds in the differenced feature space.

---

## 8. Outputs

`select_subset_survivors` returns a dict containing:

- `selected_vars` — the chosen predictor names (the subset `S`).
- `selected_df` — per-selected-variable survival evidence (variable,
  conditioning set, `r`, `p`, role), including the fixed-variable audit rows.
- `fixed_audit_df` — the fixed-variable survival audit (see §7).
- `separators_df` — omitted variables together with the conditioning set that
  separated each from the target.
- `unresolved_df` — eligible omissions that remained when the search stopped
  (non-empty only when the result is provisional).
- `aliases_df` — numerically duplicate columns and their representatives.
- `history_df` — the add/remove path from the initial state to the final subset.
- `metadata` — target, `complete` flag, `stop_reason`, counts
  (`selected_count`, `states_visited`, `ci_tests`), the alphas, `max_k`, the
  number of unblocked omissions, and whether any invalid tests occurred.

**Important caveats** (stated in the code's own metadata note):

- A `complete` result means bounded subset-test consistency up to `max_k`, **not**
  a globally optimal subset and **not** causal identification.
- Any incomplete output is **provisional**.
- Invalid (`nan`) tests — typically from collinear conditioning sets — set the
  `unknown` / `invalid_survivor_or_candidate_tests` flag and prevent a
  completeness claim even if no separators were found.

---

## 9. How it is wired into the differencing experiment

In `run_differencing_experiment.py` (the `--conditional_independence` branch):

1. The per-target candidate pool `config.CONSIDERED_VARS[target]` and
   `config.FIXED_VARIABLES[target]` are resolved.
2. `build_ci_differenced_frame(...)` builds the **same differenced frame** the
   model uses (identical EWM smoothing, same-grammage lag pairing, same filters)
   over the candidate pool, using the same PLS group structure
   (`active_prefixes`) as the fitted model. This guarantees the selector can only
   search over features the model can actually use.
3. Fixed variables are mapped to their representation in the differenced frame
   (passthrough name, or PLS score names for a PLS source). A fixed variable
   that is neither raises an error.
4. `select_subset_survivors(...)` runs with `alpha=0.05, alpha_marginal=0.70`
   and the configured `max_k`, `max_states`, `max_tests`.
5. The selected transformed features are mapped **back** to their raw/source
   variables (PLS scores expand to their source columns). The fixed raw
   variables are always appended.
6. The resulting `ci_selected_vars` **replace** `CONTROL_VARS[target]` for that
   run, so the model is fit on the CI-selected predictors.

---

## 10. Independence diagnostics in `results.json` (confounder analysis)

A variable can **survive** the CI search and still be something you suspect is a
confounder. Survival only means that, on the differenced training data and up to
depth `max_k`, no conditioning subset drawn from the other selected variables
pushed its partial-correlation p-value above the threshold. Whether to keep it is
a modelling/causal judgement the algorithm does not make for you. To support that
judgement, each `--conditional_independence` run now writes an independence
report into `results.json` under the `ci_selection` key.

This is produced by `independence_diagnostics(...)` in `ci_feature_selection.py`,
called right after selection with the **same** CI frame, fixed set, `max_k`, and
alphas (`alpha=0.05`, `alpha_marginal=0.70`) as the selection itself, so the
numbers are directly comparable. It is **read-only**: it reports strength, it
does not change which variables were selected.

### 10.1 Structure of `results.json["ci_selection"]`

```json
{
  "metadata": { "...": "the selector's metadata (complete, stop_reason, alphas, counts)" },
  "selected_feature_independence": [ { "per selected feature": "..." } ],
  "omitted_separators": [ { "variable": "...", "conditioning_set": ["..."], "r": 0.0, "p": 0.9 } ],
  "fixed_audit": [ { "variable": "...", "status": "passes_tested_subsets | blocked_but_fixed | ..." } ]
}
```

In non-CI runs the whole `ci_selection` value is `null`.

### 10.2 Per-feature metrics (`selected_feature_independence`)

For every selected (transformed) feature:

| Field | Meaning |
| --- | --- |
| `variable` | the feature name (PLS score or passthrough) |
| `role` | `selected` (ordinary survivor) or `fixed` (forced in — see below) |
| `marginal_r`, `marginal_p` | correlation with the target with **no** conditioning |
| `min_abs_partial_r` | smallest `|partial r|` over all tested conditioning subsets — how close it ever came to looking conditionally independent (0 = fully explained away) |
| `max_p` | largest p-value reached — how close it came to the separation threshold |
| `weakest_conditioning_set` | the subset `Z` that achieved `min_abs_partial_r`; **these are the variables that most reduce the relationship — usually the confounding path** |
| `survives` | `True` if no tested subset separated it |
| `separating_set` | the first subset that separated it, or `null` if it survived |
| `margin_to_alpha` | `alpha - max_p` for a survivor: how close it was to being dropped (small positive = borderline). `null` if it did not survive |
| `borderline` | `True` if it survived but `margin_to_alpha` is within 20% of `alpha` |
| `n_invalid_tests` | count of non-evaluable (`nan`) tests (e.g. collinear conditioners) |

Entries are sorted strongest-survivor first (smallest `max_p`), so the most
fragile survivors appear at the bottom and borderline ones are also printed to
the console at the end of the run.

### 10.3 How to read a suspected confounder

1. **Find the variable** in `selected_feature_independence`. If it is a PLS
   source, look at the score feature(s) that carry it (the `weakest_conditioning_set`
   and the mapping printed as "PLS replacements").
2. **Look at `max_p` and `margin_to_alpha`.**
   - `max_p` far below `alpha` (e.g. `1e-6`) and `margin_to_alpha` near `alpha`:
     the variable survives **robustly**. Conditioning on the other selected
     variables never came close to removing it, so its relationship with the
     target is not explained by them on this data.
   - `max_p` just under `alpha` and `borderline = true`: it **barely** survived.
     It is a strong candidate for removal; a slightly stricter `alpha`, more
     data, or a larger `max_k` could flip it.
3. **Read `weakest_conditioning_set`.** This is the subset that most weakened the
   relationship. If a variable you consider the "cause" is in there and
   `min_abs_partial_r` drops a lot relative to `|marginal_r|`, that is the
   signature of partial confounding: most — but not all — of the association is
   mediated/shared by that set.
4. **Cross-check `omitted_separators`** for the same physical quantity: if a
   closely related variable was dropped with a clean separator, that is extra
   evidence about the relationship's direction.

### 10.4 Why a known confounder can survive here, and what to do

The CI search can only separate `X` from `Y` using subsets of the **currently
selected** variables, up to size `max_k`. A confounder survives when:

- the true separating set is **not fully present** among the selected variables
  (the mediator/common-cause was itself dropped, is outside the candidate pool,
  or was compressed into a PLS score that does not isolate it), or
- the separating set is **larger than `max_k`** (the search never tries it), or
- there is genuine **direct** association left after conditioning (only partial
  confounding), so no subset drives `p` above `alpha`, or
- `alpha` is strict (`0.05`): strong evidence is required before declaring
  independence, so borderline cases are kept.

**Suggested handling, in order of preference:**

1. **Do not rely on the CI test to drop a variable you have already judged to be
   a confounder on causal grounds.** The test is a data-driven filter; a causal
   decision is a different kind of claim. If you are confident it should go, make
   the removal **explicit and deterministic** rather than hoping the statistics
   agree on every run. (The cleanest place is a per-target exclusion applied to
   the candidate pool — e.g. drop it from `config.CONSIDERED_VARS[target]` so it
   never enters the search frame — which is reproducible and auditable.)
2. **If you want the CI test to catch it,** make sure the variable that separates
   it is actually available to the search: keep that separator in the candidate
   pool, ensure it is not hidden inside a PLS group, and consider raising
   `max_k` so larger conditioning sets are tried. Then re-read `max_p`.
3. **If it is only borderline** (`borderline = true`, small `margin_to_alpha`),
   tightening `alpha` or adding data may be enough; inspect the metric before and
   after rather than toggling blindly.
4. **Keep it, but document why,** if `min_abs_partial_r` stays high under every
   subset including the suspected cause — that is evidence of a real direct
   effect, not pure confounding.

My recommendation for your case: since you have already concluded (previously)
that this variable is a confounder, prefer option 1 — remove it explicitly from
the target's candidate pool so the decision is deterministic and visible in the
config — and keep the `ci_selection` diagnostics in `results.json` as the record
of *why* (its `max_p`, `weakest_conditioning_set`, and `margin_to_alpha`) so the
choice can be reviewed later. Relying on the CI survival outcome alone is fragile:
as the metrics show, survival depends on `alpha`, `max_k`, and which separators
happen to be in the selected set on a given run.

### 10.5 Fixed variables in the report

Variables forced in via `config.FIXED_VARIABLES[target]` appear with
`role = "fixed"`. Their metrics are still computed, so you can see whether a
forced variable *would* have survived on its own. A `fixed` variable with a high
`max_p` (i.e. one that would have been separated) is being kept only because it
is pinned — useful to know when auditing the final predictor set.

---

## 11. Per-score honoring of PLS groups in the model (the `feature_cols` fix)

### 11.1 The symptom

A PLS score could appear in `results.json["data"]["feature_cols"]` (i.e. be a
model predictor) even though the CI search had put it in
`ci_selection["omitted_separators"]` (i.e. the search found a conditioning set
that separated it from the target). In the clearest case, a whole group — e.g.
`exha_mois` for `Steam_power_corrected` — had **both** its scores
(`exha_mois_1`, `exha_mois_2`) rejected by CI, yet both still showed up in
`feature_cols`.

### 11.2 The root cause

CI selection and the final model operated at **different granularities**:

- The CI search works at the level of **individual PLS scores**. In the CI
  frame, `exha_mois_1` and `exha_mois_2` are separate columns, so the search
  can accept or reject each one independently.
- The final model built its PLS groups from the **static** `active_prefixes`
  list (`CI_PIPELINE_PREFIXES[target]` in CI mode, else `PIPELINE_PREFIXES`),
  which is fixed per target and does **not** depend on what CI selected.

The model's candidate list was assembled by a loop that, for every PLS group in
`active_prefixes`, appended **all** of that group's score names unconditionally:

```python
# OLD behavior (before the fix)
for _, step in prep_pip.steps:
    if isinstance(step, GroupwisePLSTransformer):
        transformed_names = [f"{step.score_prefix}_{i}" for i in range(1, step.n_components + 1)]
        exog_vars_reduced = ordered_difference(
            unique_in_order(exog_vars_reduced + transformed_names),  # ALL scores, always
            list(step.pls_columns),
        )
```

So every score of every listed group entered `exog_vars_reduced`, then
`build_pre_estimator` saw those score names in `selected_model_features`, fired
the group, and the scores landed in `feature_cols`. CI's accept/reject decisions
on scores were simply **not fed back** into which groups/scores the model used —
`ci_selected_vars` only governed the non-PLS passthrough variables.

The one-line claim in the code that the shared `active_prefixes` ensures "CI can
only select scores the model has" is true but insufficient: it guarantees the
model has a **superset** of what CI searched; it did **not** restrict the model
to the scores CI actually kept. The feedback was one-directional.

### 11.3 The fix (option 1: exact per-score honoring)

In CI mode the model now adds **only the scores CI selected** for each group,
and drops a group entirely if CI kept none of its scores. The exact set of kept
score names is captured right after selection:

```python
# scores CI kept = members of ci_selected that map to a PLS group
ci_selected_scores = [v for v in ci_selected if v in score_to_sources]
```

and the model-side loop honors it:

```python
# NEW behavior (the fix)
if ci_selected_scores is not None:
    keep_scores = [s for s in transformed_names if s in ci_selected_scores]
    if keep_scores:
        exog_vars_reduced = ordered_difference(
            unique_in_order(exog_vars_reduced + keep_scores),  # ONLY kept scores
            list(step.pls_columns),
        )
    # group fully rejected by CI -> not added, sources not consumed here
else:
    # non-CI runs: unchanged (all scores, sources consumed)
    ...
```

Because the `GroupwisePLSTransformer` always fits `n_components` and emits all of
them, the actual pruning happens in `build_pre_estimator`: its `filter` step
keeps only the scores present in `exog_vars_reduced`, so non-selected components
are dropped while the retained component is numerically identical to the one CI
tested (same sources, same fit). The result is that `feature_cols` matches the CI
decision at the individual-score level.

**Non-CI runs are unchanged.** The new branch is guarded by
`ci_selected_scores is not None`; when `--conditional_independence` is off the
old all-scores behavior runs verbatim.

### 11.4 Verified behavior (`Steam_power_corrected`, ridge, lag 0.25d)

With the fix, a CI run produced:

- `exha_mois` (both scores rejected by CI) — **absent** from `feature_cols`; the
  group no longer fires.
- Partially kept groups honored exactly:
  - `steam_pressure`: CI kept components 1 and 4, rejected 2 and 3 →
    `feature_cols` has only `steam_pressure_1`, `steam_pressure_4`.
  - `inlet_temp`: kept 1, rejected 2 → only `inlet_temp_1`.
  - `linepressure`: kept 2, rejected 1 → only `linepressure_2`.
- Every score-like entry in `feature_cols` corresponds to a
  `selected_feature_independence` entry with `survives = true`; none appears in
  `omitted_separators`.

A non-CI baseline run on the same target still emits every group with all its
scores (e.g. all of `exha_mois_1/2`, `steam_pressure_1..4`) and writes
`ci_selection: null`, confirming default behavior is preserved.

### 11.5 Caveats to keep in mind

- **Non-contiguous components.** Option 1 can keep, say, component 2 but not
  component 1 of a group (PLS components are orthogonal, so this is valid, but
  unusual). The kept component is the same one CI evaluated; interpret such a
  group as "only this latent direction mattered after conditioning."
- **Determinism of the set.** Which scores survive still depends on `alpha`,
  `max_k`, and the rest of the selected set (see §10.4). The fix makes the model
  **faithful to** the CI result; it does not make the CI result itself more
  stable. For a variable you have judged a confounder on causal grounds, still
  prefer removing it from the candidate pool explicitly (§10.4, option 1) rather
  than relying on the search to reject it every run.
