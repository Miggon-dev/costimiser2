"""
Attribute the reported backfit performance to its sources.

The backfit reports regression-only R2 ~ 0.58 on raw y. Plain Ridge on the same
data, same features, same terminal split reports ~ -0.02. Something in between
accounts for a gap of ~0.6, and only part of it is the modelling idea.

Three candidate sources, made into independent dials:

  A. FEATURE SELECTION scored on the reported block. CMA-ES runs thousands of
     evaluations ranking subsets by their RMSE on the very rows that get
     reported.
  B. ITERATION CHOICE scored on the reported block. The loop keeps the iteration
     with the best test RMSE.
  C. LEVEL SUBTRACTION from the training target. This is the genuine modelling
     contribution and is expected to survive.

Every variant reports on the SAME untouched final block, and every variant is
refitted on the same number of rows before that report, so "less leakage" is
never confounded with "less training data".

Usage:
    python run_ablation_research.py --y_column "Steam__kWh/T_" \
        --data_path ../data/costimier_turnup.parquet \
        --apply_ewm_filter --apply_ewm_filter_y
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime

import cloudpickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_selection import cmaes_feature_selection, fit_all_features

import data_prep_research as D
import backfit_research as B
import partialling_research as P


TEST_SIZE = 0.20
VAL_SIZE = 0.25          # fraction of the non-test rows held out for choices
K_RANGE = (3, 15)
MAX_EVALS = 3000
POPSIZE = 24
SEED = 42


def main():
    ap = argparse.ArgumentParser(description="Leakage ablation for the backfit result")
    ap.add_argument("--y_column", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--apply_ewm_filter", action="store_true")
    ap.add_argument("--apply_ewm_filter_y", action="store_true")
    ap.add_argument("--exclude_mediators", action="store_true")
    ap.add_argument("--n_iterations", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--max_evals", type=int, default=MAX_EVALS)
    ap.add_argument("--bandwidth_days", type=float, default=4.0,
                    help="Partialling bandwidth for the reference variant")
    args = ap.parse_args()

    start = datetime.now()
    out_dir = (Path("research_experiments") / "ablation"
               / args.y_column.replace("/", "_") / start.strftime("%y%m%d%H%M"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    print("\n=== Data ===")
    prep = D.prepare(
        args.y_column, args.data_path,
        apply_ewm_filter=args.apply_ewm_filter,
        apply_ewm_filter_y=args.apply_ewm_filter_y,
        exclude_mediators=args.exclude_mediators,
    )
    X, y = prep.X, prep.y
    n = len(X)

    # Chronological three-way split. `train` and `val` together are the rows the
    # original protocol would have called train.
    split_test = int(n * (1.0 - TEST_SIZE))
    split_val = int(split_test * (1.0 - VAL_SIZE))
    train_idx = np.arange(split_val)
    val_idx = np.arange(split_val, split_test)
    trainval_idx = np.arange(split_test)
    test_idx = np.arange(split_test, n)

    print(f"\ntrain    [0:{split_val}]        n={len(train_idx)}")
    print(f"val      [{split_val}:{split_test}]     n={len(val_idx)}")
    print(f"trainval [0:{split_test}]        n={len(trainval_idx)}")
    print(f"test     [{split_test}:{n}]     n={len(test_idx)}   <- reported, never used for choices")

    fixed_resolved = D.resolve_fixed_features(prep)
    feature_groups = D.pls_feature_groups(prep)

    def make_selector(cv_splits, use_selection=True):
        """Feature selection callable with its scoring block fixed by the caller."""
        def _fn(X_full, y_full, iteration):
            if not use_selection:
                return fit_all_features(X_full, y_full, cv_splits=cv_splits)
            return cmaes_feature_selection(
                X_full, y_full,
                k_range=K_RANGE,
                fixed_features=fixed_resolved,
                feature_groups=feature_groups,
                cv_splits=cv_splits,
                selection="topk",
                max_evals=args.max_evals,
                sigma0=1.0, seed=SEED, popsize=POPSIZE,
                penalty_fn=None, splines=False, verbose=False,
            )
        return _fn

    results = []

    def record(name, description, r2, rmse, extra=None):
        row = {"variant": name, "description": description,
               "r2_test": float(r2), "rmse_test": float(rmse)}
        row.update(extra or {})
        results.append(row)
        print(f"  -> {name:34s} test R2={r2:+.4f}  rmse={rmse:7.3f}")

    # -------------------------------------------------------------------------
    # 0. Floor: no selection, no level, raw target
    # -------------------------------------------------------------------------
    print("\n=== 0. Floor: plain Ridge, all features, raw target ===")
    est0 = B.refit(X, y, prep.feature_names, trainval_idx)
    p0 = est0.predict(X[prep.feature_names].values[test_idx])
    m0 = B.metrics(y[test_idx], p0)
    record("floor_plain_ridge", "all features, no selection, no level", m0["r2"], m0["rmse"])

    # -------------------------------------------------------------------------
    # 1. Selection leak only (no level): reproduces backfit iteration 1
    # -------------------------------------------------------------------------
    # At iteration 1 of the original loop the adjusted target IS the raw target,
    # so no level has been subtracted. Anything above the floor here is
    # attributable to selection alone.
    print("\n=== 1. Selection scored on TEST, no level (= original iteration 1) ===")
    sel_leaky = make_selector([(trainval_idx, test_idx)])
    fs_leak = sel_leaky(X, y, 0)
    p1 = fs_leak.final_estimator.predict(X[fs_leak.selected_features].values[test_idx])
    m1 = B.metrics(y[test_idx], p1)
    record("leak_selection_only", "CMA-ES scored on test, no level",
           m1["r2"], m1["rmse"], {"n_features": len(fs_leak.selected_features)})
    leaky_features = list(fs_leak.selected_features)

    # -------------------------------------------------------------------------
    # 2. Selection scored on VAL, no level
    # -------------------------------------------------------------------------
    print("\n=== 2. Selection scored on VAL, no level ===")
    sel_honest = make_selector([(train_idx, val_idx)])
    fs_honest = sel_honest(X, y, 0)
    honest_features = list(fs_honest.selected_features)
    est2 = B.refit(X, y, honest_features, trainval_idx)
    p2 = est2.predict(X[honest_features].values[test_idx])
    m2 = B.metrics(y[test_idx], p2)
    record("honest_selection_only", "CMA-ES scored on val, no level",
           m2["r2"], m2["rmse"], {"n_features": len(honest_features)})

    # -------------------------------------------------------------------------
    # 3. Full original protocol: both leaks + level
    # -------------------------------------------------------------------------
    print("\n=== 3. Original protocol: selection on TEST, iteration on TEST, + level ===")
    bf_leaky = B.backfit(
        X, y,
        fit_idx=trainval_idx, select_idx=test_idx, test_idx=test_idx,
        feature_selection_fn=sel_leaky,
        n_iterations=args.n_iterations, gamma=args.gamma, verbose=True,
    )
    best_leaky = next(r for r in bf_leaky.iterations if r.is_best)
    record("original_both_leaks", "selection+iteration on test, with level",
           best_leaky.r2_test, best_leaky.rmse_test,
           {"n_features": best_leaky.n_features, "best_iteration": bf_leaky.best_iteration})

    # -------------------------------------------------------------------------
    # 4. Selection on TEST but iteration on VAL
    # -------------------------------------------------------------------------
    print("\n=== 4. Selection on TEST, iteration on VAL, + level ===")
    bf_mixed = B.backfit(
        X, y,
        fit_idx=train_idx, select_idx=val_idx, test_idx=test_idx,
        feature_selection_fn=sel_leaky,
        n_iterations=args.n_iterations, gamma=args.gamma, verbose=True,
    )
    best_mixed = next(r for r in bf_mixed.iterations if r.is_best)
    record("leak_selection_honest_iter", "selection on test, iteration on val, with level",
           best_mixed.r2_test, best_mixed.rmse_test,
           {"n_features": best_mixed.n_features, "best_iteration": bf_mixed.best_iteration})

    # -------------------------------------------------------------------------
    # 5. Fully honest backfit
    # -------------------------------------------------------------------------
    print("\n=== 5. Honest: selection on VAL, iteration on VAL, + level ===")
    bf_honest = B.backfit(
        X, y,
        fit_idx=train_idx, select_idx=val_idx, test_idx=test_idx,
        feature_selection_fn=sel_honest,
        n_iterations=args.n_iterations, gamma=args.gamma, verbose=True,
    )
    best_honest = next(r for r in bf_honest.iterations if r.is_best)
    record("honest_backfit", "selection+iteration on val, with level",
           best_honest.r2_test, best_honest.rmse_test,
           {"n_features": best_honest.n_features, "best_iteration": bf_honest.best_iteration})

    # -------------------------------------------------------------------------
    # 6. Level only, no selection at all
    # -------------------------------------------------------------------------
    # Isolates C from A: is the level subtraction worth anything on its own?
    print("\n=== 6. No selection, iteration on VAL, + level ===")
    bf_nosel = B.backfit(
        X, y,
        fit_idx=train_idx, select_idx=val_idx, test_idx=test_idx,
        feature_selection_fn=make_selector([(train_idx, val_idx)], use_selection=False),
        n_iterations=args.n_iterations, gamma=args.gamma, verbose=True,
    )
    best_nosel = next(r for r in bf_nosel.iterations if r.is_best)
    record("level_only_no_selection", "all features, iteration on val, with level",
           best_nosel.r2_test, best_nosel.rmse_test,
           {"n_features": best_nosel.n_features, "best_iteration": bf_nosel.best_iteration})

    # -------------------------------------------------------------------------
    # 7. Partialling reference, no selection
    # -------------------------------------------------------------------------
    print("\n=== 7. Partialling, no selection ===")
    bw_h = args.bandwidth_days * 24.0
    # Nuisance support is the non-test rows only, so the test block's trend is
    # extrapolated rather than informed by its own targets.
    mask = np.zeros(n, dtype=bool)
    mask[trainval_idx] = True
    pdat = P.partial_out_time(X, y, prep.t_hours, bw_h, fit_mask=mask)
    est7 = B.refit(pd.DataFrame(pdat.X_tilde, columns=prep.feature_names, index=X.index),
                   pdat.y_tilde, prep.feature_names, trainval_idx)
    p7 = est7.predict(pdat.X_tilde[test_idx])
    m7 = B.metrics(pdat.y_tilde[test_idx], p7)
    record("partialled_no_selection", f"partialled bw={args.bandwidth_days}d, all features",
           m7["r2"], m7["rmse"])
    print("     (note: scored against the partialled target y~, not raw y - the")
    print("      other variants are scored against raw y, so this row measures a")
    print("      different quantity and is here as a reference, not a rival)")

    # -------------------------------------------------------------------------
    # Attribution
    # -------------------------------------------------------------------------
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("ABLATION SUMMARY (all rows scored on the same untouched test block)")
    print("=" * 78)
    print(res_df[["variant", "r2_test", "rmse_test", "description"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    def get(name):
        m = res_df.loc[res_df["variant"] == name, "r2_test"]
        return float(m.iloc[0]) if len(m) else float("nan")

    floor = get("floor_plain_ridge")
    attribution = {
        "floor (all features, no selection, no level)": floor,
        "A. gain from selection scored on test": get("leak_selection_only") - floor,
        "A'. gain from selection scored on val": get("honest_selection_only") - floor,
        "C. gain from level, no selection": get("level_only_no_selection") - floor,
        "honest total (selection+level, all on val)": get("honest_backfit") - floor,
        "reported total (both leaks + level)": get("original_both_leaks") - floor,
    }
    print("\nATTRIBUTION (change in test R2 vs the floor)")
    for k, v in attribution.items():
        print(f"  {k:48s} {v:+.4f}")
    inflation = get("original_both_leaks") - get("honest_backfit")
    print(f"\n  Inflation from scoring choices on the reported block: {inflation:+.4f}")

    # -------------------------------------------------------------------------
    # Coefficient stability of the honestly-selected features
    # -------------------------------------------------------------------------
    print("\n=== Coefficient stability of the honestly-selected features ===")
    print("R2 cannot decide this: the deliverable is gradients. A coefficient that")
    print("flips sign across time blocks is not identified, whatever the fit.")
    Xh = X[honest_features]
    stab = P.evaluate(Xh, y, prep.t_hours, bandwidth_hours=bw_h,
                      learner="ridge", variant="A", n_folds=5)
    coef_df = stab.coef_frame()
    if not coef_df.empty:
        print(coef_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        unstable = coef_df.loc[coef_df["sign_flips"] > 0, "feature"].tolist()
        print(f"\n  identified   : {len(coef_df) - len(unstable)}/{len(coef_df)}")
        if unstable:
            print(f"  NOT identified: {unstable}")

    sweep = P.bandwidth_sweep(
        Xh, y, prep.t_hours,
        [d * 24.0 for d in (1, 2, 4, 7, 14, 30)],
        learner="ridge", variant="A", n_folds=5, verbose=True,
    )
    paths_df = P.coefficient_paths(sweep)

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_df = res_df[res_df["variant"] != "partialled_no_selection"]
    colours = ["C7" if v == "floor_plain_ridge"
               else ("C3" if "leak" in v or "original" in v else "C0")
               for v in plot_df["variant"]]
    ax.barh(plot_df["variant"], plot_df["r2_test"], color=colours, alpha=0.85)
    ax.axvline(floor, ls="--", color="gray", alpha=0.8, label="floor")
    ax.set_xlabel("Test R2 (raw target, same untouched block)")
    ax.set_title(f"{args.y_column} - where the reported performance comes from\n"
                 f"red = a choice was scored on the reported block")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_dir / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    res_df.to_csv(out_dir / "ablation.csv", index=False)
    if not coef_df.empty:
        coef_df.to_csv(out_dir / "honest_coefficients.csv", index=False)
    if not paths_df.empty:
        paths_df.to_csv(out_dir / "honest_coefficient_paths.csv", index=False)
    for name, bf in [("original_both_leaks", bf_leaky),
                     ("leak_selection_honest_iter", bf_mixed),
                     ("honest_backfit", bf_honest),
                     ("level_only_no_selection", bf_nosel)]:
        bf.history_frame().to_csv(out_dir / f"history_{name}.csv", index=False)

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "start_time": start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "parameters": vars(args),
            "splits": {"train": len(train_idx), "val": len(val_idx),
                       "test": len(test_idx), "n": n},
            "ablation": results,
            "attribution": attribution,
            "inflation_from_scoring_on_test": inflation,
            "leaky_selected_features": leaky_features,
            "honest_selected_features": honest_features,
            "honest_coefficients": (coef_df.to_dict(orient="records")
                                    if not coef_df.empty else []),
        }, f, indent=2, default=float)

    with open(out_dir / "artifacts.pkl", "wb") as f:
        cloudpickle.dump({
            "ablation": res_df,
            "honest_features": honest_features,
            "leaky_features": leaky_features,
            "honest_coefficients": coef_df,
            "coefficient_paths": paths_df,
        }, f)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
