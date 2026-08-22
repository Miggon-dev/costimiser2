"""
Differencing vs partialling on the same rows, restricted to a stationary regime.

Two questions:

  1. Is there a consistent timescale at which X is informative about y? The lag
     sweep (differencing) and the bandwidth sweep (partialling) index the same
     band through operators with almost nothing in common - one fits a nuisance,
     the other fits nothing at all. If they peak at the same timescale that is
     convergent evidence. If they disagree, one has an artefact.

  2. Does restricting to a stationary measurement regime change the answer? The
     Steam record contains at least three regimes and its measurement noise more
     than doubles across them, so a single blended number compares regimes rather
     than models.

Usage:
    python run_differencing_research.py --y_column "Steam__kWh/T_" \
        --data_path ../data/costimier_turnup.parquet \
        --apply_ewm_filter --apply_ewm_filter_y --end_date 2026-06-01
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

import data_prep_research as D
import differencing_research as F
import partialling_research as P


DEFAULT_LAGS_DAYS = [0.25, 0.5, 1, 2, 4, 7, 14, 30]
DEFAULT_BW_DAYS = [0.5, 1, 2, 4, 7, 14, 30]


def main():
    ap = argparse.ArgumentParser(description="Differencing vs partialling, regime-aware")
    ap.add_argument("--y_column", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--apply_ewm_filter", action="store_true")
    ap.add_argument("--apply_ewm_filter_y", action="store_true")
    ap.add_argument("--exclude_mediators", action="store_true")
    ap.add_argument("--end_date", type=str, default=None,
                    help="Drop rows at or after this date (e.g. 2026-06-01)")
    ap.add_argument("--noise_quantile", type=float, default=None,
                    help="Instead of a date cut, keep rows whose local noise level "
                         "is below this quantile (objective regime selection)")
    ap.add_argument("--lags", type=str, default=None, help="Comma-separated lags in DAYS")
    ap.add_argument("--bandwidths", type=str, default=None,
                    help="Comma-separated partialling bandwidths in DAYS")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--learners", type=str, default="ridge")
    ap.add_argument("--include_contrast", action="store_true",
                    help="Also run the rejected block-contrast operator")
    args = ap.parse_args()

    lags_h = [d * 24.0 for d in
              ([float(x) for x in args.lags.split(",")] if args.lags else DEFAULT_LAGS_DAYS)]
    bws_h = [d * 24.0 for d in
             ([float(x) for x in args.bandwidths.split(",")] if args.bandwidths
              else DEFAULT_BW_DAYS)]
    learners = [s.strip() for s in args.learners.split(",") if s.strip()]

    start = datetime.now()
    out_dir = (Path("research_experiments") / "differencing"
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

    # -------------------------------------------------------------------------
    # Regime diagnosis
    # -------------------------------------------------------------------------
    print("\n=== Regime report ===")
    print("noise_std = std(y_raw - y), i.e. what the EWM removed. If it moves by")
    print("a factor of two the MEASUREMENT process is non-stationary, and a single")
    print("train/test split compares measurement regimes rather than models.")
    reg = F.regime_report(prep.y, prep.y_raw, prep.t_index)
    print(reg.to_string(float_format=lambda v: f"{v:.2f}"))
    reg.to_csv(out_dir / "regime_report.csv")

    # -------------------------------------------------------------------------
    # Regime restriction
    # -------------------------------------------------------------------------
    n_all = len(prep.X)
    if args.noise_quantile is not None:
        keep = F.stationary_mask(prep.y, prep.y_raw, prep.t_index,
                                 quantile=args.noise_quantile)
        how = f"noise below q{args.noise_quantile:g}"
    elif args.end_date:
        keep = np.asarray(prep.t_index < args.end_date)
        how = f"before {args.end_date}"
    else:
        keep = np.ones(n_all, dtype=bool)
        how = "full record"

    k = np.flatnonzero(keep)
    if len(k) < 500:
        raise RuntimeError(f"Regime restriction left only {len(k)} rows")
    X = prep.X.iloc[k]
    y = prep.y[k]
    t_index = prep.t_index[k]
    # Re-zero the clock so lags and bandwidths are relative to the kept window
    t_hours = D.to_hours(t_index)
    print(f"\nRegime restriction ({how}): {n_all} -> {len(k)} rows, "
          f"{t_index[0].date()} -> {t_index[-1].date()}, "
          f"{(t_hours[-1]/24):.0f} days")

    # -------------------------------------------------------------------------
    # Lag sweep (differencing) - no nuisance is fitted anywhere here
    # -------------------------------------------------------------------------
    print("\n=== Lag sweep: differencing (no nuisance estimated) ===")
    diff_results = {}
    for learner in learners:
        print(f"  learner={learner}")
        diff_results[learner] = F.lag_sweep(
            X, y, t_hours, lags_h, operator="diff",
            learner=learner, n_folds=args.n_folds, verbose=True,
        )
    diff_df = pd.concat(
        [F.sweep_frame(r).assign(learner=lr) for lr, r in diff_results.items()],
        ignore_index=True,
    )

    contrast_df = pd.DataFrame()
    if args.include_contrast:
        print("\n=== Lag sweep: block contrast (REJECTED operator, for the record) ===")
        cr = F.lag_sweep(X, y, t_hours, lags_h, operator="contrast",
                         learner=learners[0], n_folds=args.n_folds, verbose=True)
        contrast_df = F.sweep_frame(cr)

    # -------------------------------------------------------------------------
    # Bandwidth sweep (partialling) on the IDENTICAL rows
    # -------------------------------------------------------------------------
    print("\n=== Bandwidth sweep: partialling (same rows) ===")
    part_results = P.bandwidth_sweep(
        X, y, t_hours, bws_h,
        learner=learners[0], variant="A", n_folds=args.n_folds, verbose=True,
    )
    part_df = P.sweep_frame(part_results)

    # -------------------------------------------------------------------------
    # Agreement
    # -------------------------------------------------------------------------
    base_learner = learners[0]
    best_diff = max(diff_results[base_learner], key=lambda r: r.r2_mean)
    best_part = max(part_results, key=lambda r: r.r2_mean)
    print("\n" + "=" * 74)
    print("TIMESCALE AGREEMENT")
    print("=" * 74)
    print(f"  differencing peaks at lag       {best_diff.bandwidth_hours/24:7.2f} d "
          f"(R2={best_diff.r2_mean:+.4f} +-{best_diff.r2_std:.4f})")
    print(f"  partialling peaks at bandwidth  {best_part.bandwidth_hours/24:7.2f} d "
          f"(R2={best_part.r2_mean:+.4f} +-{best_part.r2_std:.4f})")
    print("  The two R2 are NOT comparable in level: differencing scores against")
    print("  dy and partialling against y~, which are different targets. Only the")
    print("  location of the peak is comparable, and that is the point.")

    # -------------------------------------------------------------------------
    # Coefficients at each optimum
    # -------------------------------------------------------------------------
    print("\n=== Coefficients at the differencing optimum ===")
    coef_diff = best_diff.coef_frame()
    if not coef_diff.empty:
        print(coef_diff.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        bad = coef_diff.loc[coef_diff["sign_flips"] > 0, "feature"].tolist()
        print(f"\n  identified: {len(coef_diff)-len(bad)}/{len(coef_diff)}")
        if bad:
            print(f"  NOT identified: {bad}")

    coef_part = best_part.coef_frame()
    merged = pd.DataFrame()
    if not coef_diff.empty and not coef_part.empty:
        merged = coef_diff[["feature", "coef_mean", "sign_flips"]].merge(
            coef_part[["feature", "coef_mean", "sign_flips"]],
            on="feature", suffixes=("_diff", "_part"),
        )
        both = merged[(merged["sign_flips_diff"] == 0) & (merged["sign_flips_part"] == 0)]
        print("\n=== Features identified by BOTH estimators ===")
        print("Agreement between two operators with different failure modes is the")
        print("strongest evidence available here that an effect is real.")
        if not both.empty:
            agree = both[np.sign(both["coef_mean_diff"]) == np.sign(both["coef_mean_part"])]
            print(both.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
            print(f"\n  identified by both: {len(both)}   same sign: {len(agree)}")
        else:
            print("  none")

    paths_df = F.coefficient_paths(diff_results[base_learner])

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for lr, res in diff_results.items():
        sf = F.sweep_frame(res)
        axes[0].errorbar(sf["lag_days"], sf["r2_mean"], yerr=sf["r2_std"],
                         marker="o", capsize=3, label=f"differencing ({lr})")
    axes[0].axvline(best_diff.bandwidth_hours / 24, ls="--", color="C3", alpha=0.7)
    axes[0].set_xscale("log"); axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("Lag (days, log)"); axes[0].set_ylabel("R2 on dy")
    axes[0].set_title("Differencing: no nuisance fitted")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].errorbar(part_df["bandwidth_days"], part_df["r2_mean"],
                     yerr=part_df["r2_std"], marker="s", capsize=3,
                     color="C1", label="partialling")
    axes[1].axvline(best_part.bandwidth_hours / 24, ls="--", color="C3", alpha=0.7)
    axes[1].set_xscale("log"); axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("Bandwidth (days, log)"); axes[1].set_ylabel("R2 on y~")
    axes[1].set_title("Partialling: nuisance fitted, same rows")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle(f"{args.y_column} - {how} - do the two operators agree on the timescale?")
    plt.tight_layout()
    plt.savefig(out_dir / "sweep_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    if not paths_df.empty and not coef_diff.empty:
        top = coef_diff.head(12)["feature"].tolist()
        fig, ax = plt.subplots(figsize=(11, 6))
        for f in top:
            sub = paths_df[paths_df["feature"] == f].sort_values("lag_days")
            ax.plot(sub["lag_days"], sub["coef_mean"], marker="o", ms=3, label=f)
        ax.axhline(0, color="gray", ls="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Lag (days, log)"); ax.set_ylabel("Coefficient (standardised)")
        ax.set_title(f"{args.y_column} - coefficient paths vs lag (flat = identified)")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "coefficient_paths_lag.png", dpi=150, bbox_inches="tight")
        plt.close()

    # noise profile over the whole record, with the kept window shaded
    prof = F.noise_profile(prep.y, prep.y_raw, prep.t_index)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(prof.index, prof.values, lw=1.0)
    ax.fill_between(t_index, ax.get_ylim()[0], ax.get_ylim()[1],
                    color="C2", alpha=0.12, label=f"kept ({how})")
    ax.set_ylabel("rolling std of removed noise")
    ax.set_title(f"{args.y_column} - measurement noise is not stationary")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "noise_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    diff_df.to_csv(out_dir / "lag_sweep.csv", index=False)
    part_df.to_csv(out_dir / "bandwidth_sweep.csv", index=False)
    if not contrast_df.empty:
        contrast_df.to_csv(out_dir / "contrast_sweep_rejected.csv", index=False)
    if not coef_diff.empty:
        coef_diff.to_csv(out_dir / "coefficients_differencing.csv", index=False)
    if not coef_part.empty:
        coef_part.to_csv(out_dir / "coefficients_partialling.csv", index=False)
    if not merged.empty:
        merged.to_csv(out_dir / "coefficient_agreement.csv", index=False)
    if not paths_df.empty:
        paths_df.to_csv(out_dir / "coefficient_paths_lag.csv", index=False)

    code_dir = out_dir / "code"
    code_dir.mkdir(exist_ok=True)
    import shutil
    for src in [Path(__file__).name, "differencing_research.py",
                "partialling_research.py", "data_prep_research.py"]:
        p = Path(__file__).parent / src
        if p.exists():
            shutil.copy2(p, code_dir / src)

    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "start_time": start.isoformat(),
            "end_time": datetime.now().isoformat(),
            "parameters": vars(args),
            "regime_restriction": how,
            "n_rows_all": int(n_all),
            "n_rows_kept": int(len(k)),
            "window": [str(t_index[0]), str(t_index[-1])],
            "regime_report": reg.reset_index().astype(str).to_dict(orient="records"),
            "best_differencing_lag_days": float(best_diff.bandwidth_hours / 24),
            "best_differencing_r2": best_diff.r2_mean,
            "best_partialling_bandwidth_days": float(best_part.bandwidth_hours / 24),
            "best_partialling_r2": best_part.r2_mean,
            "lag_sweep": diff_df.to_dict(orient="records"),
            "bandwidth_sweep": part_df.to_dict(orient="records"),
            "coefficients_differencing": (coef_diff.to_dict(orient="records")
                                          if not coef_diff.empty else []),
            "coefficient_agreement": (merged.to_dict(orient="records")
                                      if not merged.empty else []),
        }, f, indent=2, default=float)

    with open(out_dir / "artifacts.pkl", "wb") as f:
        cloudpickle.dump({
            "lag_sweep": diff_df, "bandwidth_sweep": part_df,
            "coefficients_differencing": coef_diff,
            "coefficients_partialling": coef_part,
            "agreement": merged, "regime_report": reg,
        }, f)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
