"""
Predicted vs actual, on the raw target scale, with out-of-fold predictions only.

Both estimators model a TRANSFORMED target, so a raw-scale plot needs an explicit
reconstruction, and the two reconstructions are not equally honest:

  DIFFERENCING (causal, no nuisance)
      y_hat(t) = y(t-k) + predicted delta
      Uses only the observed value k ago plus the change in X. Nothing is fitted
      to produce the anchor, and no future information is used. This is a genuine
      k-ahead prediction, and it comes with the right baseline for free:
      PERSISTENCE, y_hat(t) = y(t-k). If the model cannot beat persistence then
      the regression is contributing nothing.

  PARTIALLING (nowcast, optimistic)
      y_hat(t) = predicted y_tilde(t) + E[y|t]
      The trend added back is a two-sided smooth of y, so it uses neighbouring
      observations of the target. Legitimate as a nowcast, but it is NOT a
      forecast and its raw-scale R2 must not be read as explanatory power. It is
      shown for continuity with the earlier Ridge+Level plots, which had the same
      property.

All predictions are out-of-fold over contiguous time blocks.

Usage:
    python run_prediction_plots_research.py --y_column "Steam__kWh/T_" \
        --data_path ../data/costimier_turnup.parquet \
        --apply_ewm_filter --apply_ewm_filter_y --end_date 2026-06-01
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

import data_prep_research as D
import differencing_research as F
import partialling_research as P


def _r2_rmse_mae(a, b):
    return (float(r2_score(a, b)),
            float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))),
            float(mean_absolute_error(a, b)))


def break_gaps(t_index, series_list, max_gap_hours: float = 12.0):
    """
    Insert NaN across production gaps so matplotlib does not draw a straight line
    between points that are days apart. Without this the plot invents a smooth
    ramp across every shutdown, which reads as model behaviour but is an artefact.
    """
    t = pd.DatetimeIndex(t_index)
    if len(t) < 2:
        return t, series_list
    gap_after = np.flatnonzero(np.diff(t.asi8) / 3.6e12 > max_gap_hours)
    if len(gap_after) == 0:
        return t, series_list
    # A NaN placed just after each gap start severs the line segment
    ins_at = gap_after + 1
    ins_t = t[gap_after] + pd.Timedelta(seconds=1)
    t_new = pd.DatetimeIndex(np.insert(t.values, ins_at, ins_t.values))
    out = [np.insert(np.asarray(s, float), ins_at, np.nan) for s in series_list]
    return t_new, out


def main():
    ap = argparse.ArgumentParser(description="Predicted vs actual timeseries")
    ap.add_argument("--y_column", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--apply_ewm_filter", action="store_true")
    ap.add_argument("--apply_ewm_filter_y", action="store_true")
    ap.add_argument("--exclude_mediators", action="store_true")
    ap.add_argument("--end_date", type=str, default=None)
    ap.add_argument("--lag_days", type=float, default=1.0,
                    help="Differencing lag (sweep optimum on the clean regime: 1d)")
    ap.add_argument("--bandwidth_days", type=float, default=2.0,
                    help="Partialling bandwidth (sweep optimum on the clean regime: 2d)")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--zoom_days", type=float, default=30.0)
    args = ap.parse_args()

    start = datetime.now()
    out_dir = (Path("research_experiments") / "predictions"
               / args.y_column.replace("/", "_") / start.strftime("%y%m%d%H%M"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # -------------------------------------------------------------------------
    # Data, restricted to the requested regime
    # -------------------------------------------------------------------------
    prep = D.prepare(
        args.y_column, args.data_path,
        apply_ewm_filter=args.apply_ewm_filter,
        apply_ewm_filter_y=args.apply_ewm_filter_y,
        exclude_mediators=args.exclude_mediators,
    )
    keep = (np.asarray(prep.t_index < args.end_date) if args.end_date
            else np.ones(len(prep.X), dtype=bool))
    k = np.flatnonzero(keep)
    X = prep.X.iloc[k]
    y = prep.y[k]
    y_raw = prep.y_raw[k]
    t_index = prep.t_index[k]
    t_hours = D.to_hours(t_index)
    n = len(X)
    print(f"\nRows: {n}  {t_index[0].date()} -> {t_index[-1].date()}  "
          f"({t_hours[-1]/24:.0f} days)")

    folds = P.contiguous_blocks(n, args.n_folds)
    make_ridge = P.LEARNERS["ridge"]()
    summary = {}

    # -------------------------------------------------------------------------
    # 1. Differencing: causal k-ahead reconstruction
    # -------------------------------------------------------------------------
    lag_h = args.lag_days * 24.0
    d = F.difference(X, y, t_hours, lag_h)
    dpred = np.full(d.n, np.nan)

    for tr_rows, va_rows in folds:
        in_val = np.zeros(n, dtype=bool)
        in_val[va_rows] = True
        a_val, p_val = in_val[d.anchor_idx], in_val[d.partner_idx]
        val_sel, tr_sel = a_val & p_val, ~a_val & ~p_val
        if val_sel.sum() < 20 or tr_sel.sum() < 50:
            continue
        m = make_ridge()
        m.fit(d.dX[tr_sel], d.dy[tr_sel])
        dpred[val_sel] = m.predict(d.dX[val_sel])

    ok = ~np.isnan(dpred)
    anchor = d.anchor_idx[ok]
    partner = d.partner_idx[ok]
    t_anchor = t_index[anchor]

    # `y` here is the EWM-filtered target when --apply_ewm_filter_y is set. The
    # model is fitted and scored on THAT, so it must be labelled as such. The
    # smoothed series has less variance than the raw one, so predicting it is
    # easier and its R2 is not interchangeable with a raw-target R2.
    actual_ewm = y[anchor]
    persistence = y[partner]                 # y_hat = EWM(y)(t-k)
    model_ewm = y[partner] + dpred[ok]        # y_hat = EWM(y)(t-k) + delta_hat

    # The bill is paid on the unsmoothed target, so also score against raw y.
    # The anchor stays causal either way.
    actual_raw = y_raw[anchor]
    persistence_raw = y_raw[partner]
    model_raw = y[partner] + dpred[ok]

    r2_p, rmse_p, mae_p = _r2_rmse_mae(actual_ewm, persistence)
    r2_m, rmse_m, mae_m = _r2_rmse_mae(actual_ewm, model_ewm)
    r2_d, rmse_d, _ = _r2_rmse_mae(d.dy[ok], dpred[ok])
    r2_pr, rmse_pr, mae_pr = _r2_rmse_mae(actual_raw, persistence_raw)
    r2_mr, rmse_mr, mae_mr = _r2_rmse_mae(actual_raw, model_raw)

    ewm_note = "EWM-filtered" if args.apply_ewm_filter_y else "raw (no EWM on y)"
    print(f"\n=== Differencing, lag {args.lag_days}d, out-of-fold, causal ===")
    print(f"  target used for fitting: {ewm_note}")
    print(f"  on the differenced target dy      : R2={r2_d:+.4f}  rmse={rmse_d:7.3f}")
    print(f"  vs EWM(y) target, persistence     : R2={r2_p:+.4f}  rmse={rmse_p:7.3f}  mae={mae_p:6.3f}")
    print(f"  vs EWM(y) target, anchor + model  : R2={r2_m:+.4f}  rmse={rmse_m:7.3f}  mae={mae_m:6.3f}")
    print(f"      gain over persistence         : {r2_m - r2_p:+.4f} R2, "
          f"{100*(rmse_p-rmse_m)/rmse_p:+.1f}% rmse")
    print(f"  vs RAW y target, persistence      : R2={r2_pr:+.4f}  rmse={rmse_pr:7.3f}  mae={mae_pr:6.3f}")
    print(f"  vs RAW y target, anchor + model   : R2={r2_mr:+.4f}  rmse={rmse_mr:7.3f}  mae={mae_mr:6.3f}")
    print(f"      gain over persistence         : {r2_mr - r2_pr:+.4f} R2, "
          f"{100*(rmse_pr-rmse_mr)/rmse_pr:+.1f}% rmse")
    summary["differencing"] = {
        "lag_days": args.lag_days, "n_contrasts": int(ok.sum()),
        "target_for_fitting": ewm_note,
        "r2_dy": r2_d,
        "r2_ewm_persistence": r2_p, "r2_ewm_model": r2_m,
        "rmse_ewm_persistence": rmse_p, "rmse_ewm_model": rmse_m,
        "r2_gain_over_persistence_ewm": r2_m - r2_p,
        "r2_raw_persistence": r2_pr, "r2_raw_model": r2_mr,
        "rmse_raw_persistence": rmse_pr, "rmse_raw_model": rmse_mr,
        "r2_gain_over_persistence_raw": r2_mr - r2_pr,
    }

    # -------------------------------------------------------------------------
    # 2. Partialling: nowcast reconstruction
    # -------------------------------------------------------------------------
    bw_h = args.bandwidth_days * 24.0
    pdat = P.partial_out_time(X, y, t_hours, bw_h)
    ppred = np.full(n, np.nan)
    for tr_rows, va_rows in folds:
        m = make_ridge()
        m.fit(pdat.X_tilde[tr_rows], pdat.y_tilde[tr_rows])
        ppred[va_rows] = m.predict(pdat.X_tilde[va_rows])
    pok = ~np.isnan(ppred)

    r2_t, rmse_t, _ = _r2_rmse_mae(pdat.y_tilde[pok], ppred[pok])
    recon = ppred[pok] + pdat.y_hat_time[pok]
    r2_r, rmse_r, mae_r = _r2_rmse_mae(y[pok], recon)
    r2_trend, _, _ = _r2_rmse_mae(y[pok], pdat.y_hat_time[pok])

    print(f"\n=== Partialling, bandwidth {args.bandwidth_days}d, out-of-fold ===")
    print(f"  on the partialled target y~  : R2={r2_t:+.4f}  rmse={rmse_t:7.3f}")
    print(f"  raw scale, trend alone       : R2={r2_trend:+.4f}   <- the nowcast baseline")
    print(f"  raw scale, y~_hat + trend    : R2={r2_r:+.4f}  rmse={rmse_r:7.3f}  mae={mae_r:6.3f}")
    print(f"  NOTE: the trend is a two-sided smooth of y, so the raw-scale numbers")
    print(f"        are a nowcast and must not be read as explanatory power.")
    summary["partialling"] = {
        "bandwidth_days": args.bandwidth_days,
        "r2_y_tilde": r2_t, "r2_raw_trend_only": r2_trend,
        "r2_raw_reconstructed": r2_r, "rmse_raw_reconstructed": rmse_r,
    }

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    zoom_end = t_anchor[0] + pd.Timedelta(days=args.zoom_days)
    zoom = t_anchor < zoom_end

    tg, (ag, pg, mg, arg) = break_gaps(
        t_anchor, [actual_ewm, persistence, model_ewm, actual_raw])

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    ax = axes[0]
    ax.plot(tg, arg, lw=0.5, color="C7", alpha=0.45, label="actual, raw y", zorder=1)
    ax.plot(tg, ag, lw=0.8, color="k", label=f"actual, {ewm_note}", zorder=3)
    ax.plot(tg, pg, lw=0.6, color="C8", alpha=0.75,
            label=f"persistence  R2={r2_p:.3f} (raw {r2_pr:.3f})")
    ax.plot(tg, mg, lw=0.7, color="C0", alpha=0.85,
            label=f"anchor + model  R2={r2_m:.3f} (raw {r2_mr:.3f})")
    ax.set_ylabel(args.y_column)
    ax.set_title(f"{args.y_column} - DIFFERENCING, causal {args.lag_days:g}-day-ahead, "
                 f"out-of-fold (no nuisance fitted). Fitted on the {ewm_note} target.")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    tz, (az, pz, mz, arz) = break_gaps(
        t_anchor[zoom], [actual_ewm[zoom], persistence[zoom],
                         model_ewm[zoom], actual_raw[zoom]])
    ax.plot(tz, arz, lw=0.7, color="C7", alpha=0.5, label="actual, raw y")
    ax.plot(tz, az, lw=1.3, color="k", marker="o", ms=2.5,
            label=f"actual, {ewm_note}")
    ax.plot(tz, pz, lw=1.0, color="C8", alpha=0.8, label="persistence")
    ax.plot(tz, mz, lw=1.2, color="C0", label="model")
    ax.set_ylabel(args.y_column)
    ax.set_title(f"first {args.zoom_days:g} days, zoomed")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "differencing_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    tp, (yp, trp, rcp, ytp, yhp) = break_gaps(
        t_index[pok],
        [y[pok], pdat.y_hat_time[pok], recon, pdat.y_tilde[pok], ppred[pok]])

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    ax = axes[0]
    ax.plot(tp, yp, lw=0.7, color="k", label="actual", zorder=3)
    ax.plot(tp, trp, lw=1.6, color="C3", alpha=0.9,
            label=f"trend E[y|t], bw={args.bandwidth_days:g}d  R2={r2_trend:.3f}")
    ax.plot(tp, rcp, lw=0.7, color="C1", alpha=0.85,
            label=f"trend + model  R2={r2_r:.3f}")
    ax.set_ylabel(args.y_column)
    ax.set_title(f"{args.y_column} - PARTIALLING reconstruction, out-of-fold "
                 f"(NOWCAST: the trend uses neighbouring y)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(tp, ytp, lw=0.7, color="k", label="y~ actual")
    ax.plot(tp, yhp, lw=0.8, color="C0", alpha=0.85,
            label=f"y~ predicted (out-of-fold)  R2={r2_t:.3f}")
    ax.axhline(0, color="gray", ls="--", alpha=0.6)
    ax.set_ylabel("partialled " + args.y_column)
    ax.set_title("The quantity the model is actually responsible for")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "partialling_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (a, b, ttl) in zip(axes, [
        (actual_ewm, persistence, f"persistence vs {ewm_note}  R2={r2_p:.3f}"),
        (actual_ewm, model_ewm,
         f"model {args.lag_days:g}d-ahead vs {ewm_note}  R2={r2_m:.3f}"),
        (actual_raw, model_raw, f"model vs RAW y  R2={r2_mr:.3f}"),
    ]):
        ax.scatter(a, b, s=6, alpha=0.25)
        lim = [min(np.min(a), np.min(b)), max(np.max(a), np.max(b))]
        ax.plot(lim, lim, "r--", lw=1.3)
        ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.set_title(ttl)
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    pd.DataFrame({
        "timestamp": t_anchor,
        "actual_ewm": actual_ewm, "actual_raw": actual_raw,
        "persistence_ewm": persistence, "persistence_raw": persistence_raw,
        "model": model_ewm,
        "dy_actual": d.dy[ok], "dy_predicted": dpred[ok],
    }).to_csv(out_dir / "differencing_predictions.csv", index=False)
    pd.DataFrame({
        "timestamp": t_index[pok], "y": y[pok], "y_tilde": pdat.y_tilde[pok],
        "y_tilde_pred": ppred[pok], "trend": pdat.y_hat_time[pok],
        "reconstructed": recon,
    }).to_csv(out_dir / "partialling_predictions.csv", index=False)

    with open(out_dir / "results.json", "w") as f:
        json.dump({"parameters": vars(args), "n_rows": int(n),
                   "window": [str(t_index[0]), str(t_index[-1])],
                   "summary": summary}, f, indent=2, default=float)

    print(f"\nDone. Plots in: {out_dir}")


if __name__ == "__main__":
    main()
