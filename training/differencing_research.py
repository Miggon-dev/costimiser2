r"""
Identify f in  y = f(X) + s(t) + eps  by DIFFERENCING, not by estimating s.

    y_t - y_{t-k}  =  (X_t - X_{t-k}) . beta  +  (s_t - s_{t-k})  +  noise
                                                 \_____________/
                                                  ~ 0 if s is smooth over k

Why this is worth running alongside the partialling approach:

  * NO NUISANCE IS ESTIMATED. Differencing is a fixed linear operator with zero
    fitted parameters, so it is immune to the whole family of leakage that comes
    from fitting a nuisance and then scoring against rows it has seen. The
    partialling estimator scored 0.211 with full-sample nuisance interpolation
    and 0.008 with train-only extrapolation; there is no such gap here.

  * WEAKER ASSUMPTION. It only requires s to be smooth over the lag, not to obey
    a global smoothness scale or a random-walk law.

  * INVARIANT TO LEVEL SHIFTS. This record contains at least three regimes
    (Nov-Feb noisy, Mar-Jun clean, Jul-Aug variance explosion) plus a ~40 unit
    level drop in June. Differencing removes any level shift outright. Variance
    changes still hurt, but a jumping level costs nothing.

THE LAG IS THE KNOB, AND IT MUST BE SWEPT
Differencing raw data at lag 1 is a pure high-pass and lands in the band that
grade-aware EWM already showed to be noise - which is why a fixed lag-1 attempt
fails. Applied to EWM-filtered data with the lag swept, the composition is a
BAND-PASS: EWM sets the upper frequency limit, the lag sets the lower one. The
lag here is the counterpart of the smoother bandwidth in partialling_research,
so if the two sweeps peak at the same timescale that is convergent evidence from
operators with almost nothing in common.

TWO OPERATORS
  "diff"      y_t - y_{t-k}                    RECOMMENDED
  "contrast"  mean(window at t) - mean(window at t-k)
                                               REJECTED - see the warning on
                                               block_contrast below

Validated on synthetic data with a known beta and a deliberate -40 level shift:
plain differencing recovers beta to within 0.04 at lags of 0.25-4 days where
naive OLS errs by 2.23, and it is unaffected by the level shift. It degrades at
very long lags (error 2.5 at 45 days) once the drift no longer cancels over the
lag, which is the expected upper limit rather than a defect.

Lags and window widths are specified in HOURS, never in rows, because the reel
spacing is irregular and gaps are common.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import partialling_research as P

# FoldResult / EvalResult are reused so the coefficient-stability reporting is
# identical across estimators. For results produced here, `bandwidth_hours`
# carries the LAG in hours and `variant` is "diff" or "contrast".
FoldResult = P.FoldResult
EvalResult = P.EvalResult


# =============================================================================
# Regime diagnosis
# =============================================================================

def regime_report(
    y: np.ndarray,
    y_raw: np.ndarray,
    t_index: pd.DatetimeIndex,
    freq: str = "MS",
) -> pd.DataFrame:
    """
    Per-period target statistics including the noise band.

    `noise_std` is std(y_raw - y), i.e. what the EWM removed. If it moves by a
    factor of two across the record then the measurement process itself is
    non-stationary, and a single train/test split will compare different
    measurement regimes rather than different models.
    """
    df = pd.DataFrame({"y": np.asarray(y, float),
                       "y_raw": np.asarray(y_raw, float)},
                      index=pd.DatetimeIndex(t_index))
    out = df.resample(freq).agg(n=("y", "size"), mean=("y", "mean"), std=("y", "std"))
    out["noise_std"] = (df["y_raw"] - df["y"]).resample(freq).std()
    return out


def noise_profile(
    y: np.ndarray,
    y_raw: np.ndarray,
    t_index: pd.DatetimeIndex,
    window: str = "14D",
) -> pd.Series:
    """Rolling std of the removed noise band, for objective regime masking."""
    resid = pd.Series(np.asarray(y_raw, float) - np.asarray(y, float),
                      index=pd.DatetimeIndex(t_index))
    return resid.rolling(window, center=True, min_periods=20).std()


def stationary_mask(
    y: np.ndarray,
    y_raw: np.ndarray,
    t_index: pd.DatetimeIndex,
    *,
    window: str = "14D",
    quantile: float = 0.75,
) -> np.ndarray:
    """
    Rows whose local noise level is below `quantile` of the record.

    An objective alternative to cutting by date: it selects the measurement
    regime rather than a hand-picked calendar window.
    """
    prof = noise_profile(y, y_raw, t_index, window=window)
    thr = float(prof.quantile(quantile))
    return np.asarray((prof <= thr).fillna(False))


# =============================================================================
# Time-based pairing
# =============================================================================

def lag_pairs(
    t_hours: np.ndarray,
    lag_hours: float,
    *,
    tolerance: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each row t, the index of the row closest to (t - lag_hours).

    Pairs whose realised separation deviates from `lag_hours` by more than
    `tolerance * lag_hours` are dropped, so production gaps never masquerade as a
    valid contrast.

    Returns (later_idx, earlier_idx).
    """
    t = np.asarray(t_hours, float)
    target = t - lag_hours
    j = np.searchsorted(t, target)
    j = np.clip(j, 0, len(t) - 1)
    # searchsorted may land one past the closest point
    j_alt = np.clip(j - 1, 0, len(t) - 1)
    better = np.abs(t[j_alt] - target) < np.abs(t[j] - target)
    j = np.where(better, j_alt, j)

    realised = t - t[j]
    ok = (j < np.arange(len(t))) & (np.abs(realised - lag_hours) <= tolerance * lag_hours)
    later = np.flatnonzero(ok)
    return later, j[later]


def _window_means(
    values: np.ndarray,
    t_hours: np.ndarray,
    centres_hours: np.ndarray,
    width_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean of `values` over (c - width, c] for each c, via cumulative sums.

    Returns (means, counts). Counts of zero mark unusable windows.
    """
    values = np.asarray(values, float)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    csum = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    lo = np.searchsorted(t_hours, centres_hours - width_hours, side="right")
    hi = np.searchsorted(t_hours, centres_hours, side="right")
    cnt = (hi - lo).astype(float)
    safe = np.maximum(cnt, 1.0)[:, None]
    means = (csum[hi] - csum[lo]) / safe
    if squeeze:
        means = means[:, 0]
    return means, cnt


# =============================================================================
# Operators
# =============================================================================

@dataclass
class DifferencedData:
    """Differenced target and design, plus where each contrast sits in time."""
    dy: np.ndarray
    dX: np.ndarray
    anchor_idx: np.ndarray      # row index of the LATER member of each contrast
    partner_idx: np.ndarray     # row index of the earlier member
    feature_names: list[str]
    lag_hours: float
    operator: str

    @property
    def n(self) -> int:
        return len(self.dy)


def difference(
    X: pd.DataFrame,
    y: np.ndarray,
    t_hours: np.ndarray,
    lag_hours: float,
    *,
    tolerance: float = 0.35,
) -> DifferencedData:
    """Plain differencing at a time-based lag."""
    later, earlier = lag_pairs(t_hours, lag_hours, tolerance=tolerance)
    if len(later) == 0:
        raise ValueError(f"no valid pairs at lag {lag_hours}h")
    Xv = X.values.astype(float)
    y = np.asarray(y, float).ravel()
    return DifferencedData(
        dy=y[later] - y[earlier],
        dX=Xv[later] - Xv[earlier],
        anchor_idx=later,
        partner_idx=earlier,
        feature_names=list(X.columns),
        lag_hours=lag_hours,
        operator="diff",
    )


def block_contrast(
    X: pd.DataFrame,
    y: np.ndarray,
    t_hours: np.ndarray,
    lag_hours: float,
    *,
    width_hours: float | None = None,
    min_count: int = 5,
) -> DifferencedData:
    """
    Difference of two window means separated by `lag_hours`.

    REJECTED - KEPT ONLY TO DOCUMENT THE FAILURE. Do not use for identification.

    The intent was to keep the level-shift invariance of plain differencing while
    averaging noise down by sqrt(window count) instead of doubling it. It does not
    work, and the reason is instructive: averaging X over a window removes the
    fast INDEPENDENT variation between features far faster than it removes the
    noise. What survives is each feature's slow component, so after averaging the
    columns become nearly collinear and the coefficients blow up into large
    offsetting pairs.

    Measured on synthetic data with true beta = [3, -2, 1]:

        lag      beta_hat                    max error
        1 d      [ 2.93, -2.15,  1.16]           0.16
        4 d      [ 2.56, -2.44,  1.44]           0.44
        14 d     [ 6.58, -3.74, -2.76]           3.76
        45 d     [38.44, -35.19, 46.78]         45.78

    Block-CV R2 degrades in step (-5.4 at 4 days, -85684 at 45 days). Noise
    reduction is already handled by the grade-aware EWM upstream; adding block
    averaging on top over-smooths and destroys the variation that identifies f.
    """
    if width_hours is None:
        width_hours = lag_hours / 3.0
    t = np.asarray(t_hours, float)
    y = np.asarray(y, float).ravel()
    Xv = X.values.astype(float)

    late_c = t
    early_c = t - lag_hours
    yA, cA = _window_means(y, t, late_c, width_hours)
    yB, cB = _window_means(y, t, early_c, width_hours)
    XA, _ = _window_means(Xv, t, late_c, width_hours)
    XB, _ = _window_means(Xv, t, early_c, width_hours)

    ok = (cA >= min_count) & (cB >= min_count) & (early_c >= t[0])
    anchor = np.flatnonzero(ok)
    if len(anchor) == 0:
        raise ValueError(f"no valid contrasts at lag {lag_hours}h")

    partner = np.searchsorted(t, early_c[anchor])
    partner = np.clip(partner, 0, len(t) - 1)

    return DifferencedData(
        dy=yA[anchor] - yB[anchor],
        dX=XA[anchor] - XB[anchor],
        anchor_idx=anchor,
        partner_idx=partner,
        feature_names=list(X.columns),
        lag_hours=lag_hours,
        operator="contrast",
    )


OPERATORS = {"diff": difference, "contrast": block_contrast}


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    t_hours: np.ndarray,
    *,
    lag_hours: float,
    operator: str = "diff",
    learner: str = "ridge",
    n_folds: int = 5,
    learner_kwargs: dict | None = None,
    tolerance: float = 0.35,
    width_hours: float | None = None,
) -> EvalResult:
    """
    Block-CV evaluation of the differenced regression.

    A contrast spans two timestamps, so a fold boundary can cut through one. Any
    contrast with one member inside the validation block and the other outside is
    DROPPED rather than assigned, otherwise validation rows would leak into
    training targets. Nothing else needs guarding, because no nuisance is fitted.

    R2 is measured on the differenced target, which changes with the lag - so the
    sweep answers "at which timescale is X informative about y", not "which model
    is better".
    """
    if operator not in OPERATORS:
        raise ValueError(f"operator must be one of {list(OPERATORS)}")
    kwargs = {} if operator == "diff" else {"width_hours": width_hours}
    if operator == "diff":
        kwargs = {"tolerance": tolerance}
    d = OPERATORS[operator](X, y, t_hours, lag_hours, **kwargs)

    learner_kwargs = learner_kwargs or {}
    make = P.LEARNERS[learner](**learner_kwargs)

    result = EvalResult(
        bandwidth_hours=lag_hours,          # reused field: LAG in hours
        learner=learner,
        variant=operator,
        feature_names=list(X.columns),
        nuisance_fit="none",                # nothing is fitted, by construction
    )

    n_rows = len(X)
    folds = P.contiguous_blocks(n_rows, n_folds)
    n_last = len(folds) - 1

    for k, (tr_rows, va_rows) in enumerate(folds):
        in_val = np.zeros(n_rows, dtype=bool)
        in_val[va_rows] = True

        a_val = in_val[d.anchor_idx]
        p_val = in_val[d.partner_idx]
        val_sel = a_val & p_val            # wholly inside the validation block
        tr_sel = ~a_val & ~p_val           # wholly outside it
        # contrasts straddling the boundary are discarded

        if val_sel.sum() < 20 or tr_sel.sum() < 50:
            continue

        model = make()
        model.fit(d.dX[tr_sel], d.dy[tr_sel])
        pred = np.asarray(model.predict(d.dX[val_sel])).ravel()

        est = model[-1] if hasattr(model, "__getitem__") else model
        raw = getattr(est, "coef_", None)
        coefs = np.asarray(raw).ravel().copy() if raw is not None else None

        result.folds.append(FoldResult(
            fold=k,
            r2_val=float(r2_score(d.dy[val_sel], pred)),
            rmse_val=float(np.sqrt(np.mean((d.dy[val_sel] - pred) ** 2))),
            n_train=int(tr_sel.sum()),
            n_val=int(val_sel.sum()),
            coefs=coefs,
            is_edge=(k == 0 or k == n_last),
        ))

    result.share_of_y_removed = float(
        1.0 - np.var(d.dy) / (2.0 * np.var(np.asarray(y, float)))
    )
    return result


def lag_sweep(
    X: pd.DataFrame,
    y: np.ndarray,
    t_hours: np.ndarray,
    lags_hours,
    *,
    operator: str = "diff",
    learner: str = "ridge",
    n_folds: int = 5,
    verbose: bool = True,
    **kwargs,
) -> list[EvalResult]:
    """Sweep the lag. Counterpart of partialling_research.bandwidth_sweep."""
    results = []
    for lag in lags_hours:
        try:
            res = evaluate(X, y, t_hours, lag_hours=lag, operator=operator,
                           learner=learner, n_folds=n_folds, **kwargs)
        except Exception as exc:
            if verbose:
                print(f"  lag={lag:8.1f}h  FAILED: {exc}")
            continue
        results.append(res)
        if verbose:
            n_used = sum(f.n_train + f.n_val for f in res.folds)
            print(f"  lag={lag:8.1f}h ({lag/24:6.2f}d)  "
                  f"R2(dy)={res.r2_mean:+.4f} +-{res.r2_std:.4f}  "
                  f"rmse={res.rmse_mean:8.3f}  contrasts~{n_used}")
    return results


def sweep_frame(results) -> pd.DataFrame:
    """Sweep results as a tidy table, with the lag named as such."""
    return pd.DataFrame([{
        "lag_hours": r.bandwidth_hours,
        "lag_days": r.bandwidth_hours / 24.0,
        "operator": r.variant,
        "learner": r.learner,
        "r2_mean": r.r2_mean,
        "r2_std": r.r2_std,
        "rmse_mean": r.rmse_mean,
        "n_folds_used": len(r.folds),
    } for r in results])


def coefficient_paths(results) -> pd.DataFrame:
    """Mean coefficient per feature at each lag (long format)."""
    rows = []
    for r in results:
        cf = r.coef_frame()
        if cf.empty:
            continue
        for _, row in cf.iterrows():
            rows.append({
                "lag_hours": r.bandwidth_hours,
                "lag_days": r.bandwidth_hours / 24.0,
                "operator": r.variant,
                "feature": row["feature"],
                "coef_mean": row["coef_mean"],
                "coef_std": row["coef_std"],
                "sign_flips": row["sign_flips"],
                "stability_ratio": row["stability_ratio"],
            })
    return pd.DataFrame(rows)
