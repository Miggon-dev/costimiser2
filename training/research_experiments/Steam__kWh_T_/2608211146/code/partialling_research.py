"""
Time-partialling (Robinson / Double-ML style) estimation of f in

    y = f(X) + s(t) + eps

where s(t) is an unobserved, slowly-varying machine state (felt condition,
fouling, seasonal ambient load, ...).

WHY THIS EXISTS
---------------
Estimating f and s *jointly* does not identify them: a random-walk level has
close to one degree of freedom per observation, and the split between f and s is
decided by a variance ratio estimated from the same likelihood that fits f. The
level then absorbs any covariate variation that is slow in time. Backfitting
does not fix this either - it only limits the absorption by stopping early,
which is an accidental regulariser with no principled stopping rule.

The fix is to remove the nuisance from BOTH sides with the SAME fixed linear
operator, before fitting anything:

    y_tilde = y - E[y | t]
    X_tilde = X - E[X | t]
    fit  y_tilde = f(X_tilde)

For linear f this is exact: a linear filter L commutes with a linear model, so
L(y) = L(X)b + L(s) + L(eps) and b is preserved while L(s) ~ 0. No absorption is
possible, because the trend has already been projected out of the covariates
too. The smoother bandwidth is a hyperparameter chosen by validation, not
something MLE picks to maximise in-sample fit - that is what breaks the
circularity.

TWO IMPORTANT CONSEQUENCES
--------------------------
1. Estimating f is an OFFLINE identification task, not forecasting, so
   two-sided (non-causal) smoothing over history is legitimate and much better
   conditioned than a one-sided filter. Causality only matters if you later want
   to nowcast s - and for setpoint optimisation you do not, because s is
   additive and cancels when comparing setpoints at the same moment:
       argmax_X [f(X) + s] == argmax_X f(X)
2. Because the nuisance is gone before fitting, ANY learner can be used for f.
   No backfitting, no iteration, no convergence question.

Frequency view: your grade-aware EWM removes the noise band; time-partialling
removes the drift band. Together they form a band-pass that isolates the
mid-band where the actionable signal lives. Differencing (delta_y = g(delta_X))
fails because it is a pure high-pass - it amplifies exactly the band the EWM
already showed to be noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Two-sided Gaussian smoother on irregular timestamps
# =============================================================================

def _to_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """Timestamps -> float hours since the first observation."""
    t = pd.DatetimeIndex(index).asi8.astype(np.float64)  # nanoseconds
    return (t - t[0]) / 3.6e12


def gaussian_time_smooth(
    t_hours: np.ndarray,
    values: np.ndarray,
    bandwidth_hours: float,
    *,
    fit_mask: np.ndarray | None = None,
    truncate: float = 3.0,
) -> np.ndarray:
    """
    Two-sided Gaussian kernel estimate of E[values | t], evaluated at every
    t_hours, using only rows where `fit_mask` is True as support.

    Irregular spacing is handled naturally (weights depend on elapsed time, not
    on row position). The kernel is truncated at `truncate` bandwidths so cost
    and memory stay linear-ish instead of O(n^2).

    Parameters
    ----------
    t_hours : (n,) float, must be non-decreasing
    values  : (n,) or (n, p)
    bandwidth_hours : kernel sigma in hours. FIXED by the caller - this is the
        hyperparameter that decides where "slow nuisance" ends and "signal"
        begins, and it must be chosen by validation, not by MLE on the fit.
    fit_mask : rows usable as smoother support. Rows outside the mask still get
        a fitted value (that is the point - it lets a validation block be
        smoothed from training rows only, with no leakage).

    Returns
    -------
    (n,) or (n, p) fitted conditional means.
    """
    values = np.asarray(values, dtype=np.float64)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    n = len(t_hours)
    if values.shape[0] != n:
        raise ValueError(f"values has {values.shape[0]} rows, expected {n}")
    if bandwidth_hours <= 0:
        raise ValueError("bandwidth_hours must be > 0")

    if fit_mask is None:
        fit_mask = np.ones(n, dtype=bool)
    fit_idx = np.flatnonzero(fit_mask)
    if len(fit_idx) == 0:
        raise ValueError("fit_mask selects no rows")

    t_fit = t_hours[fit_idx]
    v_fit = values[fit_idx]
    order = np.argsort(t_fit, kind="stable")
    t_fit, v_fit = t_fit[order], v_fit[order]

    half = truncate * bandwidth_hours
    lo = np.searchsorted(t_fit, t_hours - half, side="left")
    hi = np.searchsorted(t_fit, t_hours + half, side="right")

    out = np.empty_like(values)
    inv2s2 = 1.0 / (2.0 * bandwidth_hours ** 2)
    for i in range(n):
        a, b = lo[i], hi[i]
        if b <= a:
            # No support inside the truncation window: fall back to the nearest
            # available point rather than producing a NaN.
            j = min(max(np.searchsorted(t_fit, t_hours[i]), 0), len(t_fit) - 1)
            out[i] = v_fit[j]
            continue
        d = t_fit[a:b] - t_hours[i]
        w = np.exp(-(d * d) * inv2s2)
        wsum = w.sum()
        if wsum <= 0:
            j = min(max(np.searchsorted(t_fit, t_hours[i]), 0), len(t_fit) - 1)
            out[i] = v_fit[j]
        else:
            out[i] = (w @ v_fit[a:b]) / wsum

    return out[:, 0] if squeeze else out


def smoother_effective_dof(
    t_hours: np.ndarray, bandwidth_hours: float, truncate: float = 3.0
) -> float:
    """
    trace(S) for the Gaussian smoothing matrix - how many degrees of freedom the
    nuisance estimate actually consumes.

    This is the number to sanity-check against a fitted random-walk level. A
    level with per-step innovation std comparable to the residual std has
    effectively O(n) dof and will track the target rather than the drift; a
    smoother with dof in the tens can only represent genuine slow structure.
    """
    n = len(t_hours)
    half = truncate * bandwidth_hours
    lo = np.searchsorted(t_hours, t_hours - half, side="left")
    hi = np.searchsorted(t_hours, t_hours + half, side="right")
    inv2s2 = 1.0 / (2.0 * bandwidth_hours ** 2)
    trace = 0.0
    for i in range(n):
        a, b = lo[i], hi[i]
        if b <= a:
            trace += 1.0
            continue
        d = t_hours[a:b] - t_hours[i]
        w = np.exp(-(d * d) * inv2s2)
        s = w.sum()
        if s > 0:
            trace += 1.0 / s  # self-weight is exp(0)=1, normalised by sum
    return float(trace)


# =============================================================================
# Partialling out
# =============================================================================

@dataclass
class PartialledData:
    """Result of removing E[. | t] from both sides."""
    y_tilde: np.ndarray
    X_tilde: np.ndarray
    y_hat_time: np.ndarray
    X_hat_time: np.ndarray
    feature_names: list[str]
    bandwidth_hours: float

    @property
    def share_of_y_removed(self) -> float:
        """Fraction of y variance attributed to the time trend."""
        tot = np.var(self.y_tilde) + np.var(self.y_hat_time)
        return float(np.var(self.y_hat_time) / tot) if tot > 0 else 0.0


def partial_out_time(
    X: pd.DataFrame,
    y: np.ndarray,
    t_hours: np.ndarray,
    bandwidth_hours: float,
    *,
    fit_mask: np.ndarray | None = None,
) -> PartialledData:
    """
    Remove the time-explainable component from y and from every column of X,
    using the same smoother for both (this is what preserves the coefficients).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    Xv = X.values.astype(np.float64)

    y_hat = gaussian_time_smooth(t_hours, y, bandwidth_hours, fit_mask=fit_mask)
    X_hat = gaussian_time_smooth(t_hours, Xv, bandwidth_hours, fit_mask=fit_mask)

    return PartialledData(
        y_tilde=y - y_hat,
        X_tilde=Xv - X_hat,
        y_hat_time=y_hat,
        X_hat_time=X_hat,
        feature_names=list(X.columns),
        bandwidth_hours=bandwidth_hours,
    )


# =============================================================================
# Contiguous block cross-validation
# =============================================================================

def contiguous_blocks(n: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    (train_idx, val_idx) per fold, where each validation block is a contiguous
    stretch of time and training is everything else.

    Contiguous (not shuffled) because the nuisance is a time trend: shuffled
    folds would let neighbouring rows leak the local trend value into the
    held-out rows. Interior folds are surrounded by training data on both sides,
    so the nuisance is interpolated; the first and last folds extrapolate, which
    is the realistic worst case and is averaged over.
    """
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        val = np.arange(edges[k], edges[k + 1])
        train = np.setdiff1d(np.arange(n), val, assume_unique=True)
        if len(val) and len(train):
            folds.append((train, val))
    return folds


# =============================================================================
# Learner factories
# =============================================================================

def ridge_factory(alphas: Sequence[float] | None = None) -> Callable:
    """Standardise then RidgeCV. The interpretable / inference workhorse."""
    if alphas is None:
        alphas = np.logspace(0, 3, 20)

    def _make():
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=list(alphas))),
        ])
    return _make


def hist_gbr_factory(**kwargs) -> Callable:
    """Gradient boosting - flexible baseline, always available in sklearn."""
    params = dict(max_iter=300, learning_rate=0.05, max_depth=None,
                  min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    params.update(kwargs)

    def _make():
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(**params)
    return _make


def realmlp_factory(**kwargs) -> Callable:
    """
    RealMLP (pytabkit). Only usable because the nuisance is removed BEFORE
    fitting - previously a flexible f was competing with a flexible level for
    the same variance, which is why it lost to Ridge.
    """
    params = dict(device="cpu", random_state=0, verbosity=0)
    params.update(kwargs)

    def _make():
        from pytabkit import RealMLP_TD_S_Regressor
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", RealMLP_TD_S_Regressor(**params)),
        ])
    return _make


LEARNERS: dict[str, Callable[[], Callable]] = {
    "ridge": ridge_factory,
    "hist_gbr": hist_gbr_factory,
    "realmlp": realmlp_factory,
}


# =============================================================================
# Variant B: small fixed time basis
# =============================================================================

def fixed_time_basis(t_hours: np.ndarray, n_columns: int = 5) -> np.ndarray:
    """
    A deliberately tiny smooth basis in time (Gaussian bumps at evenly spaced
    centres, normalised).

    Used by Variant B, where a flexible learner is given f(X, B(t)). The point
    is that B(t) is too coarse to represent mid-band covariate signal, so the
    learner can only reach slow drift through it. This bounds absorption by
    construction instead of hoping a penalty will do it.
    """
    if n_columns < 1:
        return np.empty((len(t_hours), 0))
    span = t_hours[-1] - t_hours[0]
    if span <= 0:
        return np.zeros((len(t_hours), n_columns))
    centres = np.linspace(t_hours[0], t_hours[-1], n_columns)
    width = span / max(n_columns - 1, 1)
    d = t_hours[:, None] - centres[None, :]
    B = np.exp(-0.5 * (d / width) ** 2)
    return B / np.clip(B.sum(axis=1, keepdims=True), 1e-12, None)


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class FoldResult:
    fold: int
    r2_val: float
    rmse_val: float
    n_train: int
    n_val: int
    coefs: np.ndarray | None = None
    # First/last blocks have training data on one side only, so a fold-fitted
    # nuisance must EXTRAPOLATE across them instead of interpolating. Their
    # y_tilde therefore still contains drift and their R2 is not meaningful.
    is_edge: bool = False


@dataclass
class EvalResult:
    """Block-CV evaluation of one (bandwidth, learner, variant) combination."""
    bandwidth_hours: float
    learner: str
    variant: str
    folds: list[FoldResult] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    share_of_y_removed: float = 0.0
    smoother_dof: float = float("nan")
    nuisance_fit: str = "full"

    def _scoring_folds(self) -> list[FoldResult]:
        """Folds whose R2 is interpretable given how the nuisance was fitted."""
        if self.nuisance_fit == "fold":
            interior = [f for f in self.folds if not f.is_edge]
            if interior:
                return interior
        return self.folds

    @property
    def r2_mean(self) -> float:
        f = self._scoring_folds()
        return float(np.mean([x.r2_val for x in f])) if f else float("nan")

    @property
    def r2_std(self) -> float:
        f = self._scoring_folds()
        return float(np.std([x.r2_val for x in f])) if f else float("nan")

    @property
    def r2_mean_all_folds(self) -> float:
        return float(np.mean([x.r2_val for x in self.folds])) if self.folds else float("nan")

    @property
    def rmse_mean(self) -> float:
        f = self._scoring_folds()
        return float(np.mean([x.rmse_val for x in f])) if f else float("nan")

    def coef_frame(self) -> pd.DataFrame:
        """Per-feature coefficients across folds, with a stability verdict."""
        rows = [f.coefs for f in self.folds if f.coefs is not None]
        if not rows or not self.feature_names:
            return pd.DataFrame()
        C = np.vstack(rows)
        mean, std = C.mean(axis=0), C.std(axis=0)
        flips = (np.sign(C) != np.sign(mean)[None, :]).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(std > 0, np.abs(mean) / std, np.inf)
        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "coef_mean": mean,
                "coef_std": std,
                "coef_min": C.min(axis=0),
                "coef_max": C.max(axis=0),
                "sign_flips": flips,
                "n_folds": len(rows),
                "stability_ratio": ratio,
            })
            .sort_values("stability_ratio", ascending=False)
            .reset_index(drop=True)
        )


def evaluate(
    X: pd.DataFrame,
    y: np.ndarray,
    t_hours: np.ndarray,
    *,
    bandwidth_hours: float,
    learner: str = "ridge",
    variant: str = "A",
    n_folds: int = 5,
    time_basis_columns: int = 5,
    learner_kwargs: dict | None = None,
    nuisance_fit: str = "full",
) -> EvalResult:
    """
    Block-CV evaluation with per-fold, training-only partialling.

    Variant "A": partial time out of y and X, fit learner on the residualised
        data. Exact for linear f, and the coefficients are the deliverable for
        setpoint optimisation.
    Variant "B": leave y and X alone, hand the learner a small fixed time basis
        alongside X. Permits a flexible f while bounding how much slow structure
        it can absorb. Scored on the same partialled target as A so the two are
        directly comparable.

    nuisance_fit : "full" or "fold"
        "full" (default): E[.|t] is estimated once on all rows. The nuisance is
            a FIXED low-dof smoother (see `smoother_dof`), not something tuned to
            maximise fit, so this is the standard Robinson formulation and it
            gives every fold a consistently defined target. Use for comparing
            bandwidths and learners.
        "fold": E[.|t] is re-estimated from each fold's training rows only, so no
            target information crosses the fold boundary at all.
            USE THIS FOR COEFFICIENTS, NOT FOR R2. A validation block spans
            n/n_folds rows; if that is wide relative to the bandwidth the
            smoother has little or no support inside the hole, so y_tilde there
            still contains drift and R2 collapses even when the coefficients are
            exactly right. Its value is as an audit: if the coefficients match
            those from "full" mode, full-sample partialling is not leaking in any
            way that affects the deliverable. Edge folds are excluded from the
            reported score since they can only be extrapolated into.

    The reported R2 is against y_tilde - the component f is actually responsible
    for. R2 against raw y is not comparable and is inflated by any nuisance that
    has seen the target. Note that y_tilde itself CHANGES with bandwidth, so R2
    across the sweep answers "at which timescale is X most informative about y",
    not "which model is better".
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    learner_kwargs = learner_kwargs or {}
    if learner not in LEARNERS:
        raise ValueError(f"unknown learner '{learner}', expected one of {list(LEARNERS)}")
    make = LEARNERS[learner](**learner_kwargs)

    if nuisance_fit not in ("full", "fold"):
        raise ValueError(f"nuisance_fit must be 'full' or 'fold', got {nuisance_fit!r}")

    result = EvalResult(
        bandwidth_hours=bandwidth_hours,
        learner=learner,
        variant=variant,
        feature_names=list(X.columns),
        smoother_dof=smoother_effective_dof(t_hours, bandwidth_hours),
        nuisance_fit=nuisance_fit,
    )

    full = partial_out_time(X, y, t_hours, bandwidth_hours)
    result.share_of_y_removed = full.share_of_y_removed

    folds = contiguous_blocks(len(y), n_folds)
    n_last = len(folds) - 1

    for k, (tr, va) in enumerate(folds):
        if nuisance_fit == "full":
            pd_fold = full
        else:
            # Nuisance support = this fold's training rows only, so no target
            # information crosses the fold boundary.
            mask = np.zeros(len(y), dtype=bool)
            mask[tr] = True
            pd_fold = partial_out_time(X, y, t_hours, bandwidth_hours, fit_mask=mask)

        if variant.upper() == "A":
            Xtr, Xva = pd_fold.X_tilde[tr], pd_fold.X_tilde[va]
        elif variant.upper() == "B":
            B = fixed_time_basis(t_hours, time_basis_columns)
            Xraw = X.values.astype(np.float64)
            Xtr = np.hstack([Xraw[tr], B[tr]])
            Xva = np.hstack([Xraw[va], B[va]])
        else:
            raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")

        ytr, yva = pd_fold.y_tilde[tr], pd_fold.y_tilde[va]

        model = make()
        model.fit(Xtr, ytr)
        pred = np.asarray(model.predict(Xva)).ravel()

        coefs = None
        if variant.upper() == "A":
            est = model[-1] if hasattr(model, "__getitem__") else model
            raw = getattr(est, "coef_", None)
            if raw is not None:
                # The learner pipeline already standardises its input, so coef_
                # is ALREADY in standardised-feature units and directly
                # comparable across folds and features. Do not rescale.
                coefs = np.asarray(raw).ravel().copy()

        result.folds.append(FoldResult(
            fold=k,
            r2_val=float(r2_score(yva, pred)),
            rmse_val=float(np.sqrt(np.mean((yva - pred) ** 2))),
            n_train=len(tr),
            n_val=len(va),
            coefs=coefs,
            is_edge=(k == 0 or k == n_last),
        ))

    return result


def bandwidth_sweep(
    X: pd.DataFrame,
    y: np.ndarray,
    t_hours: np.ndarray,
    bandwidths_hours: Sequence[float],
    *,
    learner: str = "ridge",
    variant: str = "A",
    n_folds: int = 5,
    nuisance_fit: str = "full",
    verbose: bool = True,
) -> list[EvalResult]:
    """
    Sweep the smoother bandwidth.

    This is the most informative diagnostic available for this problem. It
    replaces guessing where f ends and s begins: coefficients that hold steady
    across bandwidths are real effects, coefficients that swing are trend
    contamination. It also shows the timescale separation empirically instead of
    by assumption.
    """
    results = []
    for bw in bandwidths_hours:
        res = evaluate(
            X, y, t_hours, bandwidth_hours=bw,
            learner=learner, variant=variant, n_folds=n_folds,
            nuisance_fit=nuisance_fit,
        )
        results.append(res)
        if verbose:
            print(f"  bw={bw:8.1f}h ({bw/24:6.2f}d)  "
                  f"R2(y~)={res.r2_mean:+.4f} +-{res.r2_std:.4f}  "
                  f"rmse={res.rmse_mean:7.3f}  "
                  f"y_var_removed={res.share_of_y_removed:5.1%}  "
                  f"dof={res.smoother_dof:6.1f}")
    return results


def sweep_frame(results: Sequence[EvalResult]) -> pd.DataFrame:
    """Sweep results as a tidy table."""
    return pd.DataFrame([{
        "bandwidth_hours": r.bandwidth_hours,
        "bandwidth_days": r.bandwidth_hours / 24.0,
        "learner": r.learner,
        "variant": r.variant,
        "r2_mean": r.r2_mean,
        "r2_std": r.r2_std,
        "rmse_mean": r.rmse_mean,
        "share_of_y_removed": r.share_of_y_removed,
        "smoother_dof": r.smoother_dof,
    } for r in results])


def coefficient_paths(results: Sequence[EvalResult]) -> pd.DataFrame:
    """
    Mean coefficient per feature at each bandwidth (long format).

    A feature whose path is flat across bandwidths is identified from mid-band
    variation and is safe to optimise over. A feature whose path drifts or
    changes sign as the bandwidth widens was borrowing from the trend.
    """
    rows = []
    for r in results:
        cf = r.coef_frame()
        if cf.empty:
            continue
        for _, row in cf.iterrows():
            rows.append({
                "bandwidth_hours": r.bandwidth_hours,
                "bandwidth_days": r.bandwidth_hours / 24.0,
                "feature": row["feature"],
                "coef_mean": row["coef_mean"],
                "coef_std": row["coef_std"],
                "sign_flips": row["sign_flips"],
                "stability_ratio": row["stability_ratio"],
            })
    return pd.DataFrame(rows)
