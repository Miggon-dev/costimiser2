"""
Iterative backfitting with EXPLICIT control over where each choice is scored.

The original `state_estimation.iterative_backfit` makes two decisions using the
reported test block:

  1. the feature subset, because the caller passes cv_splits=[(train, test)] to
     CMA-ES, so 3000 candidate subsets are ranked by their test RMSE
  2. the best iteration, because it tracks `rmse_ridge` computed on `y_test`

Both inflate the reported number. This module reimplements the loop with the
index sets separated, so each decision can be pointed at a validation block or
(to reproduce the original) at the test block. That makes the leakage a dial
rather than a hidden default, which is what lets us attribute the reported
performance to its sources.

Nothing here modifies existing modules. `cmaes_feature_selection`, `_fit_ridge`
and `ResidualStateEstimator` are imported read-only.

MECHANICS (deliberately identical to the original, apart from the index sets)
    y_adjusted[fit] = y[fit] - gamma * level
    residuals are taken from the ORIGINAL y, never the adjusted one
    the level is refitted from scratch each iteration on those residuals

WHY BACKFITTING HELPS AT ALL
Subtracting the level from the *training* target removes the slow confounder, so
coefficients are estimated from less-confounded variation. That is the same
mechanism as time-partialling, which is why both have an interior optimum: the
iteration count and the smoother bandwidth are the same knob. Removing too
little leaves the confounder; removing too much lets the nuisance absorb real
signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from feature_selection import _fit_ridge, _DEFAULT_ALPHAS
from state_estimation import ResidualStateEstimator


# =============================================================================
# Results
# =============================================================================

@dataclass
class IterationRecord:
    iteration: int
    n_features: int
    selected_features: list[str]
    ridge_alpha: float
    # Scored on the block used to CHOOSE the iteration
    rmse_select: float
    r2_select: float
    # Scored on the untouched test block, recorded for diagnosis only and never
    # used to pick anything
    rmse_test: float
    r2_test: float
    level_scale: float
    obs_noise_scale: float
    is_best: bool = False


@dataclass
class BackfitResult:
    iterations: list[IterationRecord] = field(default_factory=list)
    best_iteration: int = -1
    selected_features: list[str] = field(default_factory=list)
    estimator: object = None
    level_fit: pd.Series | None = None
    ridge_alpha: float = float("nan")
    selection_scored_on: str = ""
    iteration_scored_on: str = ""

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "iteration": r.iteration,
            "n_features": r.n_features,
            "ridge_alpha": r.ridge_alpha,
            "rmse_select": r.rmse_select,
            "r2_select": r.r2_select,
            "rmse_test": r.rmse_test,
            "r2_test": r.r2_test,
            "level_scale": r.level_scale,
            "obs_noise_scale": r.obs_noise_scale,
            "is_best": r.is_best,
        } for r in self.iterations])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# =============================================================================
# The loop
# =============================================================================

def backfit(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    fit_idx: np.ndarray,
    select_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_selection_fn: Callable,
    n_iterations: int = 10,
    gamma: float = 1.0,
    patience: int = 2,
    state_estimator_kwargs: dict | None = None,
    verbose: bool = True,
) -> BackfitResult:
    """
    Parameters
    ----------
    fit_idx     rows used to fit the regression and the latent level
    select_idx  rows used to CHOOSE the best iteration. Point this at a
                validation block for an honest result, or at test_idx to
                reproduce the original behaviour.
    test_idx    untouched block, recorded per iteration for diagnosis only
    feature_selection_fn
                (X_full, y_adjusted_full, iteration) -> FeatureSelectionResult.
                Where SELECTION is scored is baked into this callable by the
                caller via its cv_splits, so both leaks are controlled
                independently.
    """
    y = np.asarray(y, dtype=float).ravel()
    state_estimator_kwargs = state_estimator_kwargs or {"level": True}

    result = BackfitResult()
    # Full-length target; only the fit rows get the level subtracted, exactly as
    # in the original implementation.
    y_adjusted_full = y.copy()

    best_rmse = np.inf
    no_improve = 0

    for it in range(n_iterations):
        fs = feature_selection_fn(X, y_adjusted_full, it)
        selected = list(fs.selected_features)
        est = fs.final_estimator

        Xs = X[selected].values
        pred_fit = np.asarray(est.predict(Xs[fit_idx])).ravel()
        pred_select = np.asarray(est.predict(Xs[select_idx])).ravel()
        pred_test = np.asarray(est.predict(Xs[test_idx])).ravel()

        # Residuals always come from the ORIGINAL target
        resid_fit = pd.Series(y[fit_idx] - pred_fit, index=X.index[fit_idx])

        state = ResidualStateEstimator(**state_estimator_kwargs)
        state_res = state.fit(resid_fit)
        level_fit = state_res.level

        # Adjust the target for the next iteration (fit rows only)
        y_adjusted_full = y.copy()
        y_adjusted_full[fit_idx] = y[fit_idx] - gamma * np.asarray(level_fit.values).ravel()

        m_sel = metrics(y[select_idx], pred_select)
        m_test = metrics(y[test_idx], pred_test)

        rec = IterationRecord(
            iteration=it + 1,
            n_features=len(selected),
            selected_features=selected,
            ridge_alpha=float(getattr(fs, "best_alpha", float("nan"))),
            rmse_select=m_sel["rmse"], r2_select=m_sel["r2"],
            rmse_test=m_test["rmse"], r2_test=m_test["r2"],
            level_scale=float(state_res.level_scale),
            obs_noise_scale=float(state_res.observation_noise_scale),
        )
        result.iterations.append(rec)

        if verbose:
            print(f"    it {it+1:2d}  k={len(selected):3d}  "
                  f"select: rmse={m_sel['rmse']:7.3f} r2={m_sel['r2']:+.4f}   "
                  f"(test: r2={m_test['r2']:+.4f})  "
                  f"level_scale={state_res.level_scale:6.3f}")

        if m_sel["rmse"] < best_rmse:
            best_rmse = m_sel["rmse"]
            result.best_iteration = it + 1
            result.selected_features = selected
            result.estimator = est
            result.level_fit = level_fit
            result.ridge_alpha = rec.ridge_alpha
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            if verbose:
                print(f"    early stop at iteration {it+1} "
                      f"(no improvement for {patience})")
            break

    for r in result.iterations:
        r.is_best = (r.iteration == result.best_iteration)

    return result


# =============================================================================
# Refit on a wider block
# =============================================================================

def refit(
    X: pd.DataFrame,
    y_target: np.ndarray,
    selected_features: list[str],
    refit_idx: np.ndarray,
    *,
    splines: bool = False,
    alphas: np.ndarray | None = None,
):
    """
    Refit the chosen feature subset on a wider set of rows.

    Needed to keep an honest variant comparable to a leaky one: if selection used
    a validation block, that block is still legitimate TRAINING data once the
    choice is made, and withholding it would confound "less leakage" with "less
    data".
    """
    alphas = _DEFAULT_ALPHAS if alphas is None else alphas
    return _fit_ridge(
        X[selected_features].values[refit_idx],
        np.asarray(y_target, dtype=float).ravel()[refit_idx],
        alphas,
        feature_names=list(selected_features),
        splines=splines,
    )


def level_adjusted_target(
    X: pd.DataFrame,
    y: np.ndarray,
    estimator,
    selected_features: list[str],
    idx: np.ndarray,
    *,
    gamma: float = 1.0,
    state_estimator_kwargs: dict | None = None,
) -> tuple[np.ndarray, pd.Series, object]:
    """
    One level-removal pass: fit the level on residuals over `idx`, return the
    adjusted target on those rows.

    Used to reproduce the training target the backfit converged to, so a refit on
    a wider block sees the same kind of target rather than the raw one.
    """
    state_estimator_kwargs = state_estimator_kwargs or {"level": True}
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(estimator.predict(X[selected_features].values[idx])).ravel()
    resid = pd.Series(y[idx] - pred, index=X.index[idx])
    state = ResidualStateEstimator(**state_estimator_kwargs)
    res = state.fit(resid)
    adjusted = y[idx] - gamma * np.asarray(res.level.values).ravel()
    return adjusted, res.level, state
