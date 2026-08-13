"""
Residual state estimation via Unobserved Components Model (UCM).

Decomposes regression residuals into:
- Local level (random walk drift)
- Optional periodic/seasonal components
- Observation noise

Uses statsmodels UnobservedComponents with exact MLE + Kalman smoother.
Fast (seconds), no TensorFlow dependency.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


# =============================================================================
# Result container
# =============================================================================

@dataclass
class StateEstimationResult:
    """Result of residual state decomposition."""
    level: pd.Series
    seasonal: pd.Series | None
    residual_noise: pd.Series
    smoothed_prediction: pd.Series
    model: UnobservedComponents
    results: object  # MLEResults

    @property
    def level_scale(self) -> float:
        """Estimated std of the level innovation."""
        return float(np.sqrt(self.results.params.iloc[self._level_var_idx]))

    @property
    def observation_noise_scale(self) -> float:
        """Estimated std of the observation noise."""
        return float(np.sqrt(self.results.params.iloc[0]))

    @property
    def _level_var_idx(self) -> int:
        # sigma2.irregular is param[0], sigma2.level is param[1]
        return 1


# =============================================================================
# Main class
# =============================================================================

class ResidualStateEstimator:
    """
    Estimate hidden state from regression residuals using UCM.

    Parameters
    ----------
    freq_seasonal : List of seasonal period specifications.
        Each entry is either an int (period) or a dict with keys:
        - "period": int (required)
        - "harmonics": int (optional, default=period//2)
        Example: [24, {"period": 168, "harmonics": 3}]
        Use 24 for daily cycles (hourly data), 168 for weekly.
        None = no seasonal component.
    level : Whether to include a local level (random walk). Default True.
    stochastic_level : Whether the level is stochastic. Default True.
    trend : Whether to include a local linear trend. Default False.
        If True, adds a slope component on top of the level.
    cycle : Whether to include a damped cycle. Default False.
    cycle_period_bounds : (min, max) period bounds for the cycle. Default (12, 200).
    maxiter : Maximum MLE iterations. Default 500.
    method : Optimization method for MLE. Default "lbfgs".

    Examples
    --------
    # Simple local level (drift only):
    est = ResidualStateEstimator()
    result = est.fit(residuals)

    # Local level + daily + weekly seasonality:
    est = ResidualStateEstimator(freq_seasonal=[24, {"period": 168, "harmonics": 3}])
    result = est.fit(residuals)

    # With damped cycle:
    est = ResidualStateEstimator(cycle=True, cycle_period_bounds=(20, 100))
    result = est.fit(residuals)
    """

    def __init__(
        self,
        *,
        freq_seasonal: list | None = None,
        level: bool = True,
        stochastic_level: bool = True,
        trend: bool = False,
        cycle: bool = False,
        cycle_period_bounds: tuple[float, float] = (12, 200),
        maxiter: int = 500,
        method: str = "lbfgs",
    ):
        self.freq_seasonal = freq_seasonal
        self.level = level
        self.stochastic_level = stochastic_level
        self.trend = trend
        self.cycle = cycle
        self.cycle_period_bounds = cycle_period_bounds
        self.maxiter = maxiter
        self.method = method

        # Fitted state
        self._model = None
        self._results = None

    def fit(self, residuals: pd.Series) -> StateEstimationResult:
        """
        Fit the UCM on residuals and return decomposed components.

        Parameters
        ----------
        residuals : pd.Series with DatetimeIndex.
            Typically y_actual - ridge_prediction.

        Returns
        -------
        StateEstimationResult with level, seasonal, noise components.
        """
        if not isinstance(residuals, pd.Series):
            raise TypeError("residuals must be a pandas Series.")
        if not isinstance(residuals.index, pd.DatetimeIndex):
            raise TypeError("residuals must have a DatetimeIndex.")
        if residuals.isna().any():
            residuals = residuals.interpolate(method="time").bfill().ffill()

        # Build seasonal spec for statsmodels
        seasonal_specs = self._build_seasonal_specs()

        # Build model (suppress statsmodels warnings about frequency/stochastic_level)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = UnobservedComponents(
                residuals,
                level=("local level" if self.level else False),
                trend=self.trend,
                cycle=self.cycle,
                freq_seasonal=seasonal_specs if seasonal_specs else None,
                **({
                    "damped_cycle": True,
                    "stochastic_cycle": True,
                } if self.cycle else {}),
            )

        # Fit via MLE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.fit(
                maxiter=self.maxiter,
                method=self.method,
                disp=False,
            )

        self._model = model
        self._results = results

        return self._build_result()

    def predict(
        self,
        steps: int,
        *,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Forecast the state forward (open-loop).

        Parameters
        ----------
        steps : number of steps ahead.
        alpha : confidence level for intervals.

        Returns
        -------
        DataFrame with columns: forecast, lower, upper.
        """
        if self._results is None:
            raise RuntimeError("Call fit() first.")

        forecast = self._results.get_forecast(steps=steps, alpha=alpha)
        return pd.DataFrame({
            "forecast": forecast.predicted_mean,
            "lower": forecast.conf_int().iloc[:, 0],
            "upper": forecast.conf_int().iloc[:, 1],
        })

    def update(
        self,
        new_residuals: pd.Series,
    ) -> StateEstimationResult:
        """
        Incrementally update the state with new observations.

        For irregular time series, re-fits the model on the full concatenated
        series (train + new) using the same parameters (no MLE re-optimization).
        This is equivalent to extending the Kalman filter forward.
        """
        if self._results is None:
            raise RuntimeError("Call fit() first.")

        if new_residuals.isna().any():
            new_residuals = new_residuals.interpolate(method="time").bfill().ffill()

        # Only keep observations strictly after the last fitted timestamp
        last_fitted_time = self._results.fittedvalues.index[-1]
        new_residuals = new_residuals[new_residuals.index > last_fitted_time]

        if len(new_residuals) == 0:
            return self._build_result()

        # Concatenate original + new data
        original_endog = pd.Series(
            self._results.data.endog.ravel(),
            index=self._results.fittedvalues.index,
        )
        full_endog = pd.concat([original_endog, new_residuals])

        # Build new model on full data, apply same parameters (no re-estimation)
        seasonal_specs = self._build_seasonal_specs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new_model = UnobservedComponents(
                full_endog,
                level=("local level" if self.level else False),
                trend=self.trend,
                cycle=self.cycle,
                freq_seasonal=seasonal_specs if seasonal_specs else None,
                **({
                    "damped_cycle": True,
                    "stochastic_cycle": True,
                } if self.cycle else {}),
            )

        # Apply fitted parameters to the new (longer) model — runs Kalman filter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._results = new_model.smooth(self._results.params)
        self._model = new_model

        return self._build_result()

    def get_state(self) -> dict:
        """Current state summary."""
        if self._results is None:
            raise RuntimeError("Call fit() first.")
        return {
            "current_level": float(self._results.filtered_state[-1, 0]),
            "n_observations": len(self._results.fittedvalues),
            "observation_noise_var": float(self._results.params.iloc[0]),
            "level_var": float(self._results.params.iloc[1]) if self.level else 0.0,
            "aic": float(self._results.aic),
            "bic": float(self._results.bic),
        }

    # --- Private helpers ---

    def _build_result(self) -> StateEstimationResult:
        """Extract components from current results into a StateEstimationResult."""
        full_index = self._results.fittedvalues.index
        level_component = self._extract_level(self._results, full_index)
        seasonal_component = self._extract_seasonal(self._results, full_index)
        smoothed = self._results.fittedvalues
        endog = self._results.data.endog.ravel()[:len(smoothed)]
        noise = pd.Series(endog - smoothed.values, index=full_index, name="noise")

        return StateEstimationResult(
            level=level_component,
            seasonal=seasonal_component,
            residual_noise=noise,
            smoothed_prediction=smoothed,
            model=self._model,
            results=self._results,
        )

    def _build_seasonal_specs(self) -> list[dict] | None:
        if not self.freq_seasonal:
            return None

        specs = []
        for item in self.freq_seasonal:
            if isinstance(item, int):
                specs.append({"period": item, "harmonics": item // 2})
            elif isinstance(item, dict):
                period = item["period"]
                harmonics = item.get("harmonics", period // 2)
                specs.append({"period": period, "harmonics": harmonics})
            else:
                raise ValueError(f"Invalid seasonal spec: {item}")
        return specs

    def _extract_level(self, results, index) -> pd.Series:
        if not self.level:
            return pd.Series(0.0, index=index, name="level")
        # Level is the first state in the state vector
        smoothed_state = results.smoothed_state
        return pd.Series(
            smoothed_state[0, :len(index)],
            index=index,
            name="level",
        )

    def _extract_seasonal(self, results, index) -> pd.Series | None:
        if not self.freq_seasonal:
            return None

        # Seasonal contribution = fitted - level - trend - cycle
        fitted = results.fittedvalues.values[:len(index)]
        level = self._extract_level(results, index).values

        # If trend exists, it's state[1]
        trend_vals = np.zeros(len(index))
        if self.trend:
            trend_vals = results.smoothed_state[1, :len(index)]

        seasonal = fitted - level - trend_vals
        return pd.Series(seasonal, index=index, name="seasonal")


# =============================================================================
# Iterative backfitting: regression + state decomposition
# =============================================================================

def iterative_backfit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_selection_fn,
    n_iterations: int = 10,
    patience: int = 2,
    state_estimator_kwargs: dict | None = None,
    splines: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Iteratively decompose target into regression + local level.

    Algorithm:
    ----------
    1. y_adjusted = y[train] (initially)
    2. For each iteration:
       a. Run feature selection + Ridge on (X, y_adjusted) with outer CV
       b. Compute residuals = y[train] - Ridge prediction on train
       c. Fit local level state estimator on residuals
       d. y_adjusted = y[train] - estimated_level (subtract drift)
    3. Final model: y = Ridge(X) + level + noise

    Parameters
    ----------
    X : full design matrix (train + test rows)
    y : full target (train + test rows, pd.Series with DatetimeIndex)
    train_idx : integer indices for training rows
    test_idx : integer indices for test rows
    feature_selection_fn : callable(X, y, iteration, splines) -> FeatureSelectionResult
        Receives the iteration number (0-based) and whether to use splines.
    n_iterations : maximum number of backfitting iterations
    patience : stop if best combined RMSE hasn't improved for this many iterations
    state_estimator_kwargs : dict of kwargs for ResidualStateEstimator
    splines : if True, use splines on iterations > 0 (first iteration is always linear)
    verbose : print progress

    Returns
    -------
    dict with:
        - "feature_selection_result": final FeatureSelectionResult
        - "selected_features": feature names used by the final estimator
        - "state_estimator": fitted ResidualStateEstimator
        - "state_result": final StateEstimationResult (full train+test)
        - "y_train_pred": prediction on train
        - "y_test_pred": prediction on test
        - "level_train": estimated level on train period
        - "level_test": estimated level on test period
        - "y_test_combined": prediction + level on test
        - "iteration_history": list of per-iteration metrics
    """
    if state_estimator_kwargs is None:
        state_estimator_kwargs = {}

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    history = []
    y_adjusted_full = y.copy()  # full-length target, modified on train portion
    level_train = pd.Series(0.0, index=y_train.index, name="level")

    # Early stopping state
    best_rmse_combined = np.inf
    best_iteration = -1
    best_state = None
    no_improve_count = 0

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'='*60}")

        # --- Step A: Feature selection on full X with adjusted y ---
        if verbose:
            print(f"\n  Step A: Feature selection on adjusted target...")

        fs_result = feature_selection_fn(
            X, np.array(y_adjusted_full).ravel(), iteration,
            splines=(splines and iteration > 0),
        )

        # Predict on train and test
        selected = fs_result.selected_features
        estimator = fs_result.final_estimator
        y_train_pred = estimator.predict(X_train[selected].values)
        y_test_pred = estimator.predict(X_test[selected].values)

        # --- Step B: Compute residuals from ORIGINAL train target ---
        residuals_train = y_train - y_train_pred

        # --- Step C: Fit state estimator on residuals ---
        if verbose:
            print(f"  Step C: Fitting state estimator on residuals...")

        state_est = ResidualStateEstimator(**state_estimator_kwargs)
        state_result_train = state_est.fit(residuals_train)
        level_train = state_result_train.level

        # --- Step D: Adjust target for next iteration (train portion only) ---
        y_adjusted_full = y.astype(np.float64).copy()
        y_adjusted_full.iloc[train_idx] = (y_train - level_train.values).astype(np.float64)

        # --- Evaluate on test ---
        residuals_test = y_test - y_test_pred
        state_result_full = state_est.update(residuals_test)
        n_test = len(y_test)
        level_test = state_result_full.level.iloc[-n_test:]
        y_test_combined = y_test_pred + level_test.values

        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score
        rmse_ridge = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        rmse_combined = float(np.sqrt(mean_squared_error(y_test, y_test_combined)))
        r2_ridge = float(r2_score(y_test, y_test_pred))
        r2_combined = float(r2_score(y_test, y_test_combined))

        iter_info = {
            "iteration": iteration + 1,
            "n_features": fs_result.n_features,
            "selected_features": selected,
            "ridge_alpha": fs_result.best_alpha,
            "rmse_ridge": rmse_ridge,
            "rmse_combined": rmse_combined,
            "r2_ridge": r2_ridge,
            "r2_combined": r2_combined,
            "level_scale": state_result_train.level_scale,
            "obs_noise_scale": state_result_train.observation_noise_scale,
        }
        history.append(iter_info)

        # Track best iteration
        if rmse_ridge < best_rmse_combined:
            best_rmse_combined = rmse_ridge
            best_iteration = iteration
            no_improve_count = 0
            best_state = {
                "feature_selection_result": fs_result,
                "selected_features": selected,
                "state_estimator": state_est,
                "state_result": state_result_full,
                "y_train_pred": pd.Series(y_train_pred, index=y_train.index),
                "y_test_pred": pd.Series(y_test_pred, index=y_test.index),
                "level_train": level_train,
                "level_test": level_test,
                "y_test_combined": pd.Series(y_test_combined, index=y_test.index),
            }
        else:
            no_improve_count += 1

        if verbose:
            print(f"\n  Results (iteration {iteration + 1}):")
            print(f"    Features: {len(selected)}")
            print(f"    Ridge RMSE: {rmse_ridge:.3f}, R2: {r2_ridge:.4f}")
            print(f"    Combined RMSE: {rmse_combined:.3f}, R2: {r2_combined:.4f}")
            print(f"    Level scale: {state_result_train.level_scale:.4f}")
            if iteration == best_iteration:
                print(f"    *** New best ***")

        # Early stopping
        if no_improve_count >= patience:
            if verbose:
                print(f"\n  Early stopping: no improvement for {patience} iterations.")
                print(f"  Best iteration: {best_iteration + 1} (RMSE combined: {best_rmse_combined:.3f})")
            break

    # Return best state
    best_state["iteration_history"] = history
    return best_state
