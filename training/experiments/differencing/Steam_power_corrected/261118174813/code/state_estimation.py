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

    def one_step_ahead_level(self, new_series: pd.Series) -> pd.Series:
        """
        Strictly causal one-step-ahead level estimate for new observations.

        For each new point t, returns the predicted state E[s_t | data up to t-1],
        using the parameters estimated on the training fit. This uses ONLY past
        information (filtered/predicted state, no backward smoothing), so it
        reflects what would actually be known in production before observing y_t.

        Works with irregular indices by rebuilding the model on the concatenated
        (train + new) series with a clean integer index, applying the trained
        parameters (no refit), and reading the predicted (one-step-ahead) state.

        Parameters
        ----------
        new_series : pd.Series of new (test) observations, chronologically after
                     the training data.

        Returns
        -------
        pd.Series of one-step-ahead level predictions, aligned with new_series.
        """
        if self._results is None:
            raise RuntimeError("Call fit() first.")

        if new_series.isna().any():
            new_series = new_series.interpolate(method="time").bfill().ffill()

        # Concatenate original training endog + new data, use a clean RangeIndex
        original_endog = np.asarray(self._results.data.endog).ravel()
        new_endog = np.asarray(new_series.values, dtype=float).ravel()
        full_endog = np.concatenate([original_endog, new_endog])
        n_train = len(original_endog)

        seasonal_specs = self._build_seasonal_specs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full_model = UnobservedComponents(
                full_endog,
                level=("local level" if self.level else False),
                trend=self.trend,
                cycle=self.cycle,
                freq_seasonal=seasonal_specs if seasonal_specs else None,
                **({"damped_cycle": True, "stochastic_cycle": True} if self.cycle else {}),
            )
            # Apply trained parameters WITHOUT refitting; filter only (no smoothing)
            res = full_model.filter(self._results.params)

        # predicted_state[:, t] = E[state_t | observations 0..t-1] (one-step-ahead)
        # Take the test portion.
        predicted_state = res.predicted_state  # (n_states, n_obs + 1)
        level_pred = predicted_state[0, n_train:n_train + len(new_series)]

        return pd.Series(level_pred, index=new_series.index, name="level_one_step")

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
    gamma: float = 1.0,
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
        y_adjusted_full.iloc[train_idx] = (y_train - gamma * level_train.values).astype(np.float64)

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


# =============================================================================
# Orthogonal backfitting: ElasticNet + projected Local Level
# =============================================================================

def _orthogonal_projection(X_active: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Project s onto the orthogonal complement of the column space of X_active.

    Returns s_perp = (I - P_A) s where P_A = X_A (X_A'X_A)^+ X_A'.
    """
    if X_active.shape[1] == 0:
        return s.copy()
    # Use pseudoinverse for numerical stability (handles collinearity)
    pinv = np.linalg.pinv(X_active)  # shape: (n_features, n_samples)
    projection = X_active @ pinv @ s  # P_A @ s
    return s - projection


def iterative_backfit_orthogonal(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_fn=None,
    n_iterations: int = 10,
    patience: int = 2,
    tau: float = 0.0,
    state_estimator_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Elastic Net + Orthogonal Local-Level Backfitting.

    The level estimate is projected onto the orthogonal complement of
    the active regression features, ensuring no signal theft between
    the regression and state components.

    Algorithm:
    ----------
    β̂^(k) → r^(k) = y - Xβ̂^(k) → s̃^(k) = LocalLevel(r^(k))
    → s_perp^(k) = (I - P_A^(k)) s̃^(k) → z^(k+1) = y - s_perp^(k)
    → β̂^(k+1) = ElasticNet(X, z^(k+1))

    Parameters
    ----------
    X : full design matrix (train + test rows)
    y : full target (pd.Series with DatetimeIndex)
    train_idx : integer indices for training rows
    test_idx : integer indices for test rows
    model_fn : callable() -> unfitted sklearn estimator with .fit/.predict/.coef_
               Default: GridSearchCV(ElasticNet, TimeSeriesSplit)
    n_iterations : maximum iterations
    patience : early stopping patience (based on test RMSE of regression)
    tau : threshold for active set (|β_j| > tau). Use 0 for ElasticNet.
    state_estimator_kwargs : kwargs for ResidualStateEstimator
    verbose : print progress

    Returns
    -------
    dict with results from the best iteration.
    """
    if state_estimator_kwargs is None:
        state_estimator_kwargs = {}

    if model_fn is None:
        from sklearn.linear_model import ElasticNet
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        def model_fn():
            return GridSearchCV(
                ElasticNet(max_iter=10000),
                param_grid={
                    "alpha": np.logspace(-3, 2, 30),
                    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                },
                cv=TimeSeriesSplit(n_splits=5),
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
                refit=True,
            )

    y_full = np.asarray(y, dtype=np.float64).ravel()
    X_np = X.values.astype(np.float64)
    all_features = list(X.columns)

    y_train = y_full[train_idx]
    y_test = y_full[test_idx]
    X_train = X_np[train_idx]
    X_test = X_np[test_idx]

    train_index = y.index[train_idx]
    test_index = y.index[test_idx]

    history = []
    s_perp_train = np.zeros(len(train_idx))
    z_train = y_train.copy()  # initial adjusted target = y

    # Early stopping
    best_rmse = np.inf
    best_iteration = -1
    best_state = None
    no_improve_count = 0

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'='*60}")

        # --- Step 1: Fit ElasticNet on (X_train, z_train) ---
        if verbose:
            print(f"  Fitting ElasticNet...")
        est = model_fn()
        est.fit(X_train, z_train)

        # Get coefficients (handle GridSearchCV wrapper)
        if hasattr(est, 'best_estimator_'):
            coef = est.best_estimator_.coef_
        else:
            coef = est.coef_

        # Predictions
        y_train_pred = est.predict(X_train)
        y_test_pred = est.predict(X_test)

        # --- Step 2: Residuals from ORIGINAL target ---
        r_train = y_train - y_train_pred

        # --- Step 3: Fit LocalLevel on residuals ---
        if verbose:
            print(f"  Fitting LocalLevel on residuals...")
        residuals_series = pd.Series(r_train, index=train_index)
        state_est = ResidualStateEstimator(**state_estimator_kwargs)
        state_result = state_est.fit(residuals_series)
        s_tilde_train = state_result.level.values

        # --- Step 4: Active set and orthogonal projection ---
        active_mask = np.abs(coef) > tau
        n_active = active_mask.sum()
        X_active_train = X_train[:, active_mask]

        s_perp_train = _orthogonal_projection(X_active_train, s_tilde_train)

        if verbose:
            print(f"  Active features: {n_active}/{len(coef)}")
            print(f"  ||s_tilde||={np.std(s_tilde_train):.3f}, "
                  f"||s_perp||={np.std(s_perp_train):.3f}")

        # --- Step 5: Corrected target for next iteration ---
        z_train = y_train - s_perp_train

        # --- Evaluate on test ---
        # Project level onto test using the state estimator
        r_test = y_test - y_test_pred
        residuals_test_series = pd.Series(r_test, index=test_index)
        state_result_full = state_est.update(residuals_test_series)
        n_test = len(test_idx)
        s_tilde_test = state_result_full.level.iloc[-n_test:].values

        # Project test level orthogonally
        X_active_test = X_test[:, active_mask]
        s_perp_test = _orthogonal_projection(X_active_test, s_tilde_test)

        y_test_combined = y_test_pred + s_perp_test

        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score
        rmse_reg = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        rmse_combined = float(np.sqrt(mean_squared_error(y_test, y_test_combined)))
        r2_reg = float(r2_score(y_test, y_test_pred))
        r2_combined = float(r2_score(y_test, y_test_combined))

        iter_info = {
            "iteration": iteration + 1,
            "n_active": int(n_active),
            "active_features": [all_features[i] for i, m in enumerate(active_mask) if m],
            "rmse_regression": rmse_reg,
            "rmse_combined": rmse_combined,
            "r2_regression": r2_reg,
            "r2_combined": r2_combined,
            "level_std": float(np.std(s_tilde_train)),
            "level_perp_std": float(np.std(s_perp_train)),
        }
        history.append(iter_info)

        # Track best
        if rmse_reg < best_rmse:
            best_rmse = rmse_reg
            best_iteration = iteration
            no_improve_count = 0
            best_state = {
                "estimator": est,
                "selected_features": all_features,
                "active_features": iter_info["active_features"],
                "active_mask": active_mask.copy(),
                "state_estimator": state_est,
                "state_result": state_result_full,
                "s_perp_train": s_perp_train.copy(),
                "s_perp_test": s_perp_test.copy(),
                "y_train_pred": pd.Series(y_train_pred, index=train_index),
                "y_test_pred": pd.Series(y_test_pred, index=test_index),
                "y_test_combined": pd.Series(y_test_combined, index=test_index),
                "coef": coef.copy(),
            }
        else:
            no_improve_count += 1

        if verbose:
            print(f"\n  Results (iteration {iteration + 1}):")
            print(f"    Regression RMSE: {rmse_reg:.3f}, R2: {r2_reg:.4f}")
            print(f"    Combined RMSE:   {rmse_combined:.3f}, R2: {r2_combined:.4f}")
            if iteration == best_iteration:
                print(f"    *** New best ***")

        if no_improve_count >= patience:
            if verbose:
                print(f"\n  Early stopping at iteration {iteration + 1}.")
                print(f"  Best: iteration {best_iteration + 1} (RMSE={best_rmse:.3f})")
            break

    best_state["iteration_history"] = history
    return best_state


# =============================================================================
# Orthogonal backfitting with CMA-ES + Ridge
# =============================================================================

def _ridge_projection(X_active: np.ndarray, s: np.ndarray, alpha: float) -> np.ndarray:
    """
    Project s onto the orthogonal complement of the Ridge smoother space.

    P_lambda = X_A (X_A'X_A + lambda*I)^{-1} X_A'
    s_perp = (I - P_lambda) s
    """
    if X_active.shape[1] == 0:
        return s.copy()
    n_features = X_active.shape[1]
    XtX = X_active.T @ X_active
    XtX_reg = XtX + alpha * np.eye(n_features)
    # P_lambda @ s = X_A @ inv(X_A'X_A + lambda*I) @ X_A' @ s
    Xts = X_active.T @ s
    proj_coef = np.linalg.solve(XtX_reg, Xts)
    projection = X_active @ proj_coef
    return s - projection


def iterative_backfit_orthogonal_ridge(
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
    CMA-ES/Ridge + Orthogonal Local-Level Backfitting with Ridge projection.

    Uses P_lambda = X_A (X_A'X_A + lambda*I)^{-1} X_A' as the projection
    matrix, where lambda is the Ridge alpha from the fitted model.

    Algorithm:
    ----------
    β̂^(k) → r^(k) = y - Xβ̂^(k) → s̃^(k) = LocalLevel(r^(k))
    → s_perp^(k) = (I - P_λ^(k)) s̃^(k) → z^(k+1) = y - s_perp^(k)
    → β̂^(k+1) = Ridge/CMA-ES(X, z^(k+1))

    Parameters
    ----------
    X : full design matrix (train + test rows)
    y : full target (pd.Series with DatetimeIndex)
    train_idx : integer indices for training rows
    test_idx : integer indices for test rows
    feature_selection_fn : callable(X, y, iteration, splines) -> FeatureSelectionResult
    n_iterations : maximum iterations
    patience : early stopping patience
    state_estimator_kwargs : kwargs for ResidualStateEstimator
    splines : passed to feature_selection_fn for iterations > 0
    verbose : print progress

    Returns
    -------
    dict with results from the best iteration.
    """
    if state_estimator_kwargs is None:
        state_estimator_kwargs = {}

    y_full = np.asarray(y, dtype=np.float64).ravel()
    X_np = X.values.astype(np.float64)
    all_features = list(X.columns)

    y_train = y_full[train_idx]
    y_test = y_full[test_idx]
    X_train = X_np[train_idx]
    X_test = X_np[test_idx]

    train_index = y.index[train_idx]
    test_index = y.index[test_idx]

    history = []
    s_perp_train = np.zeros(len(train_idx))

    # Initial adjusted target = y (no level subtracted yet)
    y_adjusted_full = y_full.copy()

    # Early stopping
    best_rmse = np.inf
    best_iteration = -1
    best_state = None
    no_improve_count = 0

    for iteration in range(n_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'='*60}")

        # --- Step 1: Feature selection + Ridge ---
        if verbose:
            print(f"  Feature selection + Ridge...")

        fs_result = feature_selection_fn(
            X, y_adjusted_full, iteration,
            splines=(splines and iteration > 0),
        )

        selected = fs_result.selected_features
        selected_idx = [all_features.index(f) for f in selected]
        estimator = fs_result.final_estimator
        ridge_alpha = fs_result.best_alpha

        # Predictions
        y_train_pred = estimator.predict(X_train[:, selected_idx])
        y_test_pred = estimator.predict(X_test[:, selected_idx])

        # --- Step 2: Residuals from ORIGINAL target ---
        r_train = y_train - y_train_pred

        # --- Step 3: Fit LocalLevel on residuals ---
        if verbose:
            print(f"  Fitting LocalLevel on residuals...")
        residuals_series = pd.Series(r_train, index=train_index)
        state_est = ResidualStateEstimator(**state_estimator_kwargs)
        state_result = state_est.fit(residuals_series)
        s_tilde_train = state_result.level.values

        # --- Step 4: Ridge projection ---
        X_active_train = X_train[:, selected_idx]
        s_perp_train = _ridge_projection(X_active_train, s_tilde_train, ridge_alpha)

        if verbose:
            print(f"  Selected features: {len(selected)}, alpha: {ridge_alpha:.2f}")
            print(f"  ||s_tilde||={np.std(s_tilde_train):.3f}, "
                  f"||s_perp||={np.std(s_perp_train):.3f}")

        # --- Step 5: Corrected target for next iteration ---
        y_adjusted_full = y_full.copy()
        y_adjusted_full[train_idx] = y_train - s_perp_train

        # --- Evaluate on test ---
        r_test = y_test - y_test_pred
        residuals_test_series = pd.Series(r_test, index=test_index)
        state_result_full = state_est.update(residuals_test_series)
        n_test = len(test_idx)
        s_tilde_test = state_result_full.level.iloc[-n_test:].values

        X_active_test = X_test[:, selected_idx]
        s_perp_test = _ridge_projection(X_active_test, s_tilde_test, ridge_alpha)

        y_test_combined = y_test_pred + s_perp_test

        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score
        rmse_reg = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        rmse_combined = float(np.sqrt(mean_squared_error(y_test, y_test_combined)))
        r2_reg = float(r2_score(y_test, y_test_pred))
        r2_combined = float(r2_score(y_test, y_test_combined))

        iter_info = {
            "iteration": iteration + 1,
            "n_features": len(selected),
            "selected_features": selected,
            "ridge_alpha": ridge_alpha,
            "rmse_regression": rmse_reg,
            "rmse_combined": rmse_combined,
            "r2_regression": r2_reg,
            "r2_combined": r2_combined,
            "level_std": float(np.std(s_tilde_train)),
            "level_perp_std": float(np.std(s_perp_train)),
        }
        history.append(iter_info)

        # Track best
        if rmse_reg < best_rmse:
            best_rmse = rmse_reg
            best_iteration = iteration
            no_improve_count = 0
            best_state = {
                "feature_selection_result": fs_result,
                "selected_features": selected,
                "state_estimator": state_est,
                "state_result": state_result_full,
                "s_perp_train": s_perp_train.copy(),
                "s_perp_test": s_perp_test.copy(),
                "y_train_pred": pd.Series(y_train_pred, index=train_index),
                "y_test_pred": pd.Series(y_test_pred, index=test_index),
                "y_test_combined": pd.Series(y_test_combined, index=test_index),
                "ridge_alpha": ridge_alpha,
            }
        else:
            no_improve_count += 1

        if verbose:
            print(f"\n  Results (iteration {iteration + 1}):")
            print(f"    Regression RMSE: {rmse_reg:.3f}, R2: {r2_reg:.4f}")
            print(f"    Combined RMSE:   {rmse_combined:.3f}, R2: {r2_combined:.4f}")
            if iteration == best_iteration:
                print(f"    *** New best ***")

        if no_improve_count >= patience:
            if verbose:
                print(f"\n  Early stopping at iteration {iteration + 1}.")
                print(f"  Best: iteration {best_iteration + 1} (RMSE={best_rmse:.3f})")
            break

    best_state["iteration_history"] = history
    return best_state


# =============================================================================
# OOF Residuals + Local-Level Correction
# =============================================================================

def oof_level_correction(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_fn=None,
    n_splits: int = 5,
    min_train_size: int | None = None,
    state_estimator_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    OOF Residuals + Local-Level Correction.

    Procedure:
    1. Time-aware OOF predictions on training data
    2. OOF residuals -> fit LocalLevel -> estimate state
    3. De-state target: y* = y - s_hat
    4. Refit model on (X_train, y*)
    5. Final residuals -> final LocalLevel
    6. Predict: y_hat = f*(X) + s*

    Parameters
    ----------
    X_train : training features (DataFrame with DatetimeIndex)
    y_train : training target (Series with DatetimeIndex)
    X_test : test features
    y_test : test target
    model_fn : callable() -> unfitted sklearn-compatible model with fit/predict.
               Default: RealMLP_TD_Regressor(n_epochs=256)
    n_splits : number of sequential OOF blocks
    min_train_size : minimum training samples before first OOF block.
                     Default: len(X_train) // (n_splits + 1)
    state_estimator_kwargs : kwargs for ResidualStateEstimator
    verbose : print progress

    Returns
    -------
    dict with:
        - "final_model": refitted model on de-stated target
        - "state_estimator": final LocalLevel estimator
        - "y_train_oof": OOF predictions on train
        - "y_test_pred": structural prediction on test
        - "y_test_combined": structural + level on test
        - "level_train": estimated level on train
        - "level_test": estimated level on test
        - "metrics": dict with RMSE/R2 for regression and combined
    """
    if state_estimator_kwargs is None:
        state_estimator_kwargs = {"level": True}

    if model_fn is None:
        from pytabkit import RealMLP_TD_Regressor
        def model_fn():
            return RealMLP_TD_Regressor(n_epochs=256, random_state=42)

    n_train = len(X_train)
    if min_train_size is None:
        min_train_size = n_train // (n_splits + 1)

    # =========================================================================
    # Step 1-2: Time-aware OOF predictions
    # =========================================================================
    if verbose:
        print("Step 1: Computing time-aware OOF predictions...")

    # Create sequential blocks after min_train_size
    oof_start = min_train_size
    block_size = (n_train - oof_start) // n_splits
    oof_predictions = np.full(n_train, np.nan)

    for k in range(n_splits):
        block_start = oof_start + k * block_size
        block_end = oof_start + (k + 1) * block_size if k < n_splits - 1 else n_train

        # Train on everything before this block
        X_fit = X_train.iloc[:block_start].values
        y_fit = y_train.iloc[:block_start].values

        # Predict on block
        X_block = X_train.iloc[block_start:block_end].values

        model = model_fn()
        model.fit(X_fit, y_fit)
        preds = model.predict(X_block).ravel()
        oof_predictions[block_start:block_end] = preds

        if verbose:
            print(f"  Block {k+1}/{n_splits}: train[:{ block_start}] -> predict[{block_start}:{block_end}]")

    # Keep only rows with OOF predictions
    oof_mask = ~np.isnan(oof_predictions)
    oof_preds = oof_predictions[oof_mask]
    oof_index = y_train.index[oof_mask]
    y_oof = y_train.values[oof_mask]

    # =========================================================================
    # Step 3: OOF residuals
    # =========================================================================
    r_oof = y_oof - oof_preds

    if verbose:
        print(f"\nStep 3: OOF residuals computed ({len(r_oof)} samples)")

    # =========================================================================
    # Step 4: Local-level estimation on OOF residuals
    # =========================================================================
    if verbose:
        print("Step 4: Fitting LocalLevel on OOF residuals...")

    r_oof_series = pd.Series(r_oof, index=oof_index)
    state_est_oof = ResidualStateEstimator(**state_estimator_kwargs)
    state_result_oof = state_est_oof.fit(r_oof_series)
    s_hat_oof = state_result_oof.level.values

    if verbose:
        print(f"  Level scale: {state_result_oof.level_scale:.4f}")
        print(f"  Obs noise scale: {state_result_oof.observation_noise_scale:.4f}")

    # =========================================================================
    # Step 5: De-stated target (only for OOF-covered rows)
    # =========================================================================
    # Extend level to full training set (backfill for early rows)
    level_train_full = np.zeros(n_train)
    level_train_full[oof_mask] = s_hat_oof
    # For rows before OOF starts, use the first level value
    first_level = s_hat_oof[0] if len(s_hat_oof) > 0 else 0.0
    level_train_full[:oof_start] = first_level

    y_star = y_train.values - level_train_full

    if verbose:
        print(f"\nStep 5: De-stated target computed (y* = y - s_hat)")

    # =========================================================================
    # Step 6: Structural refit on de-stated target
    # =========================================================================
    if verbose:
        print("Step 6: Refitting model on de-stated target...")

    final_model = model_fn()
    final_model.fit(X_train.values, y_star)

    y_train_struct = final_model.predict(X_train.values).ravel()
    y_test_struct = final_model.predict(X_test.values).ravel()

    # =========================================================================
    # Step 7: Final residual state
    # =========================================================================
    if verbose:
        print("Step 7: Fitting final LocalLevel on structural residuals...")

    r_final_train = y_train.values - y_train_struct
    r_final_series = pd.Series(r_final_train, index=y_train.index)

    state_est_final = ResidualStateEstimator(**state_estimator_kwargs)
    state_result_final = state_est_final.fit(r_final_series)
    level_train_final = state_result_final.level.values

    # Extend to test
    r_final_test = y_test.values - y_test_struct
    r_final_test_series = pd.Series(r_final_test, index=y_test.index)
    state_result_full = state_est_final.update(r_final_test_series)
    n_test = len(y_test)
    level_test_final = state_result_full.level.iloc[-n_test:].values

    if verbose:
        print(f"  Final level scale: {state_result_final.level_scale:.4f}")

    # =========================================================================
    # Step 8: Final prediction
    # =========================================================================
    y_test_combined = y_test_struct + level_test_final

    # Metrics
    from sklearn.metrics import mean_squared_error, r2_score
    rmse_struct = float(np.sqrt(mean_squared_error(y_test.values, y_test_struct)))
    rmse_combined = float(np.sqrt(mean_squared_error(y_test.values, y_test_combined)))
    r2_struct = float(r2_score(y_test.values, y_test_struct))
    r2_combined = float(r2_score(y_test.values, y_test_combined))

    if verbose:
        print(f"\nResults:")
        print(f"  Structural only:  RMSE={rmse_struct:.2f}, R2={r2_struct:.4f}")
        print(f"  Struct + Level:   RMSE={rmse_combined:.2f}, R2={r2_combined:.4f}")

    return {
        "final_model": final_model,
        "state_estimator": state_est_final,
        "state_result": state_result_full,
        "y_train_oof": pd.Series(oof_predictions, index=y_train.index),
        "y_train_struct": pd.Series(y_train_struct, index=y_train.index),
        "y_test_pred": pd.Series(y_test_struct, index=y_test.index),
        "y_test_combined": pd.Series(y_test_combined, index=y_test.index),
        "level_train": pd.Series(level_train_final, index=y_train.index),
        "level_test": pd.Series(level_test_final, index=y_test.index),
        "metrics": {
            "rmse_structural": rmse_struct,
            "rmse_combined": rmse_combined,
            "r2_structural": r2_struct,
            "r2_combined": r2_combined,
        },
    }
