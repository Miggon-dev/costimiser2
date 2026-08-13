"""
Bayesian Structural Time Series module using TensorFlow Probability.

Provides:
- model_builder: construct LocalLevel + regression STS models
- fit_sts_variational_posterior: variational inference fitting
- analyse_sts_predictions: forecast analysis and decomposition
- IncrementalTFPRegressor: stateful incremental Kalman filter wrapper
- plot_sts_analysis: diagnostic plotting
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tfd = tfp.distributions
sts = tfp.sts


# =============================================================================
# Preprocessing helpers
# =============================================================================

class AddIntercept(BaseEstimator, TransformerMixin):
    """Add a constant column to a pandas DataFrame."""

    def __init__(self, name: str = "intercept", position: int = 0):
        self.name = name
        self.position = position

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AddIntercept expects a pandas DataFrame.")
        if self.name in X.columns:
            raise ValueError(f"Column {self.name!r} already exists.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AddIntercept expects a pandas DataFrame.")
        result = X.copy()
        result.insert(self.position, self.name, np.float32(1.0))
        return result

    def get_feature_names_out(self, input_features=None):
        features = (
            list(self.feature_names_in_)
            if input_features is None else list(input_features)
        )
        features.insert(self.position, self.name)
        return np.asarray(features, dtype=object)


@dataclass(frozen=True)
class CenteredTarget:
    """Training-target centering state."""
    mean: float

    @classmethod
    def fit(cls, y) -> "CenteredTarget":
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        return cls(mean=float(values.mean()))

    def transform(self, y) -> np.ndarray:
        return np.asarray(y, dtype=np.float64).reshape(-1) - self.mean

    def inverse_transform(self, y_centered) -> np.ndarray:
        return np.asarray(y_centered, dtype=np.float64) + self.mean


def prepare_centered_targets(y_train, y_test=None):
    """Fit centering on y_train, return (centerer, y_train_c, y_test_c)."""
    centerer = CenteredTarget.fit(y_train)
    y_train_c = centerer.transform(y_train)
    y_test_c = None if y_test is None else centerer.transform(y_test)
    return centerer, y_train_c, y_test_c


# =============================================================================
# Model construction
# =============================================================================

def model_has_local_level(model) -> bool:
    return any(
        isinstance(c, sts.LocalLevel)
        for c in getattr(model, "components", [])
    )


def find_regression_component(model):
    """Find the single regression component inside the STS model."""
    candidates = [
        c for c in getattr(model, "components", [])
        if type(c).__name__ in {
            "LinearRegression", "SparseLinearRegression",
            "DynamicLinearRegression",
        }
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected 1 regression component, found {len(candidates)}"
        )
    return candidates[0]


def model_builder(
    design_matrix, *,
    include_local_level: bool = True,
    level_scale_prior=None,
    initial_level_prior=None,
    regression_type: str = "linear",
    weights_prior=None,
    sparse_weights_prior_scale: float = 0.1,
    dynamic_drift_scale_prior=None,
    dynamic_initial_weights_prior=None,
    model_name: str = "model",
):
    """Build a TFP STS model: optional LocalLevel + regression."""
    design_matrix = tf.convert_to_tensor(design_matrix, dtype=tf.float32)
    n_features = int(design_matrix.shape[-1])

    # Regression component
    if regression_type == "linear":
        if weights_prior is None:
            raise ValueError("weights_prior required for linear regression.")
        reg = sts.LinearRegression(
            design_matrix=design_matrix,
            weights_prior=weights_prior, name="regression",
        )
    elif regression_type == "sparse":
        reg = sts.SparseLinearRegression(
            design_matrix=design_matrix,
            weights_prior_scale=tf.constant(
                sparse_weights_prior_scale, dtype=tf.float32
            ),
            name="regression",
        )
    elif regression_type == "dynamic":
        if dynamic_initial_weights_prior is None:
            dynamic_initial_weights_prior = tfd.MultivariateNormalDiag(
                loc=tf.zeros(n_features, dtype=tf.float32),
                scale_diag=tf.ones(n_features, dtype=tf.float32),
            )
        if dynamic_drift_scale_prior is None:
            dynamic_drift_scale_prior = tfd.LogNormal(
                loc=tf.constant(np.log(0.01), dtype=tf.float32),
                scale=tf.constant(1.0, dtype=tf.float32),
            )
        reg = sts.DynamicLinearRegression(
            design_matrix=design_matrix,
            drift_scale_prior=dynamic_drift_scale_prior,
            initial_weights_prior=dynamic_initial_weights_prior,
            name="dynamic_regression",
        )
    else:
        raise ValueError("regression_type must be 'linear', 'sparse', or 'dynamic'.")

    components = [reg]
    if include_local_level:
        if level_scale_prior is None or initial_level_prior is None:
            raise ValueError("Level priors required when include_local_level=True.")
        components.insert(0, sts.LocalLevel(
            level_scale_prior=level_scale_prior,
            initial_level_prior=initial_level_prior,
            name="local_level",
        ))

    return sts.Sum(components=components, name=model_name)


# =============================================================================
# Variational fitting
# =============================================================================

def validate_parameter_sample_alignment(*, model, parameter_samples):
    if not isinstance(parameter_samples, Mapping):
        if len(parameter_samples) != len(model.parameters):
            raise RuntimeError("Posterior samples not aligned with model.")
        return True
    model_names = {str(p.name) for p in model.parameters}
    sample_names = {str(n) for n in parameter_samples.keys()}
    if model_names != sample_names:
        raise RuntimeError("STS model and posterior samples misaligned.")
    return True


def fit_sts_variational_posterior(
    *, model, y_train_tf, target_mean: float = 0.0,
    num_variational_steps: int = 2000, learning_rate: float = 0.05,
    num_posterior_samples: int = 500, seed: int = 42,
    optimizer=None, sample_size: int = 1, verbose: bool = True,
):
    """Fit a variational posterior for a TFP STS model."""
    y_train_tf = tf.convert_to_tensor(y_train_tf, dtype=tf.float32)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    surrogate_posterior = sts.build_factored_surrogate_posterior(
        model=model, seed=seed,
    )
    target_log_prob_fn = model.joint_distribution(
        observed_time_series=y_train_tf
    ).log_prob

    if optimizer is None:
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    loss_curve = tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        optimizer=optimizer,
        num_steps=num_variational_steps,
        sample_size=sample_size, seed=seed,
    )
    loss_curve_np = np.asarray(loss_curve, dtype=float)

    parameter_samples = surrogate_posterior.sample(
        num_posterior_samples, seed=seed + 1,
    )
    validate_parameter_sample_alignment(
        model=model, parameter_samples=parameter_samples,
    )

    if verbose:
        print(f"LocalLevel: {model_has_local_level(model)}")
        print(f"VI steps: {num_variational_steps}, samples: {num_posterior_samples}")
        print(f"Loss: {loss_curve_np[0]:.1f} -> {loss_curve_np[-1]:.1f} "
              f"(min={np.min(loss_curve_np):.1f})")

    return {
        "model": model,
        "surrogate_posterior": surrogate_posterior,
        "parameter_samples": parameter_samples,
        "loss_curve": loss_curve,
        "loss_curve_np": loss_curve_np,
        "optimizer": optimizer,
        "target_log_prob_fn": target_log_prob_fn,
        "target_mean": float(target_mean),
        "include_local_level": model_has_local_level(model),
    }


# =============================================================================
# Posterior parameter extraction
# =============================================================================

def parameter_samples_to_dict(*, model, parameter_samples):
    if isinstance(parameter_samples, Mapping):
        return {str(n): v for n, v in parameter_samples.items()}
    return {
        str(p.name): s
        for p, s in zip(model.parameters, parameter_samples)
    }


def find_parameter_samples(*, samples_by_name, required_fragments,
                           excluded_fragments=None):
    required = [str(x).lower() for x in required_fragments]
    excluded = [str(x).lower() for x in (excluded_fragments or [])]
    candidates = [
        (str(n), v) for n, v in samples_by_name.items()
        if all(f in str(n).lower() for f in required)
        and not any(f in str(n).lower() for f in excluded)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely identify {required}. Found: "
            f"{[n for n, _ in candidates]}"
        )
    return candidates[0]


def extract_effective_regression_weights(*, model, parameter_samples, n_features):
    samples_by_name = parameter_samples_to_dict(
        model=model, parameter_samples=parameter_samples,
    )
    validate_parameter_sample_alignment(
        model=model, parameter_samples=parameter_samples,
    )
    component = find_regression_component(model)
    ctype = type(component).__name__

    if ctype == "SparseLinearRegression":
        lookup = {
            "global_scale_variance": ["global_scale_variance"],
            "global_scale_noncentered": ["global_scale_noncentered"],
            "local_scale_variances": ["local_scale_variances"],
            "local_scales_noncentered": ["local_scales_noncentered"],
            "weights_noncentered": ["weights_noncentered"],
        }
        sparse_args = {}
        for arg, frags in lookup.items():
            _, vals = find_parameter_samples(
                samples_by_name=samples_by_name, required_fragments=frags,
            )
            sparse_args[arg] = tf.convert_to_tensor(vals, dtype=tf.float32)

        beta = np.asarray(
            component.params_to_weights(**sparse_args), dtype=float,
        ).reshape(-1, n_features)
        return {"regression_type": "sparse", "beta_samples": beta,
                "weight_parameter_name": "effective_sparse_weights",
                "regression_component": component}

    if ctype == "LinearRegression":
        candidates = [
            (str(n), np.asarray(v, dtype=float))
            for n, v in samples_by_name.items()
            if str(n).lower().endswith("/_weights")
            and "noncentered" not in str(n).lower()
            and np.asarray(v, dtype=float).shape[-1] == n_features
        ]
        if len(candidates) != 1:
            raise RuntimeError("Cannot identify linear regression weights.")
        name, beta = candidates[0]
        return {"regression_type": "linear",
                "beta_samples": beta.reshape(-1, n_features),
                "weight_parameter_name": name,
                "regression_component": component}

    raise TypeError(f"Unsupported regression: {ctype}")


def extract_static_model_parameters(*, model, parameter_samples,
                                    n_transformed_features):
    """Extract beta, level_scale, observation_scale from fitted model."""
    samples_by_name = parameter_samples_to_dict(
        model=model, parameter_samples=parameter_samples,
    )
    weight_result = extract_effective_regression_weights(
        model=model, parameter_samples=parameter_samples,
        n_features=n_transformed_features,
    )
    _, obs_samples = find_parameter_samples(
        samples_by_name=samples_by_name,
        required_fragments=["observation", "noise", "scale"],
    )
    beta = np.asarray(weight_result["beta_samples"], dtype=float
                      ).reshape(-1, n_transformed_features)
    n_samples = beta.shape[0]

    def align(values, name):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if len(arr) == n_samples:
            return arr
        if len(arr) == 1:
            return np.full(n_samples, arr[0])
        raise RuntimeError(f"{name}: got {len(arr)} samples, need {n_samples}")

    result = {
        "beta_samples": beta,
        "observation_scale_samples": align(obs_samples, "obs_scale"),
        "regression_type": weight_result["regression_type"],
        "weight_parameter_name": weight_result["weight_parameter_name"],
        "regression_component": weight_result["regression_component"],
        "include_local_level": model_has_local_level(model),
    }

    if result["include_local_level"]:
        _, lvl_samples = find_parameter_samples(
            samples_by_name=samples_by_name,
            required_fragments=["level", "scale"],
            excluded_fragments=["observation"],
        )
        result["level_scale_samples"] = align(lvl_samples, "level_scale")
    else:
        result["level_scale_samples"] = np.zeros(n_samples)

    return result


# Backward-compatible alias
extract_local_level_parameters = extract_static_model_parameters


# =============================================================================
# Metrics
# =============================================================================

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


# =============================================================================
# Incremental wrapper
# =============================================================================

@dataclass
class IncrementalTFPState:
    predicted_state_mean: np.ndarray
    predicted_state_variance: np.ndarray
    timestamp: Any = None
    n_updates: int = 0

    def copy(self) -> "IncrementalTFPState":
        return IncrementalTFPState(
            predicted_state_mean=self.predicted_state_mean.copy(),
            predicted_state_variance=self.predicted_state_variance.copy(),
            timestamp=self.timestamp,
            n_updates=self.n_updates,
        )


class IncrementalTFPRegressor:
    """
    Stateful wrapper for fitted TFP STS models (with or without LocalLevel).

    - predict(X): read-only, does not change state.
    - update(X, y): assimilates observations via Kalman filter.
    - reset(): restores end-of-training state.
    """

    def __init__(
        self, *, fit_result=None, trained_model=None, parameter_samples=None,
        preprocessor=None, X_train=None, X_train_raw=None, y_train=None,
        target_mean: float | None = None,
        raw_feature_names: Sequence[str] | None = None,
        transformed_feature_names: Sequence[str] | None = None,
        initial_level_prior=None,
        model_name: str = "TFP static regression",
    ):
        if fit_result is not None:
            trained_model = fit_result["model"]
            parameter_samples = fit_result["parameter_samples"]
            if target_mean is None:
                target_mean = float(fit_result.get("target_mean", 0.0))

        if trained_model is None or parameter_samples is None:
            raise ValueError("Model and parameter_samples are required.")

        if X_train_raw is None:
            X_train_raw = X_train
        if X_train_raw is None or y_train is None:
            raise ValueError("X_train and y_train are required.")

        self.trained_model_ = trained_model
        self.parameter_samples_ = parameter_samples
        self.preprocessor_ = preprocessor
        self.include_local_level_ = model_has_local_level(trained_model)
        self.target_mean_ = float(target_mean or 0.0)
        self.model_name = model_name

        if raw_feature_names is None:
            if isinstance(X_train_raw, pd.DataFrame):
                raw_feature_names = list(X_train_raw.columns)
            else:
                raise ValueError("raw_feature_names required for non-DataFrame X.")

        if transformed_feature_names is None:
            if preprocessor is None:
                transformed_feature_names = list(raw_feature_names)
            elif hasattr(preprocessor, "get_feature_names_out"):
                transformed_feature_names = list(preprocessor.get_feature_names_out())
            else:
                raise ValueError("transformed_feature_names required.")

        self.feature_names_in_ = np.asarray(raw_feature_names, dtype=object)
        self.transformed_feature_names_ = np.asarray(
            transformed_feature_names, dtype=object
        )
        self.n_features_in_ = len(self.feature_names_in_)
        self.n_transformed_features_ = len(self.transformed_feature_names_)

        # Extract posterior parameters
        extracted = extract_static_model_parameters(
            model=trained_model, parameter_samples=parameter_samples,
            n_transformed_features=self.n_transformed_features_,
        )
        self.beta_samples_ = np.asarray(extracted["beta_samples"], dtype=float)
        self.observation_scale_samples_ = extracted["observation_scale_samples"]
        self.level_scale_samples_ = extracted["level_scale_samples"]
        self.regression_type_ = extracted["regression_type"]
        self.n_posterior_samples_ = self.beta_samples_.shape[0]
        self.beta_mean_ = self.beta_samples_.mean(axis=0)

        self.observation_noise_variance_ = np.maximum(
            np.square(self.observation_scale_samples_), 1e-12
        )
        self.state_noise_variance_ = np.maximum(
            np.square(self.level_scale_samples_), 0.0
        )

        # Initialize state
        if self.include_local_level_ and initial_level_prior is not None:
            init_mean = np.asarray(initial_level_prior.mean(), dtype=float).ravel()
            init_var = np.asarray(initial_level_prior.variance(), dtype=float).ravel()
        else:
            init_mean = np.zeros(1)
            init_var = np.zeros(1)

        if len(init_mean) == 1:
            init_mean = np.full(self.n_posterior_samples_, init_mean[0])
        if len(init_var) == 1:
            init_var = np.full(self.n_posterior_samples_, init_var[0])

        self.current_state_ = IncrementalTFPState(
            predicted_state_mean=init_mean,
            predicted_state_variance=np.maximum(init_var, 0.0),
        )

        # Assimilate training data
        X_raw = self._validate_X(X_train_raw, sort=True)
        y_series = self._coerce_y(y_train, expected_index=X_raw.index)
        self._update_internal(X=X_raw, y=y_series,
                             enforce_future=False, return_predictions=False)
        self.initial_snapshot_ = self.snapshot()

    # ----- Input handling -----

    def _validate_X(self, X, *, sort=False) -> pd.DataFrame:
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X = pd.DataFrame(arr, columns=self.feature_names_in_)
        result = X.loc[:, self.feature_names_in_].copy()
        if sort and not result.index.is_monotonic_increasing:
            result = result.sort_index()
        return result

    @staticmethod
    def _coerce_y(y, *, expected_index) -> pd.Series:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            return y.reindex(expected_index).astype(float)
        arr = np.asarray(y, dtype=float).reshape(-1)
        return pd.Series(arr, index=expected_index, name="target")

    def transform(self, X) -> pd.DataFrame:
        X_input = self._validate_X(X)
        if self.preprocessor_ is None:
            return X_input.loc[:, self.transformed_feature_names_].astype(float)
        transformed = self.preprocessor_.transform(X_input)
        if not isinstance(transformed, pd.DataFrame):
            transformed = pd.DataFrame(
                np.asarray(transformed, dtype=float),
                index=X_input.index, columns=self.transformed_feature_names_,
            )
        return transformed.loc[:, self.transformed_feature_names_].astype(float)

    # ----- Prediction (read-only) -----

    def predict(self, X) -> np.ndarray:
        X_t = self.transform(X).to_numpy(dtype=float)
        regression = X_t @ self.beta_samples_.T
        centered = (regression + self.current_state_.predicted_state_mean[None, :]).mean(axis=1)
        return centered + self.target_mean_

    def predict_std(self, X, *, include_observation_noise=True) -> np.ndarray:
        X_t = self.transform(X).to_numpy(dtype=float)
        samples = X_t @ self.beta_samples_.T + self.current_state_.predicted_state_mean[None, :]
        cond_var = self.current_state_.predicted_state_variance.copy()
        if include_observation_noise:
            cond_var += self.observation_noise_variance_
        first = samples.mean(axis=1)
        second = np.mean(np.square(samples) + cond_var[None, :], axis=1)
        return np.sqrt(np.maximum(second - np.square(first), 0.0))

    # ----- Kalman update -----

    def _update_internal(self, *, X, y, enforce_future, return_predictions):
        X_raw = self._validate_X(X, sort=True)
        y_original = self._coerce_y(y, expected_index=X_raw.index)
        y_centered = y_original - self.target_mean_

        if len(X_raw) == 0:
            return pd.DataFrame(index=X_raw.index) if return_predictions else self

        if (enforce_future and self.current_state_.timestamp is not None
                and X_raw.index[0] <= self.current_state_.timestamp):
            raise ValueError("Updates must be after current state timestamp.")

        X_np = self.transform(X_raw).to_numpy(dtype=float)
        y_np = y_centered.to_numpy(dtype=float)
        m = self.current_state_.predicted_state_mean.copy()
        P = self.current_state_.predicted_state_variance.copy()
        Q, R = self.state_noise_variance_, self.observation_noise_variance_
        records = []

        for ts, x_t, y_c, y_o in zip(
            X_raw.index, X_np, y_np, y_original.to_numpy(dtype=float)
        ):
            reg_t = self.beta_samples_ @ x_t
            pred_c = m + reg_t
            pred_o = pred_c + self.target_mean_

            if self.include_local_level_:
                inn = y_c - pred_c
                inn_var = np.maximum(P + R, 1e-12)
                gain = P / inn_var
                filt_m = m + gain * inn
                filt_P = np.maximum((1.0 - gain) * P, 0.0)
                m, P = filt_m, filt_P + Q
            # else: state unchanged

            if return_predictions:
                mean_pred = float(pred_o.mean())
                records.append({
                    "timestamp": ts, "actual": float(y_o),
                    "prediction": mean_pred,
                    "error": float(y_o - mean_pred),
                })

        self.current_state_ = IncrementalTFPState(
            predicted_state_mean=m, predicted_state_variance=P,
            timestamp=X_raw.index[-1],
            n_updates=self.current_state_.n_updates + len(X_raw),
        )
        if return_predictions:
            return pd.DataFrame(records).set_index("timestamp")
        return self

    def update(self, X, y, *, return_predictions=False):
        """Assimilate observations and advance the state."""
        return self._update_internal(
            X=X, y=y, enforce_future=True,
            return_predictions=return_predictions,
        )

    def snapshot(self) -> IncrementalTFPState:
        return self.current_state_.copy()

    def set_state(self, state: IncrementalTFPState):
        self.current_state_ = state.copy()
        return self

    def reset(self):
        """Restore end-of-training state."""
        self.current_state_ = self.initial_snapshot_.copy()
        return self

    @property
    def state_timestamp(self):
        return self.current_state_.timestamp

    def coefficient_summary(self, *, sort_by_absolute_mean=True) -> pd.DataFrame:
        result = pd.DataFrame({
            "mean": self.beta_samples_.mean(axis=0),
            "std": self.beta_samples_.std(axis=0),
            "q025": np.quantile(self.beta_samples_, 0.025, axis=0),
            "q975": np.quantile(self.beta_samples_, 0.975, axis=0),
        }, index=self.transformed_feature_names_)
        result["abs_mean"] = result["mean"].abs()
        result["significant"] = (result["q025"] > 0) | (result["q975"] < 0)
        if sort_by_absolute_mean:
            result = result.sort_values("abs_mean", ascending=False)
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["trained_model_"] = None
        state["parameter_samples_"] = None
        return state


# Backward-compatible alias
IncrementalLocalLevelTFPRegressor = IncrementalTFPRegressor


# =============================================================================
# Plotting
# =============================================================================

def plot_sts_analysis(analysis_result, *, title_prefix=None):
    """Plot diagnostics from analyse_sts_predictions result."""
    pr = analysis_result["prediction_results"]
    include_ll = analysis_result.get("include_local_level", False)
    reg_type = analysis_result.get("regression_type", "regression")

    if title_prefix is None:
        title_prefix = (
            f"TFP LocalLevel + {reg_type}" if include_ll
            else f"TFP {reg_type}"
        )

    # Prediction plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(pr.index, pr["updated_prediction"], label="Prediction", lw=1.4)
    if "updated_lower_95" in pr.columns:
        ax.fill_between(pr.index, pr["updated_lower_95"],
                        pr["updated_upper_95"], alpha=0.2, label="95% CI")
    ax.plot(pr.index, pr["actual"], label="Actual", lw=1.6, ls="--")
    ax.set_title(f"{title_prefix}: Prediction")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # Errors plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(pr.index, pr["open_loop_absolute_error"], label="Open-loop |error|", lw=1.2)
    ax.plot(pr.index, pr["updated_absolute_error"], label="Updated |error|", lw=1.2)
    ax.set_title(f"{title_prefix}: Absolute Errors")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
