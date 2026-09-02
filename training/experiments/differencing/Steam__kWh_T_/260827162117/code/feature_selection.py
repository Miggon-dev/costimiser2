"""
Feature selection via CMA-ES evolutionary strategy.

Selects the best features (with k optionally tuned) to minimize
cross-validated RMSE using Ridge.

Supports feature_groups: PLS score groups that must be selected
all-or-nothing, reducing the search space and preventing invalid subsets.

Performance:
- Inner TimeSeriesSplit for alpha selection (correct for temporal data)
- Caches evaluations for repeated feature subsets
- Operates on numpy arrays (no DataFrame overhead in the inner loop)
- Optional joblib parallelism
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from dataclasses import dataclass
from typing import Callable

import numpy as np
import cma
from tqdm.notebook import tqdm

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# =============================================================================
# Configuration
# =============================================================================

_DEFAULT_ALPHAS = np.logspace(0, 3, 20)
_INNER_CV_SPLITS = 5


# =============================================================================
# Inner model: Ridge with time-series-aware alpha selection
# =============================================================================

def _fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: np.ndarray,
    feature_names: list[str] | None = None,
    splines: bool = False,
    spline_n_knots: int = 4,
    spline_degree: int = 2,
):
    """
    Fit Ridge with alpha chosen via inner TimeSeriesSplit CV.

    Parameters
    ----------
    X_train : training data (numpy array)
    y_train : target
    alphas : Ridge alpha grid
    feature_names : column names (needed when splines=True to identify grammage)
    splines : if True, apply SplineTransformer to non-grammage columns
    spline_n_knots : number of knots
    spline_degree : polynomial degree
    """
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, SplineTransformer

    if splines and feature_names is not None:
        # Identify which column indices are grammage vs numeric
        numeric_idx = [i for i, f in enumerate(feature_names) if "grammage" not in f.lower()]
        grammage_idx = [i for i, f in enumerate(feature_names) if "grammage" in f.lower()]

        transformers = []
        if numeric_idx:
            transformers.append((
                "num",
                SKPipeline([
                    ("scaler_in", StandardScaler()),
                    ("splines", SplineTransformer(
                        n_knots=spline_n_knots, degree=spline_degree, include_bias=False,
                    )),
                    ("scaler_out", StandardScaler()),
                ]),
                numeric_idx,
            ))
        if grammage_idx:
            transformers.append(("cat", "passthrough", grammage_idx))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        estimator = SKPipeline([("prep", preprocessor), ("ridge", Ridge())])
        param_grid = {"ridge__alpha": alphas}
    else:
        estimator = SKPipeline([("ridge", Ridge())])
        param_grid = {"ridge__alpha": alphas}

    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=_INNER_CV_SPLITS),
        n_jobs=1,
        refit=True,
    )
    gscv.fit(X_train, y_train)
    gscv.alpha_ = gscv.best_params_["ridge__alpha"]
    return gscv


# =============================================================================
# Result container
# =============================================================================

@dataclass
class FeatureSelectionResult:
    """Result of CMA-ES feature selection."""
    selected_features: list[str]
    selected_idx: np.ndarray
    selected_atoms: list[str]
    best_rmse: float
    best_loss: float
    best_alpha: float
    best_k: int
    n_evals: int
    n_cache_hits: int = 0
    final_estimator: object = None

    @property
    def n_features(self) -> int:
        return len(self.selected_features)

    @property
    def n_atoms(self) -> int:
        return len(self.selected_atoms)


# =============================================================================
# Atom mapping: groups + standalone features -> search space
# =============================================================================

@dataclass
class _AtomMap:
    """
    Maps between CMA-ES atoms and actual feature indices.

    Each atom is either:
    - A single feature (expands to 1 column index)
    - A group (expands to N column indices)
    """
    atom_names: list[str]           # one name per atom
    atom_to_idx: list[np.ndarray]   # atom_i -> array of column indices
    n_atoms: int

    @classmethod
    def build(
        cls,
        all_features: list[str],
        free_idx: np.ndarray,
        feature_groups: dict[str, list[str]],
    ) -> "_AtomMap":
        """
        Build atom map from free features and group constraints.

        Features that belong to a group become one atom.
        Features not in any group become individual atoms.
        """
        # Which free features belong to a group?
        grouped_features: set[str] = set()
        for members in feature_groups.values():
            grouped_features.update(members)

        atom_names = []
        atom_to_idx = []

        # Add group atoms
        for group_name, members in feature_groups.items():
            # Find which of the group's members are in free_idx
            member_indices = []
            for feat in members:
                if feat in all_features:
                    idx = all_features.index(feat)
                    if idx in free_idx:
                        member_indices.append(idx)
            if member_indices:
                atom_names.append(group_name)
                atom_to_idx.append(np.array(member_indices, dtype=int))

        # Add standalone atoms (free features not in any group)
        for idx in free_idx:
            feat = all_features[idx]
            if feat not in grouped_features:
                atom_names.append(feat)
                atom_to_idx.append(np.array([idx], dtype=int))

        return cls(
            atom_names=atom_names,
            atom_to_idx=atom_to_idx,
            n_atoms=len(atom_names),
        )

    def expand_mask(self, atom_mask: np.ndarray) -> np.ndarray:
        """Convert a boolean mask over atoms to an array of feature indices."""
        selected = []
        for i, is_selected in enumerate(atom_mask):
            if is_selected:
                selected.append(self.atom_to_idx[i])
        if not selected:
            return np.array([], dtype=int)
        return np.concatenate(selected)

    def selected_atom_names(self, atom_mask: np.ndarray) -> list[str]:
        return [self.atom_names[i] for i, s in enumerate(atom_mask) if s]


# =============================================================================
# Objective
# =============================================================================

def _evaluate_subset(
    X_np: np.ndarray,
    y: np.ndarray,
    selected_idx: np.ndarray,
    cv_splits: list[tuple],
    alphas: np.ndarray,
    penalty_fn: Callable | None,
    X_columns: list[str],
    splines: bool = False,
    spline_n_knots: int = 4,
    spline_degree: int = 2,
) -> tuple[float, float, float]:
    """Evaluate a feature subset. Returns (loss, rmse, best_alpha)."""
    X_sub = X_np[:, selected_idx]
    feature_names = [X_columns[i] for i in selected_idx]
    rmses = []
    best_alpha = alphas[len(alphas) // 2]

    for tr_idx, te_idx in cv_splits:
        model = _fit_ridge(
            X_sub[tr_idx], y[tr_idx], alphas,
            feature_names=feature_names,
            splines=splines,
            spline_n_knots=spline_n_knots,
            spline_degree=spline_degree,
        )
        pred = model.predict(X_sub[te_idx])
        rmses.append(np.sqrt(mean_squared_error(y[te_idx], pred)))
        best_alpha = model.alpha_

    mean_rmse = float(np.mean(rmses))
    penalty = penalty_fn(X_np, selected_idx, X_columns) if penalty_fn else 0.0
    return mean_rmse + penalty, mean_rmse, best_alpha


# =============================================================================
# Penalty functions
# =============================================================================

def collinearity_penalty(weight: float = 0.1) -> Callable:
    """Log condition number penalty."""
    def _penalty(X_np, selected_idx, columns):
        numeric_idx = [i for i in selected_idx if "grammage" not in columns[i].lower()]
        if len(numeric_idx) < 2:
            return 0.0
        X_sub = X_np[:, numeric_idx]
        std = X_sub.std(axis=0)
        std[std == 0] = 1.0
        X_std = (X_sub - X_sub.mean(axis=0)) / std
        eigvals = np.linalg.eigvalsh(X_std.T @ X_std / len(X_std))
        return weight * float(np.log(eigvals.max() / (eigvals.min() + 1e-8)))
    return _penalty


def parsimony_penalty(weight: float = 0.5) -> Callable:
    """Penalize number of selected features."""
    def _penalty(X_np, selected_idx, columns):
        return weight * len(selected_idx)
    return _penalty


def combined_penalty(*penalties: Callable) -> Callable:
    """Combine multiple penalty functions."""
    def _penalty(X_np, selected_idx, columns):
        return sum(p(X_np, selected_idx, columns) for p in penalties)
    return _penalty


# =============================================================================
# Mask construction
# =============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _topk_mask(z: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros(len(z), dtype=bool)
    if k >= len(z):
        return np.ones(len(z), dtype=bool)
    idx = np.argpartition(_sigmoid(z), -k)[-k:]
    mask = np.zeros(len(z), dtype=bool)
    mask[idx] = True
    return mask


def _stochastic_mask(z: np.ndarray, k: int, rng) -> np.ndarray:
    if k <= 0:
        return np.zeros(len(z), dtype=bool)
    p = _sigmoid(z)
    p = p / p.sum()
    idx = rng.choice(len(z), size=min(k, len(z)), replace=False, p=p)
    mask = np.zeros(len(z), dtype=bool)
    mask[idx] = True
    return mask


def _z_to_k(z_k: float, k_min: int, k_max: int) -> int:
    t = float(_sigmoid(np.array([z_k]))[0])
    return int(np.round(k_min + t * (k_max - k_min)))


# =============================================================================
# Main public function
# =============================================================================

def cmaes_feature_selection(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    k_target: int | None = None,
    k_range: tuple[int, int] | None = None,
    fixed_features: list[str] | None = None,
    feature_groups: dict[str, list[str]] | None = None,
    cv_splits: list[tuple] | None = None,
    selection: str = "topk",
    max_evals: int = 3000,
    sigma0: float = 1.0,
    popsize: int | None = None,
    seed: int = 42,
    penalty_fn: Callable | None = None,
    alphas: np.ndarray | None = None,
    splines: bool = False,
    spline_n_knots: int = 4,
    spline_degree: int = 2,
    n_jobs: int = 1,
    verbose: bool = True,
) -> FeatureSelectionResult:
    """
    Select features using CMA-ES with grouped constraints and optional k tuning.

    Parameters
    ----------
    X : DataFrame with all candidate features (already transformed).
    y : Target array aligned with X.
    k_target : Exact number of *atoms* to select. Mutually exclusive with k_range.
               Note: actual feature count may be larger due to group expansion.
    k_range : (k_min, k_max) range of atoms to select. Jointly optimized.
    fixed_features : Feature names always included (individual features only,
                     not group names).
    feature_groups : Dict mapping group_name -> list of feature names that must
                     be selected together. E.g.:
                     {"gas_decu": ["gas_decu_1", "gas_decu_2", "gas_decu_3"]}
                     CMA-ES sees one atom per group instead of N individual features.
    cv_splits : List of (train_idx, test_idx) tuples.
    selection : "topk" (deterministic) or "stochastic".
    max_evals : Budget of objective function evaluations.
    sigma0 : Initial CMA-ES step size.
    popsize : Population size. None = auto-scaled.
    seed : Random seed.
    penalty_fn : Optional callable(X_np, selected_idx, columns) -> penalty.
    alphas : Ridge alpha grid.
    n_jobs : Parallel workers (1 = sequential).
    verbose : Print progress.

    Returns
    -------
    FeatureSelectionResult

    Examples
    --------
    feature_groups = {
        "gas_decu": ["gas_decu_1", "gas_decu_2", "gas_decu_3"],
        "exha_mois": ["exha_mois_1", "exha_mois_2"],
    }
    res = cmaes_feature_selection(
        X, y,
        k_range=(5, 15),          # 5 to 15 atoms
        fixed_features=["ambient_temp_C"],
        feature_groups=feature_groups,
        penalty_fn=parsimony_penalty(0.3),
    )
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    all_features = list(X.columns)
    n_features = len(all_features)
    X_np = X.values.astype(np.float64)

    if alphas is None:
        alphas = _DEFAULT_ALPHAS
    if feature_groups is None:
        feature_groups = {}

    # Resolve fixed features
    if fixed_features is None:
        fixed_features = []
    fixed_idx = np.array(
        [all_features.index(f) for f in fixed_features if f in all_features],
        dtype=int,
    )
    free_idx = np.setdiff1d(np.arange(n_features), fixed_idx)

    # Build atom map: groups become single atoms, rest are individual
    atom_map = _AtomMap.build(all_features, free_idx, feature_groups)
    n_atoms = atom_map.n_atoms

    if verbose:
        n_groups = sum(1 for a in atom_map.atom_to_idx if len(a) > 1)
        n_solo = n_atoms - n_groups
        print(f"Search space: {n_atoms} atoms ({n_groups} groups, {n_solo} standalone)")

    # Resolve k mode (k now refers to number of atoms, not features)
    tune_k = False
    if k_target is not None and k_range is not None:
        raise ValueError("Provide k_target or k_range, not both.")

    if k_target is not None:
        k_min_atoms = k_target
        k_max_atoms = k_target
        if k_min_atoms > n_atoms:
            raise ValueError(f"k_target ({k_target}) > available atoms ({n_atoms})")
    elif k_range is not None:
        k_min_atoms = max(k_range[0], 1)
        k_max_atoms = min(k_range[1], n_atoms)
        tune_k = (k_min_atoms != k_max_atoms)
    else:
        k_min_atoms = 1
        k_max_atoms = n_atoms
        tune_k = True

    # Default CV
    if cv_splits is None:
        split_pt = int(len(X) * 0.8)
        cv_splits = [(np.arange(split_pt), np.arange(split_pt, len(X)))]

    # --- Edge case: 0 atoms to optimize ---
    if n_atoms == 0:
        loss, rmse, alpha = _evaluate_subset(
            X_np, y, fixed_idx, cv_splits, alphas, penalty_fn, all_features,
            splines=splines, spline_n_knots=spline_n_knots, spline_degree=spline_degree,
        )
        est = _fit_ridge(
            X_np[:, fixed_idx], y, alphas,
            feature_names=[all_features[i] for i in fixed_idx],
            splines=splines, spline_n_knots=spline_n_knots, spline_degree=spline_degree,
        )
        return FeatureSelectionResult(
            selected_features=[all_features[i] for i in fixed_idx],
            selected_idx=fixed_idx,
            selected_atoms=[],
            best_rmse=rmse, best_loss=loss,
            best_alpha=float(est.alpha_),
            best_k=0, n_evals=1, final_estimator=est,
        )

    # --- CMA-ES setup ---
    n_dims = n_atoms + (1 if tune_k else 0)
    rng = np.random.default_rng(seed)

    if popsize is None:
        popsize = max(int(4 + 3 * np.log(n_dims)), 8)

    es = cma.CMAEvolutionStrategy(
        np.zeros(n_dims), sigma0,
        {"seed": seed, "maxfevals": max_evals, "verb_disp": 0, "popsize": popsize},
    )

    best_loss = np.inf
    best_rmse = np.inf
    best_alpha = float(alphas[len(alphas) // 2])
    best_selected = None
    best_atom_mask = None

    cache: dict[frozenset, tuple[float, float, float]] = {}
    cache_hits = 0
    evals_done = 0

    desc = f"CMA-ES ({n_atoms} atoms, k={'tune' if tune_k else k_min_atoms})"
    pbar = tqdm(total=max_evals, desc=desc, disable=not verbose)

    def _eval_one(z_full):
        nonlocal cache_hits
        z_full = np.asarray(z_full)

        # Decode k (number of atoms to select)
        if tune_k:
            z_atoms = z_full[:-1]
            k_atoms = _z_to_k(z_full[-1], k_min_atoms, k_max_atoms)
        else:
            z_atoms = z_full
            k_atoms = k_min_atoms

        # Select top-k atoms
        if selection == "topk":
            atom_mask = _topk_mask(z_atoms, k_atoms)
        else:
            atom_mask = _stochastic_mask(z_atoms, k_atoms, rng)

        # Expand atoms to feature indices
        free_selected = atom_map.expand_mask(atom_mask)
        selected = np.sort(np.concatenate([fixed_idx, free_selected]))
        key = frozenset(selected.tolist())

        if key in cache:
            cache_hits += 1
            return cache[key][0], cache[key], selected, atom_mask

        loss, rmse, alpha = _evaluate_subset(
            X_np, y, selected, cv_splits, alphas, penalty_fn, all_features,
            splines=splines, spline_n_knots=spline_n_knots, spline_degree=spline_degree,
        )
        cache[key] = (loss, rmse, alpha)
        return loss, (loss, rmse, alpha), selected, atom_mask

    # --- Optimization loop ---
    while not es.stop() and evals_done + es.popsize <= max_evals:
        solutions = es.ask()

        if n_jobs == 1:
            results = [_eval_one(z) for z in solutions]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_eval_one)(z) for z in solutions
            )

        losses = []
        for loss, (_, rmse, alpha), selected, atom_mask in results:
            losses.append(loss)
            if loss < best_loss:
                best_loss, best_rmse, best_alpha = loss, rmse, alpha
                best_selected = selected
                best_atom_mask = atom_mask
                if verbose:
                    atoms_sel = atom_map.selected_atom_names(atom_mask)
                    print(
                        f"  NEW BEST: loss={loss:.4f} rmse={rmse:.4f} "
                        f"atoms={len(atoms_sel)} features={len(selected)} "
                        f"evals={evals_done + len(losses)}"
                    )

        es.tell(solutions, losses)
        evals_done += len(solutions)
        pbar.update(len(solutions))
        pbar.set_postfix(
            best=f"{best_rmse:.3f}",
            k_atoms=int(best_atom_mask.sum()) if best_atom_mask is not None else "?",
            k_feat=len(best_selected) if best_selected is not None else "?",
            cache=len(cache),
        )

    pbar.close()

    # --- Fit final model (on train only) ---
    train_idx = cv_splits[0][0]
    final_est = _fit_ridge(
        X_np[train_idx][:, best_selected], y[train_idx], alphas,
        feature_names=[all_features[i] for i in best_selected],
        splines=splines, spline_n_knots=spline_n_knots, spline_degree=spline_degree,
    )

    selected_atoms = atom_map.selected_atom_names(best_atom_mask) if best_atom_mask is not None else []

    return FeatureSelectionResult(
        selected_features=[all_features[i] for i in best_selected],
        selected_idx=best_selected,
        selected_atoms=selected_atoms,
        best_rmse=best_rmse,
        best_loss=best_loss,
        best_alpha=float(final_est.alpha_),
        best_k=len(selected_atoms),
        n_evals=evals_done,
        n_cache_hits=cache_hits,
        final_estimator=final_est,
    )


# =============================================================================
# Skip feature selection: fit on all features
# =============================================================================

def fit_all_features(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    cv_splits: list[tuple] | None = None,
    model=None,
    alphas: np.ndarray | None = None,
    splines: bool = False,
    spline_n_knots: int = 4,
    spline_degree: int = 2,
) -> FeatureSelectionResult:
    """
    Fit a model on all features without any feature selection.

    Drop-in replacement for cmaes_feature_selection when you want
    to use all available features.

    Parameters
    ----------
    X : DataFrame with all candidate features.
    y : Target array.
    cv_splits : List of (train_idx, test_idx) tuples. The final estimator
                is fitted on the train portion of the first split only.
                If None, fits on all data (no holdout).
    model : Optional sklearn-compatible estimator. If provided, uses this
            instead of Ridge (ignores alphas/splines params).
            Must implement fit(X, y) and predict(X).
    alphas : Ridge alpha grid. None = default.
    splines : Apply SplineTransformer to non-grammage columns (Ridge only).
    spline_n_knots : Number of knots (Ridge only).
    spline_degree : Polynomial degree (Ridge only).

    Returns
    -------
    FeatureSelectionResult (with all features selected).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    all_features = list(X.columns)
    X_np = X.values.astype(np.float64)

    if alphas is None:
        alphas = _DEFAULT_ALPHAS

    # Fit on train portion only (consistent with _evaluate_subset)
    if cv_splits is not None and len(cv_splits) > 0:
        train_idx = cv_splits[0][0]
        X_fit = X_np[train_idx]
        y_fit = y[train_idx]
    else:
        X_fit = X_np
        y_fit = y

    if model is not None:
        # Use the provided model directly
        from sklearn.base import clone
        est = clone(model)
        est.fit(X_fit, y_fit)
        est.alpha_ = 0.0  # dummy for interface compatibility
    else:
        # Default: Ridge with alpha CV + optional splines
        est = _fit_ridge(
            X_fit, y_fit, alphas,
            feature_names=all_features,
            splines=splines,
            spline_n_knots=spline_n_knots,
            spline_degree=spline_degree,
        )

    return FeatureSelectionResult(
        selected_features=all_features,
        selected_idx=np.arange(len(all_features)),
        selected_atoms=all_features,
        best_rmse=0.0,  # not computed (no holdout eval)
        best_loss=0.0,
        best_alpha=float(est.alpha_),
        best_k=len(all_features),
        n_evals=0,
        n_cache_hits=0,
        final_estimator=est,
    )
