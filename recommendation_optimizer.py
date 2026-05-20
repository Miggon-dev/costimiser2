import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


def _invariant_penalty(candidate_df, reference_df, invariants):
    """
    Soft penalty for invariant violations.

    Each invariant is a dict with:
        name
        variables
        fn(row, ref) -> residual
        tolerance
        weight

    The invariant is satisfied when abs(residual) <= tolerance.
    """
    if not invariants:
        return 0.0, []

    row = candidate_df.iloc[0]
    ref = reference_df.iloc[0]

    total_penalty = 0.0
    evaluations = []

    for inv in invariants:
        name = inv.get("name", "unnamed_invariant")
        variables = inv.get("variables", [])
        fn = inv.get("fn")
        tolerance = float(inv.get("tolerance", 1e-6))
        weight = float(inv.get("weight", 1e6))

        if fn is None:
            continue

        missing = [
            v for v in variables
            if v not in candidate_df.columns or v not in reference_df.columns
        ]

        if missing:
            evaluations.append(
                {
                    "name": name,
                    "variables": variables,
                    "residual": None,
                    "violation": None,
                    "tolerance": tolerance,
                    "satisfied": False,
                    "error": f"Missing variables: {missing}",
                }
            )
            total_penalty += weight
            continue

        try:
            residual = float(fn(row, ref))
            violation = max(0.0, abs(residual) - tolerance)
            satisfied = violation <= 0.0

            total_penalty += weight * violation**2

            evaluations.append(
                {
                    "name": name,
                    "variables": variables,
                    "residual": residual,
                    "violation": violation,
                    "tolerance": tolerance,
                    "satisfied": satisfied,
                }
            )

        except Exception as e:
            evaluations.append(
                {
                    "name": name,
                    "variables": variables,
                    "residual": None,
                    "violation": None,
                    "tolerance": tolerance,
                    "satisfied": False,
                    "error": str(e),
                }
            )
            total_penalty += weight

    return total_penalty, evaluations

def _predict_scalar(predict_fn, df):
    y = predict_fn(df)
    arr = np.asarray(y).reshape(-1)
    return float(arr[0])


def _resolve_quality_constraints(quality_constraints):
    if not quality_constraints:
        return []

    from prediction_tools import PREDICTORS

    resolved = []

    for c in quality_constraints:
        metric = c.get("metric")
        operator = c.get("operator", ">=")
        threshold = float(c.get("threshold"))

        spec = PREDICTORS.get(metric)

        resolved.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "predict_fn": spec.get("predict_fn") if spec else None,
                "available": spec is not None,
            }
        )

    return resolved


def _quality_constraint_penalty(candidate_df, resolved_constraints, penalty_weight):
    penalty = 0.0
    evaluations = []

    for c in resolved_constraints:
        metric = c["metric"]
        operator = c["operator"]
        threshold = c["threshold"]
        predict_fn = c.get("predict_fn")

        if predict_fn is None:
            evaluations.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "predicted_value": None,
                    "satisfied": False,
                    "available": False,
                }
            )
            penalty += penalty_weight
            continue

        predicted = _predict_scalar(predict_fn, candidate_df)

        if operator == ">=":
            violation = max(0.0, threshold - predicted)
        elif operator == "<=":
            violation = max(0.0, predicted - threshold)
        else:
            violation = 0.0

        evaluations.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "predicted_value": predicted,
                "satisfied": violation <= 0.0,
                "violation": violation,
                "available": True,
            }
        )

        penalty += penalty_weight * violation**2

    return penalty, evaluations

def _as_dataframe_row(row):
    if isinstance(row, pd.Series):
        return row.to_frame().T
    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError("reference_row must contain exactly one row.")
        return row.copy()
    raise TypeError("reference_row must be a pandas Series or one-row DataFrame.")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _get_variable_bounds(
    historical_df,
    variables,
    lower_q=0.05,
    upper_q=0.95,
):
    """
    Quantile-based feasibility bounds for each actionable variable.
    This is the first approximation to the joint distribution constraint.
    """
    bounds = {}

    for var in variables:
        if var not in historical_df.columns:
            continue

        s = pd.to_numeric(historical_df[var], errors="coerce").dropna()

        if s.empty:
            continue

        lo = float(s.quantile(lower_q))
        hi = float(s.quantile(upper_q))

        if not np.isfinite(lo) or not np.isfinite(hi):
            continue

        if lo == hi:
            continue

        bounds[var] = (lo, hi)

    return bounds


def _mahalanobis_distance(x, mean, inv_cov):
    d = x - mean
    return float(np.sqrt(d @ inv_cov @ d.T))


def _fit_joint_distribution_constraint(
    historical_df,
    variables,
    quantile=0.95,
    regularization=1e-6,
):
    """
    Approximate joint-distribution feasibility using Mahalanobis distance.

    A candidate point is considered feasible if its Mahalanobis distance
    is within the historical 95% distance envelope.
    """
    X = historical_df[variables].apply(pd.to_numeric, errors="coerce").dropna()

    if len(X) < max(20, len(variables) * 3):
        return None

    mean = X.mean().to_numpy()
    cov = np.cov(X.to_numpy(), rowvar=False)

    if cov.ndim == 0:
        return None

    cov = cov + np.eye(cov.shape[0]) * regularization

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    distances = np.array([
        _mahalanobis_distance(row, mean, inv_cov)
        for row in X.to_numpy()
    ])

    max_distance = float(np.quantile(distances, quantile))

    return {
        "mean": mean,
        "inv_cov": inv_cov,
        "max_distance": max_distance,
        "n_reference_rows": len(X),
    }

def _predict_scalar(predict_fn, df):
    y = predict_fn(df)
    arr = np.asarray(y).reshape(-1)
    return float(arr[0])


def _resolve_quality_constraints(quality_constraints):
    """
    Convert parsed quality constraints into callable predictor constraints.

    Example input:
    [
        {"metric": "SCT CD", "operator": ">=", "threshold": 2.1}
    ]
    """
    if not quality_constraints:
        return []

    from prediction_tools import PREDICTORS

    resolved = []

    for c in quality_constraints:
        if not isinstance(c, dict):
            continue

        metric = c.get("metric")
        operator = c.get("operator", ">=")
        threshold = c.get("threshold")

        if metric not in PREDICTORS:
            resolved.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "predict_fn": None,
                    "available": False,
                    "message": f"No predictor registered for quality metric {metric!r}.",
                }
            )
            continue

        try:
            threshold = float(threshold)
        except Exception:
            continue

        resolved.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "predict_fn": PREDICTORS[metric]["predict_fn"],
                "available": True,
                "message": None,
            }
        )

    return resolved


def _quality_constraint_penalty(candidate_df, resolved_constraints, penalty_weight):
    """
    Penalty is zero when all quality constraints are satisfied.
    """
    penalty = 0.0
    evaluations = []

    for c in resolved_constraints:
        metric = c["metric"]
        operator = c["operator"]
        threshold = c["threshold"]
        predict_fn = c.get("predict_fn")

        if predict_fn is None:
            evaluations.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "predicted_value": None,
                    "satisfied": False,
                    "violation": None,
                    "available": False,
                    "message": c.get("message"),
                }
            )
            penalty += penalty_weight
            continue

        predicted = _predict_scalar(predict_fn, candidate_df)

        if operator == ">=":
            violation = max(0.0, threshold - predicted)
        elif operator == "<=":
            violation = max(0.0, predicted - threshold)
        else:
            violation = 0.0

        satisfied = violation <= 0.0

        evaluations.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "predicted_value": predicted,
                "satisfied": satisfied,
                "violation": violation,
                "available": True,
                "message": None,
            }
        )

        penalty += penalty_weight * violation**2

    return penalty, evaluations

def optimize_cost_over_actionable_variables(
    reference_row,
    historical_df,
    actionable_variables,
    cost_function,
    lower_q=0.05,
    upper_q=0.95,
    joint_quantile=0.95,
    joint_penalty_weight=1_000.0,
    quality_constraints=None,
    quality_penalty_weight=1_000_000.0,
    invariants = None,
    objective_mode="minimize",
    maxiter=80,
    seed=42,
):
    """
    Minimize a cost function by changing only actionable variables.

    Parameters
    ----------
    reference_row:
        Current operating point. Series or one-row DataFrame.

    historical_df:
        Historical data used to define feasible operating bounds.

    actionable_variables:
        Variables allowed to change.

    cost_function:
        Function that accepts a one-row DataFrame and returns predicted cost.

    Returns
    -------
    dict with optimized values and expected cost improvement.
    """
    row_df = _as_dataframe_row(reference_row)
    base_row = row_df.copy()

    # Ensure optimization variables accept float assignments
    for col in base_row.columns:
        try:
            base_row[col] = pd.to_numeric(base_row[col])
        except Exception:
            pass

    actionable_variables = [
        v for v in actionable_variables
        if v in base_row.columns and v in historical_df.columns
    ]

    if not actionable_variables:
        return {
            "success": False,
            "message": "No actionable variables found in both reference row and historical data.",
            "actionable_variables": [],
        }

    bounds_dict = _get_variable_bounds(
        historical_df=historical_df,
        variables=actionable_variables,
        lower_q=lower_q,
        upper_q=upper_q,
    )

    variables = list(bounds_dict.keys())

    # Explicit float casting for actionable variables
    for var in variables:
        try:
            base_row[var] = base_row[var].astype(float)
        except Exception:
            pass

    if not variables:
        return {
            "success": False,
            "message": "No valid optimization bounds could be computed.",
            "actionable_variables": actionable_variables,
        }

    bounds = [bounds_dict[v] for v in variables]

    joint_constraint = _fit_joint_distribution_constraint(
        historical_df=historical_df,
        variables=variables,
        quantile=joint_quantile,
    )

    def predict_cost(df):
        y = cost_function(df)
        arr = np.asarray(y).reshape(-1)
        return float(arr[0])

    current_objective_value = predict_cost(base_row)

    resolved_quality_constraints = _resolve_quality_constraints(quality_constraints)

    _, current_quality_evaluations = _quality_constraint_penalty(
        base_row,
        resolved_quality_constraints,
        penalty_weight=quality_penalty_weight,
    )

    _, current_invariant_evaluations = _invariant_penalty(
        base_row,
        base_row,
        invariants,
    )

    def objective(x):
        candidate = base_row.copy()

        for var, value in zip(variables, x):
            candidate.at[candidate.index[0], var] = float(value)

        objective_value = predict_cost(candidate)

        model_objective = (
            -objective_value
            if objective_mode == "maximize"
            else objective_value
        )

        quality_penalty, _ = _quality_constraint_penalty(
            candidate,
            resolved_quality_constraints,
            penalty_weight=quality_penalty_weight,
        )

        invariant_penalty, _ = _invariant_penalty(
            candidate,
            base_row,
            invariants,
        )

        penalty = 0.0

        if joint_constraint is not None:
            dist = _mahalanobis_distance(
                np.asarray(x),
                joint_constraint["mean"],
                joint_constraint["inv_cov"],
            )

            excess = max(0.0, dist - joint_constraint["max_distance"])
            penalty = joint_penalty_weight * excess**2

        return model_objective + penalty + quality_penalty + invariant_penalty

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        seed=seed,
        polish=True,
        updating="immediate",
        workers=1,
    )

    optimized_row = base_row.copy()

    for var, value in zip(variables, result.x):
        optimized_row.at[optimized_row.index[0], var] = float(value)

    optimized_objective_value = predict_cost(optimized_row)

    _, optimized_quality_evaluations = _quality_constraint_penalty(
        optimized_row,
        resolved_quality_constraints,
        quality_penalty_weight,
    )

    _, optimized_invariant_evaluations = _invariant_penalty(
        optimized_row,
        base_row,
        invariants,
    )


    changes = []

    for var in variables:
        current_value = _safe_float(base_row.iloc[0][var])
        optimized_value = _safe_float(optimized_row.iloc[0][var])

        changes.append(
            {
                "variable": var,
                "current_value": current_value,
                "optimized_value": optimized_value,
                "delta": optimized_value - current_value,
                "relative_delta": (
                    (optimized_value - current_value) / abs(current_value)
                    if current_value not in [0, np.nan] and np.isfinite(current_value)
                    else np.nan
                ),
                "lower_bound": bounds_dict[var][0],
                "upper_bound": bounds_dict[var][1],
            }
        )

    return {
        "success": bool(result.success),
        "message": str(result.message),
        # objective information
        "objective_mode": objective_mode,
        "current_objective_value": float(current_objective_value),
        "optimized_objective_value": float(optimized_objective_value),
        "objective_delta": float(
            optimized_objective_value - current_objective_value
        ),
        # backward-compatible cost fields
        "current_cost": float(current_objective_value),
        "optimized_cost": float(optimized_objective_value),
        "expected_cost_delta": float(
            optimized_objective_value - current_objective_value
        ),
        "expected_cost_reduction": float(
            current_objective_value - optimized_objective_value
            if objective_mode != "maximize"
            else optimized_objective_value - current_objective_value
        ),
        # optimization setup
        "actionable_variables": list(actionable_variables),
        "variables": list(variables),
        "bounds": {
            v: {
                "lower_bound": float(bounds_dict[v][0]),
                "upper_bound": float(bounds_dict[v][1]),
            }
            for v in variables
        },
        "lower_q": float(lower_q),
        "upper_q": float(upper_q),
        # joint feasibility
        "joint_constraint_used": joint_constraint is not None,
        "joint_quantile": float(joint_quantile),
        "joint_constraint": joint_constraint,
        # rows
        "reference_row": base_row.to_dict(orient="records")[0],
        "optimized_row": optimized_row.to_dict(orient="records")[0],
        # optimized variable changes
        "changes": changes,
        # scipy diagnostics
        "optimizer_status": int(getattr(result, "status", -1)),
        "optimizer_nit": int(getattr(result, "nit", -1)),
        "optimizer_fun": float(result.fun),
        # quality constraints
        "quality_constraints": quality_constraints or [],
        "quality_constraints_resolved": [
            {
                "metric": c.get("metric"),
                "operator": c.get("operator"),
                "threshold": c.get("threshold"),
                "available": c.get("available"),
                "message": c.get("message"),
            }
            for c in resolved_quality_constraints
        ],
        "current_quality_evaluations": current_quality_evaluations,
        "optimized_quality_evaluations": optimized_quality_evaluations,
        "quality_constraints_satisfied": (
            all(e.get("satisfied") for e in optimized_quality_evaluations)
            if optimized_quality_evaluations
            else None
        ),
        # optional raw optimizer object
        "optimizer_result": result,
        # invariants
        "invariants": invariants or [],
        "current_invariant_evaluations": current_invariant_evaluations,
        "optimized_invariant_evaluations": optimized_invariant_evaluations,
        "invariants_satisfied": (
            all(e.get("satisfied") for e in optimized_invariant_evaluations)
            if optimized_invariant_evaluations
            else None
        ),
    }

def optimize_cost_with_intervention_limit(
    reference_row,
    historical_df,
    actionable_variables,
    cost_function,
    max_interventions=5,
    objective_mode="minimize",
    invariants=None,
    quality_constraints=None,
    **optimizer_kwargs,
):
    """
    Select up to max_interventions variables and optimize them.

    Greedy forward selection:
    - start with no variables
    - at each step, try adding one candidate variable/group
    - keep the candidate that gives the best objective improvement
    - final result uses only selected variables
    """

    if max_interventions is None:
        return optimize_cost_over_actionable_variables(
            reference_row=reference_row,
            historical_df=historical_df,
            actionable_variables=actionable_variables,
            cost_function=cost_function,
            objective_mode=objective_mode,
            invariants=invariants,
            quality_constraints=quality_constraints,
            **optimizer_kwargs,
        )

    if isinstance(max_interventions, str) and max_interventions.lower() == "all":
        return optimize_cost_over_actionable_variables(
            reference_row=reference_row,
            historical_df=historical_df,
            actionable_variables=actionable_variables,
            cost_function=cost_function,
            objective_mode=objective_mode,
            invariants=invariants,
            quality_constraints=quality_constraints,
            **optimizer_kwargs,
        )

    max_interventions = int(max_interventions)

    actionable_variables = list(dict.fromkeys(actionable_variables))

    if max_interventions >= len(actionable_variables):
        return optimize_cost_over_actionable_variables(
            reference_row=reference_row,
            historical_df=historical_df,
            actionable_variables=actionable_variables,
            cost_function=cost_function,
            objective_mode=objective_mode,
            invariants=invariants,
            quality_constraints=quality_constraints,
            **optimizer_kwargs,
        )

    selected = []
    remaining = actionable_variables.copy()
    best_result = None

    def score(opt):
        value = opt.get("optimized_objective_value", opt.get("optimized_cost"))

        if value is None:
            return -np.inf

        value = float(value)

        if objective_mode == "maximize":
            return value

        return -value

    for _ in range(max_interventions):
        best_candidate = None
        best_candidate_result = None
        best_candidate_score = -np.inf

        for var in remaining:
            trial_vars = selected + [var]

            opt = optimize_cost_over_actionable_variables(
                reference_row=reference_row,
                historical_df=historical_df,
                actionable_variables=trial_vars,
                cost_function=cost_function,
                objective_mode=objective_mode,
                invariants=invariants,
                quality_constraints=quality_constraints,
                **optimizer_kwargs,
            )

            if not opt.get("changes"):
                continue

            s = score(opt)

            if s > best_candidate_score:
                best_candidate = var
                best_candidate_result = opt
                best_candidate_score = s

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_result = best_candidate_result

    if best_result is None:
        return {
            "success": False,
            "message": "No intervention subset improved the objective.",
            "actionable_variables": actionable_variables,
            "selected_variables": [],
            "changes": [],
        }

    best_result["selected_variables"] = selected
    best_result["max_interventions"] = max_interventions
    best_result["selection_mode"] = "greedy"

    return best_result
