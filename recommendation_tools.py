"""
recommendation_tools.py
Recommendation layer for the AI Process Assistant.
This module converts:
- diagnosis scope
- cost driver evidence
- extreme cluster process values
into:
- actionable recommendations
- structured recommendation candidates
Knowledge/RAG input can be added later on top of this same schema.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd

try:
    import recommendation_config as rc
except Exception:
    rc = None

def _model_features_for_target(target) -> set:
    try:
        from prediction_tools import PREDICTORS

        spec = PREDICTORS.get(target)

        if spec is None:
            # normalized fallback
            norm_target = _normalize_manual_target_name(target)
            spec = PREDICTORS.get(norm_target)

        if spec is None:
            return set()

        features_fn = spec.get("features_fn")
        if features_fn is None:
            return set()

        features = features_fn() if callable(features_fn) else features_fn

        return {str(f) for f in features}

    except Exception:
        return set()


def _records_from_manual_actionable_inputs(target=None) -> List[Dict[str, Any]]:
    manual_inputs = _manual_actionable_inputs(target)
    model_features = _model_features_for_target(target)

    if model_features:
        manual_inputs = {
            v for v in manual_inputs
            if v in model_features
        }

    records = []

    for v in manual_inputs:
        records.append(
            {
                "variable": v,
                "contribution": None,
                "candidate_source": "manual",
            }
        )

    return records

def _use_manual_actionable_inputs() -> bool:
    return bool(_cfg("RECOMMENDATION_USE_MANUAL_ACTIONABLE_INPUTS", False))


def _normalize_manual_target_name(x) -> str:
    s = str(x or "").strip().lower()
    s = s.replace("_cost", "")
    s = s.replace(" cost", "")
    s = s.replace("_", " ")
    s = " ".join(s.split())

    aliases = {
        "fiber": "fibre",
        "fibre": "fibre",
        "steam": "steam",
        "electricity": "electricity",
        "power": "electricity",
        "starch": "starch",
        "total": "total",
        "combined": "total",
        "combined cost": "total",
        "overall": "total",
        "sct cd": "SCT CD",
        "sctcd": "SCT CD",
        "sct md": "SCT MD",
        "sctmd": "SCT MD",
        "burst": "Burst",
        "cmt30": "CMT30",
        "cmt 30": "CMT30",
        "cmt": "CMT30",
    }

    return aliases.get(s, s)


def _manual_actionable_inputs(target=None) -> set:
    by_target = _cfg("RECOMMENDATION_MANUAL_ACTIONABLE_INPUTS_BY_TARGET", None)

    if isinstance(by_target, dict):
        normalized_lookup = {
            _normalize_manual_target_name(k): v
            for k, v in by_target.items()
        }

        norm_target = _normalize_manual_target_name(target)

        values = normalized_lookup.get(
            norm_target,
            _cfg("RECOMMENDATION_MANUAL_ACTIONABLE_INPUTS_DEFAULT", []),
        )

        return {str(v) for v in (values or [])}

    # backward compatibility with your current single-list setting
    values = _cfg("RECOMMENDATION_MANUAL_ACTIONABLE_INPUTS", [])

    if values is None:
        return set()

    return {str(v) for v in values}


def _optimizer_enabled() -> bool:
    return bool(_cfg("RECOMMENDATION_USE_OPTIMIZER", False))

def _cfg(name: str, default):
    if rc is None:
        return default
    return getattr(rc, name, default)


def _recommendation_feature_source() -> str:
    return str(
        _cfg("RECOMMENDATION_FEATURE_SOURCE", "drivers")
    ).strip().lower()


def _action_limit_count(n: int) -> int:
    limit = _cfg("RECOMMENDATION_ACTION_LIMIT", 3)

    if n <= 0:
        return 0

    if limit is None:
        return n

    if isinstance(limit, str) and limit.strip().lower() == "all":
        return n

    try:
        limit = int(limit)
    except Exception:
        limit = 3

    return max(1, min(limit, n))

def _top_frac_count(n: int, frac) -> int:
    if n <= 0:
        return 0

    try:
        frac = float(frac)
    except Exception:
        frac = 1.0

    frac = max(0.0, min(1.0, frac))

    if frac >= 1.0:
        return n

    return max(1, int(round(n * frac)))


def _rag_limit_count(n: int) -> int:
    limit = _cfg("RECOMMENDATION_RAG_VARIABLE_LIMIT", "all")

    if limit is None:
        return n

    if isinstance(limit, str) and limit.strip().lower() == "all":
        return n

    try:
        limit = int(limit)
    except Exception:
        return n

    return max(1, min(limit, n))


def _cfg(name: str, default):
    if rc is None:
        return default
    return getattr(rc, name, default)


def _recommendation_feature_source() -> str:
    return str(
        _cfg("RECOMMENDATION_FEATURE_SOURCE", "drivers")
    ).strip().lower()


def _top_frac_count(n: int, frac) -> int:
    if n <= 0:
        return 0

    try:
        frac = float(frac)
    except Exception:
        frac = 1.0

    frac = max(0.0, min(1.0, frac))

    if frac >= 1.0:
        return n

    return max(1, int(round(n * frac)))


def _rag_limit_count(n: int) -> int:
    limit = _cfg("RECOMMENDATION_RAG_VARIABLE_LIMIT", "all")

    if limit is None:
        return n

    if isinstance(limit, str) and limit.strip().lower() == "all":
        return n

    try:
        limit = int(limit)
    except Exception:
        return n

    return max(1, min(limit, n))


ALLOWED_RECOMMENDATION_FEATURE_SOURCES = {"drivers", "model", "auto"}


def _get_recommendation_feature_source() -> str:
    try:
        import recommendation_config as rc
        source = getattr(rc, "RECOMMENDATION_FEATURE_SOURCE", "drivers")
    except Exception:
        source = "drivers"

    source = str(source or "drivers").strip().lower()

    if source not in ALLOWED_RECOMMENDATION_FEATURE_SOURCES:
        raise ValueError(
            "RECOMMENDATION_FEATURE_SOURCE must be one of "
            f"{sorted(ALLOWED_RECOMMENDATION_FEATURE_SOURCES)}, got {source!r}"
        )

    return source


def _pick_lang(lang: Optional[str]) -> str:
    return "de" if lang == "de" else "en"

def _normalize_component(cost_component: Optional[str]) -> str:
    if cost_component is None:
        return "generic"
    key = str(cost_component).strip().lower()
    if key == "fiber":
        key = "fibre"
    return key

def _extract_focus(
    diagnosis_result: Optional[Dict[str, Any]],
    cost_driver_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:

    cost_driver_result = cost_driver_result or {}

    return {
        "grade": cost_driver_result.get("grade"),
        "cost_component": cost_driver_result.get("cost_component"),
        "target_range": cost_driver_result.get("target_range"),
        "baseline_range": cost_driver_result.get("baseline_range"),
    }

def _extract_top_driver_records(
    shapley_contrib: pd.DataFrame,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if shapley_contrib is None or shapley_contrib.empty:
        return []

    df = shapley_contrib.copy()

    if "variable" not in df.columns or "contribution" not in df.columns:
        return []

    df = df.sort_values("contribution", ascending=False)

    frac = _cfg("RECOMMENDATION_COST_DRIVER_TOP_FRAC", 1.0)
    n_keep = _top_frac_count(len(df), frac)

    df = df.head(n_keep)

    if top_n is not None:
        df = df.head(top_n)

    records = df.to_dict(orient="records")

    for r in records:
        r["candidate_source"] = "drivers"

    return records

def _extract_top_model_feature_records(
    shap_result: Optional[Dict[str, Any]],
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:

    if shap_result is None or not isinstance(shap_result, dict):
        return []

    shap_df = shap_result.get("data_frame", pd.DataFrame())

    if shap_df is None or shap_df.empty:
        return []

    if not {"feature", "shap_value"}.issubset(shap_df.columns):
        return []

    summary = (
        shap_df.copy()
        .assign(
            shap_value=lambda x: pd.to_numeric(x["shap_value"], errors="coerce"),
        )
        .dropna(subset=["feature", "shap_value"])
        .assign(abs_shap=lambda x: x["shap_value"].abs())
        .groupby("feature", as_index=False)
        .agg(
            mean_abs_shap=("abs_shap", "mean"),
            mean_signed_shap=("shap_value", "mean"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )

    frac = _cfg("RECOMMENDATION_SHAP_TOP_FRAC", 1.0)
    n_keep = _top_frac_count(len(summary), frac)

    summary = summary.head(n_keep)

    if top_n is not None:
        summary = summary.head(top_n)

    records = []

    for _, row in summary.iterrows():
        records.append(
            {
                "variable": row["feature"],
                "contribution": float(row["mean_signed_shap"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "mean_signed_shap": float(row["mean_signed_shap"]),
                "candidate_source": "model",
            }
        )

    return records

def _extract_shap_summary(shap_result: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """
    Return one row per model feature with:
    - feature
    - mean_abs_shap
    - mean_signed_shap
    """
    if shap_result is None or not isinstance(shap_result, dict):
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_signed_shap"])

    shap_df = shap_result.get("data_frame", pd.DataFrame())

    if shap_df is None or shap_df.empty:
        raw = shap_result.get("raw")
        if isinstance(raw, dict):
            shap_df = raw.get("data_frame", pd.DataFrame())

    if shap_df is None or shap_df.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_signed_shap"])

    df = shap_df.copy()

    if {"feature", "shap_value"}.issubset(df.columns):
        df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce")

        out = (
            df.dropna(subset=["feature", "shap_value"])
            .assign(abs_shap=lambda x: x["shap_value"].abs())
            .groupby("feature", as_index=False)
            .agg(
                mean_abs_shap=("abs_shap", "mean"),
                mean_signed_shap=("shap_value", "mean"),
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        return out

    return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_signed_shap"])


def _extract_top_model_feature_records(
    shap_result: Optional[Dict[str, Any]],
    top_n: int = 7,
) -> List[Dict[str, Any]]:
    summary = _extract_shap_summary(shap_result)

    if summary.empty:
        return []

    records = []

    for _, row in summary.head(top_n).iterrows():
        records.append(
            {
                "variable": row["feature"],
                "contribution": float(row["mean_signed_shap"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "mean_signed_shap": float(row["mean_signed_shap"]),
                "candidate_source": "model",
            }
        )

    return records


def _differences_by_variable(
    extreme_cluster_differences: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    if extreme_cluster_differences is None or extreme_cluster_differences.empty:
        return {}
    out = {}
    for _, row in extreme_cluster_differences.iterrows():
        out[str(row["variable"])] = {
            "baseline_mean": row.get("baseline_mean"),
            "target_mean": row.get("target_mean"),
            "delta": row.get("delta"),
        }
    return out

def _direction_hint_from_delta(delta: Optional[float]) -> str:
    if delta is None or pd.isna(delta):
        return "review"
    if delta > 0:
        return "reduce_or_optimize"
    if delta < 0:
        return "restore_or_increase"
    return "review"

def _make_action_record(
    variable: str,
    contribution: Optional[float],
    diff_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    delta = None if diff_info is None else diff_info.get("delta")
    direction_hint = _direction_hint_from_delta(delta)
    return {
        "variable": variable,
        "contribution": contribution,
        "direction_hint": direction_hint,
        "baseline_mean": None if diff_info is None else diff_info.get("baseline_mean"),
        "target_mean": None if diff_info is None else diff_info.get("target_mean"),
        "delta": delta,
    }

def _build_recommendation_lines_en(
    focus: Dict[str, Any],
    actions: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    component = focus.get("cost_component")
    grade = focus.get("grade")

    # ----------------------------
    # Header
    # ----------------------------
    if component and grade:
        lines.append(
            f"For {component} cost in grade {grade}, the following variables are the most relevant operational levers based on data and process knowledge:"
        )
    elif component:
        lines.append(
            f"For {component} cost, the following variables are the most relevant operational levers:"
        )
    else:
        lines.append(
            "The following variables are the most relevant operational levers:"
        )

    # ----------------------------
    # Per-variable recommendations
    # ----------------------------
    for i, a in enumerate(actions, start=1):
        variable = a["variable"]
        baseline_mean = a.get("baseline_mean")
        target_mean = a.get("target_mean")
        delta = a.get("delta")

        direction_hint = a.get("direction_hint")
        classification = (a.get("classification") or "").lower()
        confidence = a.get("confidence") or "unknown"
        engineering_reason = a.get("engineering_reason")

        # ----------------------------
        # Action label
        # ----------------------------
        if direction_hint == "review":
            header = f"{i}. **{variable}** — **[REVIEW BEFORE ACTION]**"
        else:
            if direction_hint == "reduce_or_optimize":
                action_txt = "reduce or optimize"
            elif direction_hint == "restore_or_increase":
                action_txt = "increase or restore"
            else:
                action_txt = "review"

            header = f"{i}. **{variable}** — recommended to {action_txt}"

        line = header

        # ----------------------------
        # Engineering reason (PRIMARY)
        # ----------------------------
        if engineering_reason:
            line += f". Engineering rationale: {engineering_reason}"

        # ----------------------------
        # Confidence
        # ----------------------------
        if confidence:
            line += f" (confidence: {confidence})"

        # ----------------------------
        # Supporting data evidence (SECONDARY)
        # ----------------------------
        if (
            baseline_mean is not None
            and target_mean is not None
            and delta is not None
        ):
            line += (
                f". Observed change between clusters: "
                f"{target_mean:.3f} vs {baseline_mean:.3f} (Δ={delta:.3f})"
            )

        # ----------------------------
        # Clarify review meaning
        # ----------------------------
        if direction_hint == "review":
            line += (
                ". This variable is actionable but requires engineering validation "
                "before applying a direct intervention."
            )

        lines.append(line)

    # ----------------------------
    # Footer
    # ----------------------------
    if actions:
        lines.append(
            "Direct actions can be used for what-if simulations. Variables marked as [REVIEW BEFORE ACTION] should be validated by process experts before defining intervention levels."
        )

    return lines

def _build_recommendation_lines_de(
    focus: Dict[str, Any],
    actions: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    component = focus.get("cost_component")
    grade = focus.get("grade")

    # ----------------------------
    # Header
    # ----------------------------
    if component and grade:
        lines.append(
            f"Für die Kostenkomponente {component} der Sorte {grade} sind die folgenden Variablen die wichtigsten operativen Stellhebel basierend auf Daten und Prozesswissen:"
        )
    elif component:
        lines.append(
            f"Für die Kostenkomponente {component} sind die folgenden Variablen die wichtigsten operativen Stellhebel:"
        )
    else:
        lines.append(
            "Die folgenden Variablen sind die wichtigsten operativen Stellhebel:"
        )

    # ----------------------------
    # Per-variable recommendations
    # ----------------------------
    for i, a in enumerate(actions, start=1):
        variable = a["variable"]
        baseline_mean = a.get("baseline_mean")
        target_mean = a.get("target_mean")
        delta = a.get("delta")

        direction_hint = a.get("direction_hint")
        classification = (a.get("classification") or "").lower()
        confidence = a.get("confidence") or "unbekannt"
        engineering_reason = a.get("engineering_reason")

        # ----------------------------
        # Action label
        # ----------------------------
        if direction_hint == "review":
            header = f"{i}. **{variable}** — **[VOR EINGRIFF PRÜFEN]**"
        else:
            if direction_hint == "reduce_or_optimize":
                action_txt = "reduzieren oder optimieren"
            elif direction_hint == "restore_or_increase":
                action_txt = "erhöhen oder wiederherstellen"
            else:
                action_txt = "überprüfen"

            header = f"{i}. **{variable}** — empfohlen: {action_txt}"

        line = header

        # ----------------------------
        # Engineering reason (PRIMARY)
        # ----------------------------
        if engineering_reason:
            line += f". Technische Begründung: {engineering_reason}"

        # ----------------------------
        # Confidence
        # ----------------------------
        if confidence:
            line += f" (Vertrauen: {confidence})"

        # ----------------------------
        # Supporting data evidence (SECONDARY)
        # ----------------------------
        if (
            baseline_mean is not None
            and target_mean is not None
            and delta is not None
        ):
            line += (
                f". Beobachtete Änderung zwischen Clustern: "
                f"{target_mean:.3f} vs {baseline_mean:.3f} (Δ={delta:.3f})"
            )

        # ----------------------------
        # Clarify review meaning
        # ----------------------------
        if direction_hint == "review":
            line += (
                ". Diese Variable ist grundsätzlich beeinflussbar, "
                "sollte jedoch vor einer direkten Maßnahme durch "
                "ingenieurtechnische Bewertung überprüft werden."
            )

        lines.append(line)

    # ----------------------------
    # Footer
    # ----------------------------
    if actions:
        lines.append(
            "Direkte Maßnahmen können für Was-wäre-wenn-Simulationen verwendet werden. "
            "Variablen mit [VOR EINGRIFF PRÜFEN] sollten vor der Festlegung konkreter Eingriffe "
            "durch Prozessexperten validiert werden."
        )

    return lines

def _records_from_cost_driver_variables(
    cost_driver_result: Dict[str, Any],
    shapley_contrib: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    Build recommendation candidate records from the same variables used by
    the cost-driver/knowledge step.

    This keeps RAG variables and final action variables aligned.
    """
    cost_driver_result = cost_driver_result or {}

    variables = cost_driver_result.get("top_driver_variables", []) or []

    if not variables:
        return _extract_top_driver_records(
            shapley_contrib=shapley_contrib,
            top_n=None,
        )

    contrib_map = {}

    if shapley_contrib is not None and not shapley_contrib.empty:
        df = shapley_contrib.copy()

        if {"variable", "contribution"}.issubset(df.columns):
            df["abs_contribution"] = pd.to_numeric(
                df["contribution"],
                errors="coerce",
            ).abs()

            df = df.sort_values("abs_contribution", ascending=False)

            for _, row in df.iterrows():
                v = row["variable"]
                if v not in contrib_map:
                    contrib_map[v] = row.get("contribution")

    records = []

    for v in variables:
        records.append(
            {
                "variable": v,
                "contribution": contrib_map.get(v),
                "candidate_source": "drivers",
            }
        )

    return records

def build_recommendations(
    cost_driver_result: Dict[str, Any],
    diagnosis_result: Optional[Dict[str, Any]] = None,
    shap_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    lang: str = "en",
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Build structured, evidence-based recommendations from cost-driver output,
    refined with SHAP evidence and engineering actionability from RAG.
    """
    lang = _pick_lang(lang)

    cost_driver_result = cost_driver_result or {}
    shap_result = shap_result or {}

    shapley_contrib = cost_driver_result.get("shapley_contrib", pd.DataFrame())
    extreme_cluster_differences = cost_driver_result.get("extreme_cluster_differences", pd.DataFrame())
    knowledge_text = _extract_knowledge_text(knowledge_result)

    focus = _extract_focus(
        diagnosis_result=diagnosis_result,
        cost_driver_result=cost_driver_result,
    )

    source = _recommendation_feature_source()

    effective_feature_source = source

    if _use_manual_actionable_inputs():
        target = focus.get("cost_component")
        top_driver_records = _records_from_manual_actionable_inputs(target)
        effective_feature_source = "manual"

    elif source == "model":
        top_driver_records = _extract_top_model_feature_records(
            shap_result=shap_result,
            top_n=None,
        )

    elif source == "drivers":
        top_driver_records = _records_from_cost_driver_variables(
            cost_driver_result=cost_driver_result,
            shapley_contrib=shapley_contrib,
        )

    else:
        # auto
        top_driver_records = _records_from_cost_driver_variables(
            cost_driver_result=cost_driver_result,
            shapley_contrib=shapley_contrib,
        )

        effective_feature_source = "drivers"

        if not top_driver_records:
            top_driver_records = _extract_top_model_feature_records(
                shap_result=shap_result,
                top_n=None,
            )

            effective_feature_source = "model"

    rag_n = _rag_limit_count(len(top_driver_records))
    top_driver_records = top_driver_records[:rag_n]

    #print("top_driver_records",top_driver_records)

    diff_map = _differences_by_variable(extreme_cluster_differences)

    # ----------------------------
    # SHAP summary map
    # ----------------------------
    shap_abs_map = {}
    shap_signed_map = {}

    if shap_result is not None and isinstance(shap_result, dict):
        shap_df = shap_result.get("data_frame", pd.DataFrame())

        if shap_df is not None and not shap_df.empty:
            shap_cols = [c for c in shap_df.columns if str(c).startswith("shap_")]

            for col in shap_cols:
                feat = col[len("shap_"):]
                series = pd.to_numeric(shap_df[col], errors="coerce").dropna()

                if not series.empty:
                    shap_abs_map[feat] = float(series.abs().mean())
                    shap_signed_map[feat] = float(series.mean())

    # ----------------------------
    # Knowledge / engineering classification
    # ----------------------------
    # print("knowledge_text",knowledge_text)
    actionability_map = _extract_actionability_map_from_json(knowledge_text)
    if not actionability_map:
        actionability_map = _extract_actionability_map(knowledge_text) # fallback

    if _use_manual_actionable_inputs():
        target = focus.get("cost_component")
        manual_inputs = _manual_actionable_inputs(target)
        model_features = _model_features_for_target(target)

        if model_features:
            manual_inputs = {
                v for v in manual_inputs
                if v in model_features
            }

        actionability_map = {
            v: {
                "classification": "actionable",
                "recommended_direction": "unknown",
                "confidence": "manual",
                "engineering_reason": (
                    "Variable included in manual actionable input list "
                    "and used by the selected prediction model."
                ),
            }
            for v in manual_inputs
        }

    # print("actionability_map",actionability_map)
    # TOREMOVE
    # print("\n--- CLASSIFICATION MAP DEBUG ---")
    # print("knowledge_text preview:")
    # print(str(knowledge_text)[:1500])

    # print("\nactionability_map type:", type(actionability_map))
    # print("actionability_map length:", len(actionability_map))
    # print("actionability_map keys:")
    # for k in actionability_map.keys():
    #     print(repr(k))

    # print("\naction variables:")
    # for r in top_driver_records:
    #     v = str(r.get("variable"))
    #     print(repr(v), "direct_match:", v in actionability_map)

    # print("--- END CLASSIFICATION MAP DEBUG ---\n")


    def _normalize_direction_from_knowledge(direction_text: Optional[str]) -> Optional[str]:
        if not direction_text:
            return None

        t = str(direction_text).strip().lower()

        if t == "decrease":
            return "reduce_or_optimize"
        if t == "increase":
            return "restore_or_increase"
        if t == "review":
            return "review"

        return None

    

    def _confidence_bucket(confidence_text: Optional[str]) -> str:
        t = (confidence_text or "").strip().lower()
        if t.startswith("high"):
            return "high"
        if t.startswith("medium"):
            return "medium"
        if t.startswith("low"):
            return "low"
        return "unknown"

    def _actionability_weight_local(classification: Optional[str], confidence: Optional[str]) -> float:
        cls = (classification or "").lower()
        conf = _confidence_bucket(confidence)

        # classification weight
        if "indicator" in cls and "actionable" not in cls:
            base = 0.0
        elif "indicator" in cls and "indirect" in cls:
            base = 0.20
        elif "indirect" in cls:
            base = 0.35
        elif "actionable" in cls:
            base = 1.00
        else:
            base = 0.25

        # confidence weight
        if conf == "high":
            conf_w = 1.00
        elif conf == "medium":
            conf_w = 0.65
        elif conf == "low":
            conf_w = 0.30
        else:
            conf_w = 0.50

        return base * conf_w

    def _fallback_priority_score(classification: Optional[str], confidence: Optional[str]) -> float:
        """
        Give RAG-validated variables a small but non-zero chance to survive
        even when analytics score is zero.
        """
        cls = (classification or "").lower()
        conf = _confidence_bucket(confidence)

        if "actionable" in cls and "indirect" not in cls:
            if conf.startswith("high"):
                return 0.30
            if conf.startswith("medium"):
                return 0.20
            if conf.startswith("low"):
                return 0.10
            return 0.15

        if "indirect" in cls:
            if conf.startswith("high"):
                return 0.12
            if conf.startswith("medium"):
                return 0.08
            if conf.startswith("low"):
                return 0.04
            return 0.05

        return 0.0

    def _suggest_intervention_from_action_with_knowledge(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        variable = action.get("variable")
        if not variable:
            return None

        direction_hint = action.get("direction_hint")
        classification = (action.get("classification") or "").lower()
        confidence = (action.get("confidence") or "").lower()
        reason = action.get("engineering_reason", "")

        # ----------------------------
        # Standard directional actions
        # ----------------------------
        if direction_hint == "reduce_or_optimize":
            return {
                "variable": variable,
                "mode": "relative",
                "value": -0.05,
                "note": "Recommended reduction based on data + knowledge"
            }

        if direction_hint == "restore_or_increase":
            return {
                "variable": variable,
                "mode": "relative",
                "value": +0.05,
                "note": "Recommended increase based on data + knowledge"
            }

        # ----------------------------
        # REVIEW case (your missing piece)
        # ----------------------------
        if (
            direction_hint == "review"
            and "actionable" in classification
            and confidence.startswith(("high", "medium"))
        ):
            return {
                "variable": variable,
                "mode": "review",
                "value": 0.0,
                "note": f"Review recommended before action. {reason}"
            }

        return None

    actions: List[Dict[str, Any]] = []

    for rec in top_driver_records:
        variable = str(rec.get("variable"))
        contribution = rec.get("contribution")

        action = _make_action_record(
            variable=variable,
            contribution=contribution,
            diff_info=diff_map.get(variable),
        )

        action["shap_mean_abs"] = rec.get(
            "mean_abs_shap",
            shap_abs_map.get(variable, 0.0),
        )

        action["shap_mean_signed"] = rec.get(
            "mean_signed_shap",
            shap_signed_map.get(variable, 0.0),
        )

        # SHAP enrichment
        shap_mean_abs = action["shap_mean_abs"]
        shap_mean_signed = action["shap_mean_signed"]

        contrib_abs = abs(contribution) if contribution is not None else 0.0
        analytics_priority = contrib_abs * shap_mean_abs

        # Knowledge enrichment
        ka = _get_actionability_for_variable(
            variable,
            actionability_map,
        )
        action["classification"] = ka.get("classification")
        action["engineering_reason"] = ka.get("engineering_reason")
        action["recommended_direction_from_knowledge"] = ka.get("recommended_direction")
        action["confidence"] = ka.get("confidence")

        # ----------------------------
        # Direction sources
        # ----------------------------
        rag_direction_hint = _normalize_direction_from_knowledge(
            action.get("recommended_direction_from_knowledge")
        )
        shap_direction_hint = _direction_from_shap_sign(action.get("shap_mean_signed"))
        delta_direction_hint = _direction_from_delta(action.get("delta"))

        conflict_info = _detect_direction_conflict(
            rag_direction=rag_direction_hint,
            shap_direction=shap_direction_hint
        )

        action["rag_direction_hint"] = rag_direction_hint
        action["shap_direction_hint"] = shap_direction_hint
        action["delta_direction_hint"] = delta_direction_hint
        action["has_direction_conflict"] = conflict_info["has_conflict"]

        # Priority of direction:
        # 1) RAG
        # 2) SHAP
        # 3) delta-based
        if conflict_info["has_conflict"]:
            action["direction_hint"] = "review"
        else:
            action["direction_hint"] = (
                rag_direction_hint
                or shap_direction_hint
                or action.get("direction_hint")
            )

        action_weight = _actionability_weight_local(
            classification=action.get("classification"),
            confidence=action.get("confidence"),
        )
        action["actionability_weight"] = action_weight

        # If analytics score is zero/near-zero, allow strong RAG support to promote the variable
        fallback_priority = _fallback_priority_score(
            classification=action.get("classification"),
            confidence=action.get("confidence"),
        )
        action["priority_score"] = max(analytics_priority, fallback_priority)

        enable_penalty_conflict = False
        if enable_penalty_conflict:
            conflict_penalty = 0.35 if action.get("has_direction_conflict") else 1.0
            action["conflict_penalty"] = conflict_penalty
            action["priority_score_final"] = action["priority_score"] * action_weight * conflict_penalty
        else:
            action["conflict_penalty"] = 1.0
            action["priority_score_final"] = action["priority_score"] * action_weight

        # Suggested intervention must be created AFTER knowledge override
        action["suggested_intervention"] = _suggest_intervention_from_action_with_knowledge(action)

        actions.append(action)

    #TOREMOVE
    # print("\n" + "=" * 100)
    # print("RECOMMENDATION DEBUG: ACTIONS CREATED BEFORE FILTERING")
    # print("=" * 100)

    # for a in actions:
    #     print(
    #         a.get("variable"),
    #         "| classification:", a.get("classification"),
    #         "| confidence:", a.get("confidence"),
    #         "| recommendable:", _is_recommendable(a.get("classification")),
    #         "| priority_score_final:", a.get("priority_score_final"),
    #         "| suggested_intervention:", a.get("suggested_intervention") is not None,
    #     )

    # print("=" * 100 + "\n")

    # ----------------------------
    # Filter / rerank using knowledge
    # ----------------------------
    if knowledge_text:
        recommendable = [a for a in actions if _is_recommendable(a.get("classification"))]
        if recommendable:
            actions = recommendable

    actions = sorted(
        actions,
        key=lambda a: (
            a.get("priority_score_final", 0.0),
            a.get("priority_score", 0.0),
            abs(a.get("contribution", 0.0)) if a.get("contribution") is not None else 0.0,
        ),
        reverse=True,
    )

    action_limit = _action_limit_count(len(actions))
    actions = actions[:action_limit]

    suggested_interventions = [
        a["suggested_intervention"]
        for a in actions
        if a.get("suggested_intervention") is not None
    ]

    # ----------------------------
    # Build text
    # ----------------------------
    if lang == "de":
        text_lines = _build_recommendation_lines_de(focus, actions)
        header = "Empfehlungen"
    else:
        text_lines = _build_recommendation_lines_en(focus, actions)
        header = "Recommendations"

    if shap_result is not None:
        if lang == "de":
            text_lines.append(
                "Die Priorisierung dieser Empfehlungen wurde zusätzlich mit SHAP-Werten aus dem Kostenmodell abgestützt."
            )
        else:
            text_lines.append(
                "These recommendations were additionally prioritized using SHAP values from the cost model."
            )

    if knowledge_text:
        if lang == "de":
            text_lines.append(
                "Zusätzliche fachliche Hinweise aus dem Wissenskontext wurden berücksichtigt, um die Empfehlungen einzuordnen und nicht direkt beeinflussbare Variablen zurückzustellen."
            )
        else:
            text_lines.append(
                "Additional domain guidance from the knowledge context was used to prioritize actionable levers and deprioritize non-actionable indicators."
            )

    if any(a.get("has_direction_conflict") for a in actions):
        if lang == "de":
            text_lines.append(
                "Bei einigen Variablen wurden widersprüchliche Signale zwischen Modell, beobachtetem Trend und Wissenskontext erkannt; diese wurden daher als Prüfpunkt statt als direkte Empfehlung behandelt."
            )
        else:
            text_lines.append(
                "Some variables showed conflicting signals between model sensitivity, observed change, and engineering knowledge; these were therefore treated as review items rather than direct recommendations."
            )
    
    text = header + "\n\n" + "\n".join(f"- {line}" for line in text_lines)

    return {
        "text": text,
        "focus": focus,
        "actions": actions,
        "suggested_interventions": suggested_interventions,
        "knowledge_text": knowledge_text,
        "knowledge_result": knowledge_result,
        "diagnosis_result": diagnosis_result,
        "shap_result": shap_result,
        "cost_driver_result": cost_driver_result,
        "recommendation_feature_source": source,        
    }


def build_knowledge_query_from_drivers(
    cost_driver_result: Dict[str, Any],
    shap_result: Optional[Dict[str, Any]] = None,
) -> str:
    import pandas as pd
    import json

    cost_driver_result = cost_driver_result or {}
    shap_result = shap_result or {}

    component = cost_driver_result.get("cost_component")
    grade = cost_driver_result.get("grade")

    source = _get_recommendation_feature_source()

    if _use_manual_actionable_inputs():
        return ""

    variables = cost_driver_result.get("top_driver_variables", [])
    diff_df = cost_driver_result.get(
        "extreme_cluster_differences",
        pd.DataFrame(),
    )

    candidate_rows = []

    # ------------------------------------------------------------
    # Candidate variables from model SHAP
    # ------------------------------------------------------------
    if source == "model":
        model_records = _extract_top_model_feature_records(
            shap_result=shap_result,
            top_n=None,
        )

        for rec in model_records:
            candidate_rows.append(
                {
                    "variable": rec["variable"],
                    "mean_abs_shap": rec.get("mean_abs_shap"),
                    "mean_signed_shap": rec.get("mean_signed_shap"),
                    "candidate_source": "model",
                }
            )

    # ------------------------------------------------------------
    # Candidate variables from cost drivers
    # ------------------------------------------------------------
    else:
        if diff_df is not None and not diff_df.empty:
            diff_df = diff_df.copy()

            if variables:
                diff_df = diff_df[
                    diff_df["variable"].isin(variables)
                ].copy()

            for _, row in diff_df.iterrows():
                candidate_rows.append(
                    {
                        "variable": row["variable"],
                        "baseline_mean": (
                            None
                            if pd.isna(row.get("baseline_mean"))
                            else float(row.get("baseline_mean"))
                        ),
                        "target_mean": (
                            None
                            if pd.isna(row.get("target_mean"))
                            else float(row.get("target_mean"))
                        ),
                        "delta": (
                            None
                            if pd.isna(row.get("delta"))
                            else float(row.get("delta"))
                        ),
                        "candidate_source": "drivers",
                    }
                )

        if not candidate_rows and variables:
            candidate_rows = [
                {
                    "variable": v,
                    "candidate_source": "drivers",
                }
                for v in variables
            ]

        # auto fallback to model SHAP if no driver candidates exist
        if source == "auto" and not candidate_rows:
            model_records = _extract_top_model_feature_records(
                shap_result=shap_result,
                top_n=None,
            )

            for rec in model_records:
                candidate_rows.append(
                    {
                        "variable": rec["variable"],
                        "mean_abs_shap": rec.get("mean_abs_shap"),
                        "mean_signed_shap": rec.get("mean_signed_shap"),
                        "candidate_source": "model",
                    }
                )

    # ------------------------------------------------------------
    # Enrich candidates with SHAP values when available
    # ------------------------------------------------------------
    shap_map = {}

    shap_df = None
    if isinstance(shap_result, dict):
        shap_df = shap_result.get("data_frame", pd.DataFrame())

        if (
            shap_df is None
            or shap_df.empty
        ) and isinstance(shap_result.get("raw"), dict):
            shap_df = shap_result["raw"].get("data_frame", pd.DataFrame())

    if shap_df is not None and not shap_df.empty:
        if {"feature", "shap_value"}.issubset(shap_df.columns):
            shap_summary = (
                shap_df.copy()
                .assign(
                    shap_value=lambda x: pd.to_numeric(
                        x["shap_value"],
                        errors="coerce",
                    )
                )
                .dropna(subset=["feature", "shap_value"])
                .assign(abs_shap=lambda x: x["shap_value"].abs())
                .groupby("feature", as_index=False)
                .agg(
                    mean_signed_shap=("shap_value", "mean"),
                    mean_abs_shap=("abs_shap", "mean"),
                )
            )

            for _, row in shap_summary.iterrows():
                shap_map[row["feature"]] = {
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                    "mean_signed_shap": float(row["mean_signed_shap"]),
                }

    enriched_candidates = []

    for row in candidate_rows:
        variable = row["variable"]
        shap_info = shap_map.get(variable)

        out_row = dict(row)

        if shap_info is not None:
            out_row["mean_abs_shap"] = shap_info["mean_abs_shap"]
            out_row["mean_signed_shap"] = shap_info["mean_signed_shap"]

        enriched_candidates.append(out_row)

    # ------------------------------------------------------------
    # Final hard limit before sending variables to RAG
    # ------------------------------------------------------------
    rag_n = _rag_limit_count(len(enriched_candidates))
    enriched_candidates = enriched_candidates[:rag_n]

    candidates_json = json.dumps(enriched_candidates, indent=2)

    query = f"""
You are acting as an experienced papermaking process engineer.

Task:
Assess which candidate variables are sensible recommendation targets to improve {component} cost for grade {grade}.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside JSON.

Return a JSON object with this exact schema:
{{
  "variables": [
    {{
      "variable": "string",
      "classification": "actionable|indirectly actionable|indicator|unknown",
      "recommended_direction": "increase|decrease|review|unknown",
      "confidence": "high|medium|low|unknown",
      "engineering_reason": "short string"
    }}
  ]
}}

Rules:
- "actionable" = direct operational lever
- "indirectly actionable" = can be influenced, but constrained or secondary
- "indicator" = not a recommended direct manipulation target
- "recommended_direction":
  - "increase" if the variable should be increased
  - "decrease" if it should be reduced
  - "review" if it should not be directly moved without further investigation
  - "unknown" if direction is unclear
- Keep engineering_reason short and practical.
- Evaluate only the variables listed below.

Candidate variables:
{candidates_json}
""".strip()

    return query


def _extract_knowledge_text(knowledge_result: Optional[Dict[str, Any]]) -> str:
    if knowledge_result is None:
        return ""
    if isinstance(knowledge_result, dict):
        if "answer" in knowledge_result and knowledge_result["answer"]:
            return str(knowledge_result["answer"])
        if "text" in knowledge_result and knowledge_result["text"]:
            return str(knowledge_result["text"])
    return str(knowledge_result) if knowledge_result else ""

def _suggest_intervention_from_action(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    variable = action.get("variable")
    direction_hint = action.get("direction_hint")
    if not variable:
        return None
    if direction_hint == "reduce_or_optimize":
        return {
            "variable": variable,
            "mode": "relative",
            "value": -0.05,
        }
    if direction_hint == "restore_or_increase":
        return {
            "variable": variable,
            "mode": "relative",
            "value": 0.05,
        }
    return None



def _actionability_weight(classification: str, confidence: str) -> float:
    """
    Convert engineering classification into a ranking weight.
    """
    classification = (classification or "").lower()
    confidence = (confidence or "").lower()

    if "actionable" in classification and "indirect" not in classification:
        base = 1.0
    elif "indirect" in classification:
        base = 0.5
    elif "indicator" in classification:
        base = 0.0
    else:
        base = 0.3

    if confidence == "high":
        conf = 1.0
    elif confidence == "medium":
        conf = 0.75
    elif confidence == "low":
        conf = 0.5
    else:
        conf = 0.6

    return base * conf


def _is_recommendable(classification: str) -> bool:
    classification = (classification or "").lower()
    if "indicator" in classification:
        return False
    if "actionable" in classification:
        return True
    if "indirect" in classification:
        return True
    return False

def filter_actionable_variables(drivers, actionability_map):
    actionable = []
    secondary = []

    for v in drivers:
        info = actionability_map.get(v, {})
        cls = info.get("classification", "")

        if "actionable" in cls:
            actionable.append(v)
        elif "indirect" in cls:
            secondary.append(v)

    return actionable, secondary


def _clean_rag_value(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    # remove common markdown bullets / emphasis noise
    x = x.lstrip("-").strip()
    x = x.replace("**", "").strip()
    x = x.strip(" :")
    return x


def _extract_after_colon(line: str) -> str:
    if ":" not in line:
        return ""
    return _clean_rag_value(line.split(":", 1)[1])

def _extract_actionability_map(knowledge_text: str) -> Dict[str, Dict[str, Any]]:
    import json

    if not knowledge_text:
        return {}

    try:
        obj = json.loads(knowledge_text)
    except Exception:
        return {}

    rows = obj.get("variables", [])
    out = {}

    for row in rows:
        variable = str(row.get("variable", "")).strip()
        if not variable:
            continue

        out[variable] = {
            "classification": str(row.get("classification", "")).strip().lower(),
            "engineering_reason": str(row.get("engineering_reason", "")).strip(),
            "recommended_direction": str(row.get("recommended_direction", "")).strip().lower(),
            "confidence": str(row.get("confidence", "")).strip().lower(),
        }

    return out


def _direction_from_shap_sign(mean_signed_shap: Optional[float]) -> Optional[str]:
    if mean_signed_shap is None:
        return None
    try:
        v = float(mean_signed_shap)
    except Exception:
        return None

    if v > 0:
        return "reduce_or_optimize"
    if v < 0:
        return "restore_or_increase"
    return None


def _direction_from_delta(delta: Optional[float]) -> Optional[str]:
    if delta is None:
        return None
    try:
        v = float(delta)
    except Exception:
        return None

    if v > 0:
        return "reduce_or_optimize"
    if v < 0:
        return "restore_or_increase"
    return None


def _detect_direction_conflict(
    rag_direction: Optional[str],
    shap_direction: Optional[str],
) -> Dict[str, Any]:
    directions = {
        "rag": rag_direction,
        "shap": shap_direction,
    }

    non_null = {k: v for k, v in directions.items() if v is not None and v != "review"}
    unique_dirs = sorted(set(non_null.values()))
    has_conflict = len(unique_dirs) > 1

    return {
        "rag_direction": rag_direction,
        "shap_direction": shap_direction,
        "has_conflict": has_conflict,
        "resolved_direction": None if has_conflict else (unique_dirs[0] if unique_dirs else None),
    }

def _strip_json_code_fence(text: str) -> str:
    if text is None:
        return ""

    s = str(text).strip()

    if s.startswith("```json"):
        s = s[len("```json"):].strip()
    elif s.startswith("```"):
        s = s[len("```"):].strip()

    if s.endswith("```"):
        s = s[:-3].strip()

    return s

def _extract_actionability_map_from_json(text: str):
    import re
    import json

    #print("knowledge_text length:", len(str(text)))
    #print(str(text)[:3000])

    if not text:
        return {}

    s = str(text)

    # Remove markdown fences but keep content
    s = re.sub(r"```json", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```", "", s)
    s = s.strip()

    # ------------------------------------------------------------
    # 1) Try strict JSON first
    # ------------------------------------------------------------
    try:
        first = s.find("{")
        last = s.rfind("}")

        if first >= 0 and last > first:
            payload = json.loads(s[first:last + 1])
            variables = payload.get("variables", [])

            out = {}
            for item in variables:
                if not isinstance(item, dict):
                    continue

                variable = item.get("variable")
                if not variable:
                    continue

                out[str(variable)] = {
                    "classification": item.get("classification"),
                    "recommended_direction": item.get("recommended_direction"),
                    "confidence": item.get("confidence"),
                    "engineering_reason": item.get("engineering_reason"),
                }

            if out:
                return out

    except Exception as e:
        print("Strict JSON parsing failed:", e)

    # ------------------------------------------------------------
    # 2) Very tolerant fallback:
    # parse every object-like block containing "variable"
    # ------------------------------------------------------------
    out = {}

    object_blocks = re.findall(
        r"\{[^{}]*?\"variable\"[^{}]*?\}",
        s,
        flags=re.DOTALL,
    )

    print("Fallback object blocks found:", len(object_blocks))

    def _field(block: str, name: str):
        m = re.search(
            rf'"{re.escape(name)}"\s*:\s*"([^"]*)"',
            block,
            flags=re.DOTALL,
        )
        return m.group(1).strip() if m else None

    for block in object_blocks:
        variable = _field(block, "variable")
        if not variable:
            continue

        out[str(variable)] = {
            "classification": _field(block, "classification"),
            "recommended_direction": _field(block, "recommended_direction"),
            "confidence": _field(block, "confidence"),
            "engineering_reason": _field(block, "engineering_reason"),
        }

    print("Fallback parsed variables:", list(out.keys()))

    return out

def _extract_actionability_map_from_jsonDEPRECATED(knowledge_text: str):
    import json

    if not knowledge_text:
        return {}

    cleaned = _strip_json_code_fence(knowledge_text)

    try:
        obj = json.loads(cleaned)
    except Exception:
        return {}

    rows = obj.get("variables", [])
    out = {}

    for row in rows:
        variable = str(row.get("variable", "")).strip()
        if not variable:
            continue

        out[variable] = {
            "classification": str(row.get("classification", "")).strip().lower(),
            "engineering_reason": str(row.get("engineering_reason", "")).strip(),
            "recommended_direction": str(row.get("recommended_direction", "")).strip().lower(),
            "confidence": str(row.get("confidence", "")).strip().lower(),
        }

    return out

def _norm_var_name(v: str) -> str:
    return (
        str(v)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _get_actionability_for_variable(
    variable: str,
    actionability_map: Dict[str, Any],
) -> Dict[str, Any]:
    if variable in actionability_map:
        return actionability_map[variable]

    nv = _norm_var_name(variable)

    for k, value in actionability_map.items():
        if _norm_var_name(k) == nv:
            return value

    return {}

def build_optimized_interventions_from_recommendation(
    recommend_result,
    cost_component,
    grade=None,
    reel_id=None,
    timestamp=None,
    target_range=None,
    baseline_range=None,
    quality_constraints=None,
    objective_mode=None,
):
    import pandas as pd
    import scenario_tools as st
    import recommendation_config as rc
    from recommendation_optimizer import optimize_cost_over_actionable_variables

    actions = recommend_result.get("actions", [])

    actionable_variables = [
        a.get("variable")
        for a in actions
        if a.get("classification") in {
            "actionable",
            "indirectly actionable",
        }
    ]

    actionable_variables = [
        v for v in actionable_variables
        if v is not None
    ]

    if not actionable_variables:
        return [], {
            "success": False,
            "message": "No actionable variables available for optimization.",
        }

    cost_function, _, _ = st._resolve_cost_component(cost_component)

    ref = st.get_reference_turnup(
        reel_id=reel_id,
        timestamp=timestamp,
        grade=grade,
        target_range=target_range,
    )

    reference_row = ref["row"]

    historical_df = st.load_turnup_data_for_scenario(
        target_range=baseline_range,
    )

    if grade is not None and "AB_Grade_ID" in historical_df.columns:
        historical_df = historical_df[
            historical_df["AB_Grade_ID"].astype(str) == str(grade)
        ].copy()

    opt = optimize_cost_over_actionable_variables(
        reference_row=reference_row,
        historical_df=historical_df,
        actionable_variables=actionable_variables,
        cost_function=cost_function,
        lower_q=getattr(rc, "RECOMMENDATION_OPTIMIZER_LOWER_Q", 0.05),
        upper_q=getattr(rc, "RECOMMENDATION_OPTIMIZER_UPPER_Q", 0.95),
        joint_quantile=getattr(rc, "RECOMMENDATION_OPTIMIZER_JOINT_QUANTILE", 0.95),
        quality_constraints=quality_constraints,
        objective_mode=objective_mode or "minimize",
        invariants=getattr(rc,"RECOMMENDATION_INVARIANTS",None),
    )

    opt["reference"] = ref.get("reference")
    opt["reference_warnings"] = ref.get("warnings", [])

    if not opt.get("success") and not opt.get("changes"):
        return [], opt

    interventions = []

    for c in opt.get("changes", []):
        interventions.append(
            {
                "variable": c["variable"],
                "mode": "absolute",
                "value": c["optimized_value"],
                "current_value": c["current_value"],
                "delta": c["delta"],
                "lower_bound": c["lower_bound"],
                "upper_bound": c["upper_bound"],
                "note": "Optimized intervention from cost-function minimization constrained by historical feasibility.",
            }
        )

    return interventions, opt