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
    cost_driver_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "grade": cost_driver_result.get("grade"),
        "cost_component": cost_driver_result.get("cost_component"),
        "target_range": cost_driver_result.get("target_range"),
        "baseline_range": cost_driver_result.get("baseline_range"),
    }

def _extract_top_driver_records(
    shapley_contrib: pd.DataFrame,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    if shapley_contrib is None or shapley_contrib.empty:
        return []
    df = shapley_contrib.copy()
    if "variable" not in df.columns or "contribution" not in df.columns:
        return []
    df = df.sort_values("contribution", ascending=False).head(top_n)
    return df.to_dict(orient="records")

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

    shapley_contrib = cost_driver_result.get("shapley_contrib", pd.DataFrame())
    extreme_cluster_differences = cost_driver_result.get("extreme_cluster_differences", pd.DataFrame())
    knowledge_text = _extract_knowledge_text(knowledge_result)

    focus = _extract_focus(
        diagnosis_result=diagnosis_result,
        cost_driver_result=cost_driver_result,
    )

    top_driver_records = _extract_top_driver_records(
        shapley_contrib=shapley_contrib,
        top_n=max(top_n, 7), # broader pool before RAG filtering
    )

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
    actionability_map = _extract_actionability_map_from_json(knowledge_text)
    if not actionability_map:
        actionability_map = _extract_actionability_map(knowledge_text) # fallback

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

        # SHAP enrichment
        shap_mean_abs = shap_abs_map.get(variable, 0.0)
        shap_mean_signed = shap_signed_map.get(variable, 0.0)

        action["shap_mean_abs"] = shap_mean_abs
        action["shap_mean_signed"] = shap_mean_signed

        contrib_abs = abs(contribution) if contribution is not None else 0.0
        analytics_priority = contrib_abs * shap_mean_abs

        # Knowledge enrichment
        ka = actionability_map.get(variable, {})
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

    actions = actions[:top_n]

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
    }

def build_knowledge_query_from_drivers(
    cost_driver_result: Dict[str, Any],
    shap_result: Optional[Dict[str, Any]] = None,
) -> str:
    import pandas as pd
    import json

    component = cost_driver_result.get("cost_component")
    grade = cost_driver_result.get("grade")
    variables = cost_driver_result.get("top_driver_variables", [])
    diff_df = cost_driver_result.get("extreme_cluster_differences", pd.DataFrame())

    candidate_rows = []
    if diff_df is not None and not diff_df.empty:
        diff_df = diff_df.copy()

        if variables:
            diff_df = diff_df[diff_df["variable"].isin(variables)].copy()

        for _, row in diff_df.iterrows():
            candidate_rows.append(
                {
                    "variable": row["variable"],
                    "baseline_mean": None if pd.isna(row["baseline_mean"]) else float(row["baseline_mean"]),
                    "target_mean": None if pd.isna(row["target_mean"]) else float(row["target_mean"]),
                    "delta": None if pd.isna(row["delta"]) else float(row["delta"]),
                }
            )

    if not candidate_rows and variables:
        candidate_rows = [{"variable": v} for v in variables]

    shap_map = {}

    if shap_result is not None and isinstance(shap_result, dict):
        shap_df = shap_result.get("data_frame", pd.DataFrame())

        if shap_df is not None and not shap_df.empty:
            shap_summary = (
                shap_df.assign(abs_shap=lambda x: x["shap_value"].abs())
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
        shap_info = shap_map.get(variable, None)

        out_row = dict(row)
        if shap_info is not None:
            out_row["mean_abs_shap"] = shap_info["mean_abs_shap"]
            out_row["mean_signed_shap"] = shap_info["mean_signed_shap"]
        enriched_candidates.append(out_row)

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

def _extract_actionability_map_from_json(knowledge_text: str):
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