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
    if component and grade:
        lines.append(
            f"For {component} cost in grade {grade}, the first priority is to review the variables with the strongest contribution to the deterioration."
        )
    elif component:
        lines.append(
            f"For {component} cost, the first priority is to review the variables with the strongest contribution to the deterioration."
        )
    else:
        lines.append(
            "The first priority is to review the variables with the strongest contribution to the deterioration."
        )
    for i, a in enumerate(actions, start=1):
        variable = a["variable"]
        baseline_mean = a.get("baseline_mean")
        target_mean = a.get("target_mean")
        delta = a.get("delta")
        direction_hint = a.get("direction_hint")
        if baseline_mean is not None and target_mean is not None and delta is not None:
            if direction_hint == "reduce_or_optimize":
                line = (
                    f"{i}. Review **{variable}** first. In the worse target extreme cluster, "
                    f"its average value was higher than in the baseline extreme cluster "
                    f"({target_mean:.3f} vs {baseline_mean:.3f}; Δ={delta:.3f}). "
                    f"This suggests that reducing or stabilizing this variable may help."
                )
            elif direction_hint == "restore_or_increase":
                line = (
                    f"{i}. Review **{variable}** first. In the worse target extreme cluster, "
                    f"its average value was lower than in the baseline extreme cluster "
                    f"({target_mean:.3f} vs {baseline_mean:.3f}; Δ={delta:.3f}). "
                    f"This suggests that restoring or increasing this variable may help."
                )
            else:
                line = (
                    f"{i}. Review **{variable}** first. It shows a meaningful difference between "
                    f"the extreme baseline and target clusters "
                    f"({target_mean:.3f} vs {baseline_mean:.3f}; Δ={delta:.3f})."
                )
        else:
            line = (
                f"{i}. Review **{variable}** first, since it is one of the strongest contributors "
                f"to the observed deterioration."
            )
        lines.append(line)
    if actions:
        lines.append(
            "These variables are good candidates for targeted what-if simulation once an intervention magnitude has been defined."
        )
    return lines

def _build_recommendation_lines_de(
    focus: Dict[str, Any],
    actions: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    component = focus.get("cost_component")
    grade = focus.get("grade")
    if component and grade:
        lines.append(
            f"Für die Komponente {component} der Sorte {grade} sollten zuerst die Variablen mit dem stärksten Beitrag zur Verschlechterung überprüft werden."
        )
    elif component:
        lines.append(
            f"Für die Komponente {component} sollten zuerst die Variablen mit dem stärksten Beitrag zur Verschlechterung überprüft werden."
        )
    else:
        lines.append(
            "Zuerst sollten die Variablen mit dem stärksten Beitrag zur Verschlechterung überprüft werden."
        )
    for i, a in enumerate(actions, start=1):
        variable = a["variable"]
        baseline_mean = a.get("baseline_mean")
        target_mean = a.get("target_mean")
        delta = a.get("delta")
        direction_hint = a.get("direction_hint")
        if baseline_mean is not None and target_mean is not None and delta is not None:
            if direction_hint == "reduce_or_optimize":
                line = (
                    f"{i}. Prüfen Sie zuerst **{variable}**. Im schlechteren Ziel-Extremcluster "
                    f"lag der Mittelwert höher als im Basis-Extremcluster "
                    f"({target_mean:.3f} vs {baseline_mean:.3f}; Δ={delta:.3f}). "
                    f"Das deutet darauf hin, dass eine Reduzierung oder Stabilisierung hilfreich sein könnte."
                )
            elif direction_hint == "restore_or_increase":
                line = (
                    f"{i}. Prüfen Sie zuerst **{variable}**. Im schlechteren Ziel-Extremcluster "
                    f"lag der Mittelwert niedriger als im Basis-Extremcluster "
                    f"({target_mean:.3f} vs {baseline_mean:.3f}; Δ={delta:.3f}). "
                    f"Das deutet darauf hin, dass eine Wiederherstellung oder Erhöhung hilfreich sein könnte."
                )
            else:
                line = (
                    f"{i}. Prüfen Sie zuerst **{variable}**. Zwischen Basis- und Ziel-Extremcluster "
                    f"zeigt sich ein relevanter Unterschied "
                    f"({target_mean:.3f} vs {baseline_mean:.3f}; Δ={delta:.3f})."
                )
        else:
            line = (
                f"{i}. Prüfen Sie zuerst **{variable}**, da diese Variable zu den stärksten "
                f"Treibern der beobachteten Verschlechterung gehört."
            )
        lines.append(line)
    if actions:
        lines.append(
            "Diese Variablen sind gute Kandidaten für gezielte Was-wäre-wenn-Simulationen, sobald eine Eingriffsgröße definiert wurde."
        )
    return lines

def build_recommendations(
    cost_driver_result: Dict[str, Any],
    diagnosis_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    lang: str = "en",
    top_n: int = 3,
) -> Dict[str, Any]:
    """
    Build structured, evidence-based recommendations from cost-driver output.
    Parameters
    ----------
    cost_driver_result : dict
        Output of answer_cost_driver_analysis()["raw"] or run_cost_driver_analysis(...)
    diagnosis_result : optional
        Reserved for future use
    knowledge_result : optional
        Reserved for future use
    lang : str
        "en" or "de"
    top_n : int
        Number of top driver variables to convert into recommendations
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
        top_n=top_n,
    )
    diff_map = _differences_by_variable(extreme_cluster_differences)
    actions: List[Dict[str, Any]] = []
    for rec in top_driver_records:
        variable = str(rec.get("variable"))
        contribution = rec.get("contribution")
        action = _make_action_record(
            variable=variable,
            contribution=contribution,
            diff_info=diff_map.get(variable),
        )
        actions.append(action)
    if lang == "de":
        text_lines = _build_recommendation_lines_de(focus, actions)
        header = "Empfehlungen"
    else:
        text_lines = _build_recommendation_lines_en(focus, actions)
        header = "Recommendations"
    if knowledge_text:
        if lang == "de":
            text_lines.append(
                "Zusätzliche fachliche Hinweise aus dem Wissenskontext wurden berücksichtigt, um die Empfehlungen einzuordnen."
            )
        else:
            text_lines.append(
                "Additional domain guidance from the knowledge context was considered when interpreting these recommendations."
            )
    text = header + "\n\n" + "\n".join(f"- {line}" for line in text_lines)
    return {
        "text": text,
        "focus": focus,
        "actions": actions,
        "knowledge_text": knowledge_text,
        "knowledge_result": knowledge_result,
        "diagnosis_result": diagnosis_result,
        "cost_driver_result": cost_driver_result,
    }

def build_knowledge_query_from_drivers(cost_driver_result: Dict[str, Any]) -> str:
    component = cost_driver_result.get("cost_component")
    grade = cost_driver_result.get("grade")
    variables = cost_driver_result.get("top_driver_variables", [])
    diff_df = cost_driver_result.get("extreme_cluster_differences", pd.DataFrame())
    diff_lines = []
    if diff_df is not None and not diff_df.empty:
        for _, row in diff_df.iterrows():
            diff_lines.append(
                f"{row['variable']}: baseline_mean={row['baseline_mean']:.3f}, "
                f"target_mean={row['target_mean']:.3f}, delta={row['delta']:.3f}"
            )
    vars_txt = ", ".join(variables) if variables else "unknown drivers"
    diff_txt = "; ".join(diff_lines)
    return (
        f"Papermaking recommendations to improve {component} cost "
        f"for grade {grade}. Main driver variables: {vars_txt}. "
        f"Observed differences between extreme clusters: {diff_txt}. "
        f"Provide operational interpretation and practical levers."
    )

def _extract_knowledge_text(knowledge_result: Optional[Dict[str, Any]]) -> str:
    if knowledge_result is None:
        return ""
    if isinstance(knowledge_result, dict):
        if "answer" in knowledge_result and knowledge_result["answer"]:
            return str(knowledge_result["answer"])
        if "text" in knowledge_result and knowledge_result["text"]:
            return str(knowledge_result["text"])
    return str(knowledge_result) if knowledge_result else ""