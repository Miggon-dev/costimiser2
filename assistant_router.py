"""
assistant_router.py
 
Router for:
1) knowledge retrieval
2) process data query
3) SHAP explanations
"""
 
from typing import Dict, Any
 
import knowledge_retrieval as kr
import process_data_tools as pdt
import shap_tools as st
import cost_driver_tools as cdt
import query_parser as qp
import prediction_tools as pt
 
 
# -------------------------------------------------
# Query classification
# -------------------------------------------------
 
def classify_query(query: str) -> str:
    q = query.lower()

    cost_driver_keywords = [
        "driver",
        "drivers",
        "what changed",
        "why did cost",
        "increase",
        "decrease",
        "variation",
    ]
 
    shap_keywords = [
        "shap",
        "shapley",
        "explain model",
        "feature importance",
        "drivers",
        "contribution",
        "contributors",
    ]
 
    data_keywords = [
        "data",
        "trend",
        "show",
        "compare",
        "last",
        "yesterday",
        "today",
        "week",
        "month",
        "speed",
        "electricity",
        "cost",
        "steam",
    ]
 
    for k in shap_keywords:
        if k in q:
            return "shap"
 
    for k in data_keywords:
        if k in q:
            return "process_data"
    
    for k in cost_driver_keywords:
        if k in q:
            return "cost_driver"
 
    return "knowledge"
 
 
# -------------------------------------------------
# Knowledge workflow
# -------------------------------------------------
 
def answer_knowledge(query: str) -> Dict[str, Any]:
    out = kr.ask(query)
 
    return {
        "type": "knowledge",
        "answer": out["answer"],
        "sources": out["sources"],
    }
 
 
# -------------------------------------------------
# Process data workflow
# -------------------------------------------------
 
def answer_process_data(
    target_range=None,
    grade: str = None,
    lang: str = "en",
) -> Dict[str, Any]:
 
    import time_context as tc
 
    resolved = tc.resolve_single_period_range(target_range=target_range)
 
    target_range = resolved["target_range"]
    used_default = resolved["used_default"]
 
    df = pdt.get_feature_snapshot(
        grade=grade,
        target_range=target_range,
    )
 
    if df.empty:
        raise ValueError(f"No data found for grade {grade} in range {target_range}")
 
    stats = df.describe().T
 
    msg = tc.build_single_period_message(
        target_range=target_range,
        used_default=used_default,
        lang=lang,
    )
 
    return {
        "type": "process_data",
        "grade": grade,
        "target_range": target_range,
        "used_default_range": used_default,
        "n_rows": len(df),
        "data": df,
        "stats": stats,
        "message": msg,
    }
 
 
# -------------------------------------------------
# SHAP workflow
# -------------------------------------------------
 
def answer_shap(
    component: str,
    grade_id: str,
    target_range=None,
    baseline_range=None,
    lang: str = "en",
) -> Dict[str, Any]:
    import time_context as tc
 
    resolved = tc.resolve_ranges(
        target_range=target_range,
        baseline_range=baseline_range,
    )
 
    target_range = resolved["target_range"]
    baseline_range = resolved["baseline_range"]
    used_any_default = resolved["used_any_default"]
 
    X_sample = pdt.get_feature_snapshot(
        grade=grade_id,
        target_range=target_range,
    )
 
    X_reference = pdt.get_feature_snapshot(
        grade=grade_id,
        target_range=baseline_range,
    )
 
    if X_sample.empty:
        raise ValueError(f"No target data found for grade {grade_id} in range {target_range}")
 
    if X_reference.empty:
        raise ValueError(f"No baseline data found for grade {grade_id} in range {baseline_range}")
 
    res = st.explain_grade_component(
        component=component,
        grade_id=grade_id,
        X_sample=X_sample,
        X_reference=X_reference,
    )
 
    fig = st.build_shap_beeswarm_figure(res, max_features=15)
 
    default_msg = tc.build_default_ranges_message(
        target_range=target_range,
        baseline_range=baseline_range,
        used_any_default=used_any_default,
        lang=lang,
    )
 
    return {
        "type": "shap",
        "component": component,
        "grade_id": grade_id,
        "target_range": target_range,
        "baseline_range": baseline_range,
        "used_default_ranges": used_any_default,
        "message": default_msg,
        "Xe_shape": res["Xe"].shape,
        "n_features": len(res["feature_names"]),
        "shap_shape": res["shap_values"].shape,
        "result": res,
        "figure": fig,
    }
 
 
# -------------------------------------------------
# Main entry point
# -------------------------------------------------
 
def answer(query: str) -> Dict[str, Any]:
    parsed = qp.parse_query(query)
    intent = parsed["intent"]
    cost_component = parsed["cost_component"]
    grade = parsed["grade"]
    # -----------------------------------
    # Orchestrated path
    # -----------------------------------
    if intent == "diagnosis":
        return answer_orchestrated(query)
    if intent == "knowledge" and (
        parsed.get("target_range") is not None
        or parsed.get("baseline_range") is not None
        or parsed.get("grade") is not None
        or parsed.get("cost_component") is not None
    ):
        return answer_orchestrated(query)
    # -----------------------------------
    # Direct paths
    # -----------------------------------
    if intent == "knowledge":
        return answer_knowledge(query)
    if intent == "process_data":
        return answer_process_data(
            target_range=parsed.get("target_range"),
            grade=grade,
        )
    if intent == "prediction":
        if cost_component is None or grade is None:
            raise ValueError(
                "Prediction query detected, but cost_component and grade could not be parsed."
            )
        return answer_prediction(
            cost_component=cost_component,
            grade=grade,
            target_range=parsed.get("target_range"),
        )
    if intent == "shap":
        if cost_component is None or grade is None:
            raise ValueError(
                "SHAP query detected, but cost_component and grade could not be parsed."
            )
        return answer_shap(
            component=cost_component,
            grade_id=grade,
            target_range=parsed.get("target_range"),
            baseline_range=parsed.get("baseline_range"),
        )
    if intent == "cost_driver":
        if cost_component is None or grade is None:
            raise ValueError(
                "Cost driver query detected, but cost_component and grade could not be parsed."
            )
        return answer_cost_driver_analysis(
            target_range=parsed.get("target_range"),
            baseline_range=parsed.get("baseline_range"),
            cost_component=cost_component,
            grade=grade,
        )
    if intent == "simulate_scenario":
        return answer_scenario(
            cost_component=parsed.get("cost_component"),
            grade=parsed.get("grade"),
            reel_id=parsed.get("reel_id"),
            timestamp=parsed.get("timestamp"),
            target_range=parsed.get("target_range"),
            interventions=parsed.get("interventions"),
        )
    raise ValueError(f"Unknown intent: {intent}")

def answer_cost_driver_analysis(
    target_range=None,
    baseline_range=None,
    cost_component: str = None,
    grade: str = None,
    lang: str = "en",
    top_frac: float = 0.20,
) -> Dict[str, Any]:
    import time_context as tc
    resolved = tc.resolve_ranges(
        target_range=target_range,
        baseline_range=baseline_range,
    )
    target_range = resolved["target_range"]
    baseline_range = resolved["baseline_range"]
    used_any_default = resolved["used_any_default"]
    cost_driver_result = cdt.run_cost_driver_analysis(
        target_range=target_range,
        baseline_range=baseline_range,
        cost_component=cost_component,
        grade=grade,
    )
    shapley_contrib = cost_driver_result.get("shapley_contrib")
    summary = cdt.summarize_contributions(shapley_contrib)

    if shapley_contrib is None or shapley_contrib.empty:
        if lang == "de":
            narrative = "Für die ausgewählte Kombination aus Zeitraum, Sorte und Kostenkomponente konnten keine belastbaren Kostentreiber bestimmt werden."
        else:
            narrative = "No reliable cost-driver result could be determined for the selected period, grade, and cost component."
    else:
        narrative = cdt.build_cost_driver_text(
            cost_driver_result=cost_driver_result,
            lang=lang,
            top_frac=top_frac,
        )
        
    narrative = cdt.build_cost_driver_text(
        cost_driver_result=cost_driver_result,
        lang=lang,
        top_frac=top_frac,
    )
    default_msg = tc.build_default_ranges_message(
        target_range=target_range,
        baseline_range=baseline_range,
        used_any_default=used_any_default,
        lang=lang,
    )
    if default_msg:
        narrative = default_msg + "\n\n" + narrative
    return {
        "type": "cost_driver_analysis",
        "component": cost_component,
        "grade": grade,
        "target_range": target_range,
        "baseline_range": baseline_range,
        "used_default_ranges": used_any_default,
        "n_rows": 0 if shapley_contrib is None else len(shapley_contrib),
        "raw": cost_driver_result,
        "shapley_contrib": shapley_contrib,
        "df1": cost_driver_result.get("df1"),
        "df2": cost_driver_result.get("df2"),
        "top_driver_variables": cost_driver_result.get("top_driver_variables", []),
        "extreme_cluster_differences": cost_driver_result.get("extreme_cluster_differences"),
        "summary": summary,
        "narrative": narrative,
    }

def answer_prediction(
    cost_component: str,
    grade: str,
    target_range=None,
    lang: str = "en",
) -> Dict[str, Any]:
 
    import time_context as tc
    import pandas as pd
 
    resolved = tc.resolve_prediction_range(target_range=target_range)
 
    target_range = resolved["target_range"]
    used_default = resolved["used_default"]
 
    # Try to use rows from the selected interval first
    df_period = pdt.get_feature_snapshot(
        grade=grade,
        target_range=target_range,
    )
 
    used_proxy_latest = False
 
    if not df_period.empty:
        # representative operating state from selected interval
        row = df_period.mean(numeric_only=True)
 
        # restore non-numeric fields if useful
        row["AB_Grade_ID"] = grade
 
    else:
        # fallback: latest available operating state for this grade
        df_all = pdt.get_feature_snapshot(grade=grade)
 
        if df_all.empty:
            raise ValueError(f"No data available for grade {grade}")
 
        # Use latest row by Wedge_Time if available
        if "Wedge_Time" in df_all.columns:
            df_all = df_all.sort_values("Wedge_Time")
 
        row = df_all.iloc[-1]
        used_proxy_latest = True
 
    prediction = pt.predict_component_from_row(row, cost_component)
 
    msg = tc.build_prediction_message(
        target_range=target_range,
        used_default=used_default,
        used_proxy_latest=used_proxy_latest,
        lang=lang,
    )
 
    return {
        "type": "prediction",
        "component": cost_component,
        "grade": grade,
        "target_range": target_range,
        "used_default_range": used_default,
        "used_proxy_latest": used_proxy_latest,
        "prediction": prediction,
        "message": msg,
    }

def answer_scenario(
    cost_component,
    grade=None,
    reel_id=None,
    timestamp=None,
    target_range=None,
    interventions=None,
):
    import scenario_tools as st
 
    if not cost_component:
        raise ValueError("Scenario simulation requires a cost_component")
 
    interventions = interventions or []
    if not interventions:
        raise ValueError("Scenario simulation requires at least one intervention")
 
    sim = st.simulate_turnup_scenario(
        cost_component=cost_component,
        grade=grade,
        reel_id=reel_id,
        timestamp=timestamp,
        target_range=target_range,
        interventions=interventions,
    )
 
    return {
        "text": st.describe_scenario_result(sim, decimals=4),
        "data": sim,
    }

def answer_diagnosis(
    target_range,
    baseline_range,
    grades=None,
    levels=None,
    objects=None,
    lang="en",
):
    import diagnosis_tools as diag
 
    if target_range is None:
        raise ValueError("Diagnosis requires target_range")
 
    if baseline_range is None:
        raise ValueError("Diagnosis requires baseline_range")
 
    out = diag.run_diagnosis(
        target_range=target_range,
        baseline_range=baseline_range,
        grades=grades,
        levels=levels,
        objects=objects,
        lang=lang,
    )
 
    return {
        "text": out["combined_text"],
        "data": out,
    }


def answer_orchestrated(query: str) -> Dict[str, Any]:
    import analysis_planner as ap
    import analysis_executor as ae
    import analysis_synthesizer as syn
    parsed = qp.parse_query(query)
    plan_bundle = ap.make_plan(parsed, raw_query=query)
    execution_out = ae.execute_plan(plan_bundle)
    final_out = syn.synthesize_execution(execution_out)
    return {
        "text": final_out["text"],
        "parsed": parsed,
        "plan": final_out["plan"],
        "step_results": final_out["step_results"],
    }