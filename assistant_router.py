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

from process_data_tools import build_process_plot, get_feature_snapshot, select_features_from_query, DEFAULT_TIME_COL
 
 
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
    grade=None,
    variables=None,
    query: str = None,
):
    import time_context as tc
    import process_data_tools as pdt

    # ------------------------
    # Resolve SINGLE range
    # ------------------------
    resolved = tc.resolve_single_period_range(target_range=target_range)
    target_range = resolved["target_range"]
    used_default = resolved["used_default"]

    # ------------------------
    # Get data
    # ------------------------
    df = get_feature_snapshot(
        target_range=target_range,
        grade=grade,
    )

    if df.empty:
        return {
            "type": "process_data",
            "figure": None,
            "text": "No data available for the selected filters.",
            "target_range": target_range,
        }

    # ------------------------
    # Feature selection
    # ------------------------
    if variables:
        cols = variables
        plot_prefs = {"secondary_axis": False}
    else:
        feature_query = pdt.extract_feature_request_phrase(query or "")
        plot_prefs = pdt.parse_plot_preferences(query or "")
        negative_terms = pdt.extract_negative_terms(query or "")

        cols = pdt.select_features_from_query(
            query=feature_query,
            df_columns=df.columns.tolist(),
            use_llm_fallback=True,
        )

        cols = pdt.filter_columns_by_negative_terms(
            cols,
            negative_terms,
        )

    cols = [c for c in cols if c in df.columns]

    # if secondary axis requested, keep it simple/predictable
    if plot_prefs.get("secondary_axis", False) and len(cols) > 2:
        cols = cols[:2]

    # ------------------------
    # Plot
    # ------------------------
    fig = pdt.build_process_plot(
        df=df,
        columns=cols,
        time_col=DEFAULT_TIME_COL,
        secondary_axis=plot_prefs.get("secondary_axis", False),
    )

    # ------------------------
    # Optional default message
    # ------------------------
    default_msg = tc.build_single_period_message(
        target_range=target_range,
        used_default=used_default,
        lang="en",
    )
    text = default_msg if default_msg else None

    return {
        "type": "process_data",
        "figure": fig,
        "data_frame": df,
        "columns": cols,
        "n_rows": len(df),
        "target_range": target_range,
        "used_default_range": used_default,
        "plot_preferences": plot_prefs,
        "text": text,
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
            variables=parsed.get("variables"),
            query=query,  # 👈 IMPORTANT
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
        "figure": cost_driver_result.get("figure"),
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
    plan_bundle, execution_out = _maybe_append_scenario_step(
        plan_bundle=plan_bundle,
        execution_out=execution_out,
        parsed=parsed,
    )
    final_out = syn.synthesize_execution(execution_out)
    return {
        "text": final_out["text"],
        "parsed": parsed,
        "plan": final_out["plan"],
        "step_results": final_out["step_results"],
    }

def _maybe_append_scenario_step(plan_bundle, execution_out, parsed):
    """
    Second-pass orchestration:
    if the query asks for estimate/savings and recommendations produced a
    suggested intervention, append a scenario step.
    """
    wants_estimate = any(
        k in parsed.get("raw_query", "").lower()
        for k in ["expected savings", "savings", "expected impact", "impact", "how much", "estimate"]
    )
    if not wants_estimate:
        return plan_bundle, execution_out
    plan = execution_out["plan"]
    step_results = execution_out["step_results"]
    # avoid duplicating scenario
    if any(step["tool"] == "scenario" for step in plan.get("steps", [])):
        return plan_bundle, execution_out
    recommend_result = None
    for step in step_results:
        if step["tool"] == "recommend":
            recommend_result = step["result"]
            break
    if not isinstance(recommend_result, dict):
        return plan_bundle, execution_out
    suggested_interventions = recommend_result.get("suggested_interventions", [])
    if not suggested_interventions:
        return plan_bundle, execution_out
    focus = recommend_result.get("focus", {})
    cost_component = focus.get("cost_component") or parsed.get("cost_component")
    grade = focus.get("grade") or parsed.get("grade")
    scenario_step = {
        "tool": "scenario",
        "purpose": "Estimate the expected effect of the top recommended intervention.",
        "args": {
            "cost_component": cost_component,
            "grade": grade,
            "reel_id": parsed.get("reel_id"),
            "timestamp": parsed.get("timestamp"),
            "interventions": [suggested_interventions[0]],
        },
    }
    new_plan = dict(plan)
    new_plan["steps"] = list(plan["steps"]) + [scenario_step]
    final_template = plan.get("final_template")
    if final_template == "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations":
        new_plan["final_template"] = "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations_plus_scenario"
    elif final_template == "diagnosis_plus_cost_driver_plus_recommendations":
        new_plan["final_template"] = "diagnosis_plus_cost_driver_plus_recommendations_plus_scenario"
    new_bundle = {
        "planning_context": plan_bundle["planning_context"],
        "plan": new_plan,
    }
    import analysis_executor as ae
    new_execution_out = ae.execute_plan(new_bundle)
    return new_bundle, new_execution_out

def _maybe_append_scenario_step(plan_bundle, execution_out, parsed):
    wants_estimate = any(
        k in parsed.get("raw_query", "").lower()
        for k in ["expected savings", "savings", "expected impact", "impact", "how much", "estimate"]
    )
    if not wants_estimate:
        return plan_bundle, execution_out
    plan = execution_out["plan"]
    if any(step["tool"] == "scenario" for step in plan.get("steps", [])):
        return plan_bundle, execution_out
    recommend_result = None
    for step in execution_out["step_results"]:
        if step["tool"] == "recommend":
            recommend_result = step["result"]
            break
    if not isinstance(recommend_result, dict):
        return plan_bundle, execution_out
    suggested_interventions = recommend_result.get("suggested_interventions", [])
    if not suggested_interventions:
        return plan_bundle, execution_out
    focus = recommend_result.get("focus", {})
    cost_component = focus.get("cost_component") or parsed.get("cost_component")
    grade = focus.get("grade") or parsed.get("grade")
    scenario_step = {
        "tool": "scenario",
        "purpose": "Estimate the expected effect of the top recommended intervention.",
        "args": {
            "cost_component": cost_component,
            "grade": grade,
            "reel_id": parsed.get("reel_id"),
            "timestamp": parsed.get("timestamp"),
            "interventions": [suggested_interventions[0]],
        },
    }
    new_plan = dict(plan)
    new_plan["steps"] = list(plan["steps"]) + [scenario_step]
    final_template = plan.get("final_template")
    if final_template == "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations":
        new_plan["final_template"] = "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations_plus_scenario"
    elif final_template == "diagnosis_plus_cost_driver_plus_recommendations":
        new_plan["final_template"] = "diagnosis_plus_cost_driver_plus_recommendations_plus_scenario"
    new_bundle = {
        "planning_context": plan_bundle["planning_context"],
        "plan": new_plan,
    }
    import analysis_executor as ae
    new_execution_out = ae.execute_plan(new_bundle)
    return new_bundle, new_execution_out

