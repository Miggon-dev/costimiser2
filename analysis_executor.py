"""
analysis_executor.py
Step 20 - execution layer for the AI Process Assistant.
"""
from __future__ import annotations
from typing import Any, Dict, List

def _execute_step(
    step: Dict[str, Any],
    raw_query: str,
    executed_steps_so_far: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tool = step["tool"]
    purpose = step.get("purpose", "")
    args = step.get("args", {})
    import assistant_router as ar
    if tool == "diagnosis":
        result = ar.answer_diagnosis(
            target_range=args.get("target_range"),
            baseline_range=args.get("baseline_range"),
            grades=args.get("grades"),
            levels=args.get("levels"),
            objects=args.get("objects"),
            lang=args.get("lang", "en"),
        )
    elif tool == "cost_driver":
        result = ar.answer_cost_driver_analysis(
            target_range=args.get("target_range"),
            baseline_range=args.get("baseline_range"),
            cost_component=args.get("cost_component"),
            grade=args.get("grade"),
        )
    elif tool == "shap":
        result = ar.answer_shap(
            component=args.get("cost_component"),
            grade_id=args.get("grade"),
            target_range=args.get("target_range"),
            baseline_range=args.get("baseline_range"),
        )
    elif tool == "recommend":
        import recommendation_tools as rt

        diagnosis_result = None
        cost_driver_result = None
        shap_result = None
        knowledge_result = None

        for prev in executed_steps_so_far:
            if prev["tool"] == "diagnosis":
                diagnosis_result = prev["result"]

            elif prev["tool"] == "cost_driver":
                if isinstance(prev["result"], dict):
                    cost_driver_result = prev["result"].get("raw", prev["result"])
                else:
                    cost_driver_result = prev["result"]

            elif prev["tool"] == "shap":
                shap_result = prev["result"]

            elif prev["tool"] == "knowledge":
                knowledge_result = prev["result"]

        if cost_driver_result is None:
            raise ValueError("Recommendation step requires a prior cost_driver result")

        result = rt.build_recommendations(
            cost_driver_result=cost_driver_result,
            diagnosis_result=diagnosis_result,
            shap_result=shap_result,
            knowledge_result=knowledge_result,
            lang=args.get("lang", "en"),
        )
    elif tool == "scenario":
        result = ar.answer_scenario(
            cost_component=args.get("cost_component"),
            grade=args.get("grade"),
            reel_id=args.get("reel_id"),
            timestamp=args.get("timestamp"),
            target_range=args.get("target_range"),
            interventions=args.get("interventions"),
        )
    elif tool == "process_data":
        result = ar.answer_process_data(
            target_range=args.get("target_range"),
            grade=args.get("grade"),
        )
    elif tool == "knowledge":
        import recommendation_tools as rt

        cost_driver_result = None
        shap_result = None

        for prev in executed_steps_so_far:
            if prev["tool"] == "cost_driver":
                if isinstance(prev["result"], dict):
                    cost_driver_result = prev["result"].get("raw", prev["result"])
                else:
                    cost_driver_result = prev["result"]

            elif prev["tool"] == "shap":
                shap_result = prev["result"]

        if cost_driver_result is not None:
            query_text = rt.build_knowledge_query_from_drivers(
                cost_driver_result=cost_driver_result,
                shap_result=shap_result,
            )
        else:
            query_text = args.get("query", raw_query)

        # print("RAG",query_text)
        result = ar.answer_knowledge(query_text)
    else:
        raise ValueError(f"Unsupported tool in executor: {tool!r}")
    return {
        "tool": tool,
        "purpose": purpose,
        "args": args,
        "result": result,
    }

def execute_plan(plan_bundle: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(plan_bundle, dict):
        raise ValueError("plan_bundle must be a dict")
    planning_context = plan_bundle.get("planning_context")
    plan = plan_bundle.get("plan")
    if planning_context is None or plan is None:
        raise ValueError("plan_bundle must contain 'planning_context' and 'plan'")
    raw_query = planning_context.get("user_query", "")
    steps = plan.get("steps", [])
    if not isinstance(steps, list) or not steps:
        raise ValueError("plan must contain a non-empty 'steps' list")
    step_results: List[Dict[str, Any]] = []
    for step in steps:
        step_result = _execute_step(
            step,
            raw_query=raw_query,
            executed_steps_so_far=step_results,
        )
        step_results.append(step_result)
    return {
        "planning_context": planning_context,
        "plan": plan,
        "step_results": step_results,
    }