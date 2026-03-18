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
 
def answer_process_data(query: str) -> Dict[str, Any]:
    df = pdt.get_wide_process_data(
        columns=["MBS_Current_reel_ID", "Wedge_Time", "AB_Grade_ID", "Speed", "Electricity__€/T_"]
    )
 
    return {
        "type": "process_data",
        "message": f"Loaded {len(df)} rows from engineered dataset.",
        "data_preview": df.head().to_dict(),
    }
 
 
# -------------------------------------------------
# SHAP workflow
# -------------------------------------------------
 
def answer_shap(
    component: str,
    grade_id: str,
) -> Dict[str, Any]:
    res = st.explain_grade_component(
        component=component,
        grade_id=grade_id,
    )
 
    fig = st.build_shap_beeswarm_figure(res, max_features=15)
 
    return {
        "type": "shap",
        "component": component,
        "grade_id": grade_id,
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
    intent = classify_query(query)
 
    if intent == "knowledge":
        return answer_knowledge(query)
 
    if intent == "process_data":
        return answer_process_data(query)
 
    if intent == "shap":
        raise ValueError(
            "SHAP route detected, but component and grade_id must currently be passed explicitly. "
            "Use answer_shap(component=..., grade_id=...) for now."
        )
    
    if intent == "cost_driver":
        raise ValueError(
            "Cost driver analysis detected, but parameters must currently be passed explicitly. "
            "Use answer_cost_driver_analysis(target_range=..., baseline_range=..., cost_component=..., grade=...)"
        )
 
    raise ValueError(f"Unknown intent: {intent}")

def answer_cost_driver_analysis(
    target_range,
    baseline_range,
    cost_component: str,
    grade: str,
) -> Dict[str, Any]:
 
    df = cdt.run_cost_driver_analysis(
        target_range=target_range,
        baseline_range=baseline_range,
        cost_component=cost_component,
        grade=grade,
    )
 
    summary = cdt.summarize_contributions(df)
 
    return {
        "type": "cost_driver_analysis",
        "component": cost_component,
        "grade": grade,
        "target_range": target_range,
        "baseline_range": baseline_range,
        "n_rows": len(df),
        "raw": df,
        "summary": summary,
    }