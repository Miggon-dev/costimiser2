"""
analysis_synthesizer.py
 
Step 20 - synthesis layer for the AI Process Assistant.
 
This module converts executed plan outputs into one final response.
For now, synthesis is template-based and deterministic.
 
Expected flow:
    parsed query
        -> planning_context + plan
        -> execute_plan(...)
        -> synthesize_execution(...)
"""
 
from __future__ import annotations
 
from typing import Any, Dict, List
 
 
def _extract_text(step_result: Dict[str, Any]) -> str:
    """
    Extract human-readable text from one executed step.
    """
    result = step_result.get("result")
 
    if isinstance(result, dict):
        if "text" in result and result["text"] is not None:
            return str(result["text"])
        if "combined_text" in result and result["combined_text"] is not None:
            return str(result["combined_text"])
 
    return str(result)
 
 
def _single_step_response_TOREMOVE(execution_out: Dict[str, Any]) -> Dict[str, Any]:
    step = execution_out["step_results"][0]
    text = _extract_text(step)
 
    return {
        "text": text,
        "plan": execution_out["plan"],
        "step_"
        "results": execution_out["step_results"],
    }

def _single_step_response(execution_out: Dict[str, Any]) -> Dict[str, Any]:
    step = execution_out["step_results"][0]
    result = step.get("result", {})
    text = _extract_text(step)

    figure = result.get("figure") if isinstance(result, dict) else None
    data_frame = result.get("data_frame") if isinstance(result, dict) else None
    blocks = _collect_blocks_from_steps(execution_out["step_results"])

    return {
        "type": "orchestrated",
        "text": text,
        "figure": figure,
        "data_frame": data_frame,
        "blocks": blocks,
        "raw": execution_out,
        "plan": execution_out["plan"],
        "step_results": execution_out["step_results"],
    }    
 
def _collect_blocks_from_steps(step_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten figures from executed step results into a standard top-level blocks list.

    Each block may come from:
    - a direct step figure (cost_driver, scenario, shap, etc.)
    - nested diagnosis blocks
    """
    blocks: List[Dict[str, Any]] = []

    for step in step_results:
        tool = step.get("tool")
        result = step.get("result", {})

        if not isinstance(result, dict):
            continue

        # direct figure from a step
        if result.get("figure") is not None:
            blocks.append(
                {
                    "tool": tool,
                    "level": None,
                    "object_drilldown": None,
                    "text": None,
                    "figure": result.get("figure"),
                    "raw": result,
                }
            )

        # nested blocks (especially diagnosis)
        for block in result.get("blocks", []):
            if not isinstance(block, dict):
                continue

            if block.get("figure") is not None:
                blocks.append(
                    {
                        "tool": tool,
                        "level": block.get("level"),
                        "object_drilldown": block.get("object_drilldown"),
                        "text": block.get("text") or block.get("description_md"),
                        "figure": block.get("figure"),
                        "raw": block,
                    }
                )

        # backward compatibility for old diagnosis wrapper returning {"data": {...}}
        data = result.get("data")
        if isinstance(data, dict):
            for block in data.get("blocks", []):
                if not isinstance(block, dict):
                    continue

                if block.get("figure") is not None:
                    blocks.append(
                        {
                            "tool": tool,
                            "level": block.get("level"),
                            "object_drilldown": block.get("object_drilldown"),
                            "text": block.get("text") or block.get("description_md"),
                            "figure": block.get("figure"),
                            "raw": block,
                        }
                    )

    return blocks 
 
def _synthesize_diagnosis_plus_cost_driver(execution_out: Dict[str, Any]) -> str:
    diagnosis_text = ""
    driver_text = ""
 
    for step in execution_out["step_results"]:
        if step["tool"] == "diagnosis":
            diagnosis_text = _extract_text(step)
        elif step["tool"] == "cost_driver":
            driver_text = _extract_text(step)
 
    parts = []
    if diagnosis_text:
        parts.append(diagnosis_text)
    if driver_text:
        parts.append("Main contributing drivers\n\n" + driver_text)
 
    return "\n\n---\n\n".join(parts)
 
 
def _synthesize_cost_driver_plus_shap(execution_out: Dict[str, Any]) -> str:
    driver_text = ""
    shap_text = ""
 
    for step in execution_out["step_results"]:
        if step["tool"] == "cost_driver":
            driver_text = _extract_text(step)
        elif step["tool"] == "shap":
            shap_text = _extract_text(step)
 
    parts = []
    if driver_text:
        parts.append("Observed drivers\n\n" + driver_text)
    if shap_text:
        parts.append("Model explanation\n\n" + shap_text)
 
    return "\n\n---\n\n".join(parts)
 
 
def _synthesize_knowledge_plus_analysis(execution_out: Dict[str, Any]) -> str:
    knowledge_text = ""
    analysis_parts: List[str] = []
 
    for step in execution_out["step_results"]:
        if step["tool"] == "knowledge":
            knowledge_text = _extract_text(step)
        else:
            analysis_parts.append(_extract_text(step))
 
    parts = []
    if knowledge_text:
        parts.append("Knowledge context\n\n" + knowledge_text)
    if analysis_parts:
        parts.append("Analysis\n\n" + "\n\n".join(analysis_parts))
 
    return "\n\n---\n\n".join(parts)
 
def _synthesize_diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations(
    execution_out: Dict[str, Any]
) -> str:
    diagnosis_text = ""
    driver_text = ""
    recommendation_text = ""
    for step in execution_out["step_results"]:
        if step["tool"] == "diagnosis":
            diagnosis_text = _extract_text(step)
        elif step["tool"] == "cost_driver":
            driver_text = _extract_text(step)
        elif step["tool"] == "recommend":
            recommendation_text = _extract_text(step)
    parts = []
    if diagnosis_text:
        parts.append(diagnosis_text)
    if driver_text:
        parts.append("Main contributing drivers\n\n" + driver_text)
    if recommendation_text:
        parts.append(recommendation_text)
    return "\n\n---\n\n".join(parts)

def _synthesize_diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations_plus_scenario(
    execution_out: Dict[str, Any]
) -> str:
    diagnosis_text = ""
    driver_text = ""
    recommendation_text = ""
    scenario_text = ""
    for step in execution_out["step_results"]:
        if step["tool"] == "diagnosis":
            diagnosis_text = _extract_text(step)
        elif step["tool"] == "cost_driver":
            driver_text = _extract_text(step)
        elif step["tool"] == "recommend":
            recommendation_text = _extract_text(step)
        elif step["tool"] == "scenario":
            scenario_text = _extract_text(step)
    parts = []
    if diagnosis_text:
        parts.append(diagnosis_text)
    if driver_text:
        parts.append("Main contributing drivers\n\n" + driver_text)
    if recommendation_text:
        parts.append(recommendation_text)
    if scenario_text:
        parts.append("Expected impact\n\n" + scenario_text)
    return "\n\n---\n\n".join(parts)

def _synthesize_diagnosis_plus_cost_driver_plus_recommendations(
    execution_out: Dict[str, Any]
) -> str:
    diagnosis_text = ""
    driver_text = ""
    recommendation_text = ""
    for step in execution_out["step_results"]:
        if step["tool"] == "diagnosis":
            diagnosis_text = _extract_text(step)
        elif step["tool"] == "cost_driver":
            driver_text = _extract_text(step)
        elif step["tool"] == "recommend":
            recommendation_text = _extract_text(step)
    parts = []
    if diagnosis_text:
        parts.append(diagnosis_text)
    if driver_text:
        parts.append("Main contributing drivers\n\n" + driver_text)
    if recommendation_text:
        parts.append(recommendation_text)
    return "\n\n---\n\n".join(parts)

def _synthesize_diagnosis_plus_cost_driver_plus_recommendations_plus_scenario(
    execution_out: Dict[str, Any]
) -> str:
    diagnosis_text = ""
    driver_text = ""
    recommendation_text = ""
    scenario_text = ""
    for step in execution_out["step_results"]:
        if step["tool"] == "diagnosis":
            diagnosis_text = _extract_text(step)
        elif step["tool"] == "cost_driver":
            driver_text = _extract_text(step)
        elif step["tool"] == "recommend":
            recommendation_text = _extract_text(step)
        elif step["tool"] == "scenario":
            scenario_text = _extract_text(step)
    parts = []
    if diagnosis_text:
        parts.append(diagnosis_text)
    if driver_text:
        parts.append("Main contributing drivers\n\n" + driver_text)
    if recommendation_text:
        parts.append(recommendation_text)
    if scenario_text:
        parts.append("Expected impact\n\n" + scenario_text)
    return "\n\n---\n\n".join(parts)
 
def synthesize_execution_TOREMOVE(execution_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize executed plan output into one final response.
 
    Parameters
    ----------
    execution_out : dict
        Output of analysis_executor.execute_plan(...)
 
    Returns
    -------
    dict
        {
            "text": ...,
            "plan": ...,
            "step_results": ...,
        }
    """
    if not isinstance(execution_out, dict):
        raise ValueError("execution_out must be a dict")
 
    plan = execution_out.get("plan")
    step_results = execution_out.get("step_results")
 
    if plan is None or step_results is None:
        raise ValueError("execution_out must contain 'plan' and 'step_results'")
 
    if not isinstance(step_results, list) or not step_results:
        raise ValueError("execution_out['step_results'] must be a non-empty list")
 
    if len(step_results) == 1:
        return _single_step_response(execution_out)
 
    final_template = plan.get("final_template")
 
    if final_template == "diagnosis_plus_cost_driver":
        text = _synthesize_diagnosis_plus_cost_driver(execution_out)
 
    elif final_template == "cost_driver_plus_shap":
        text = _synthesize_cost_driver_plus_shap(execution_out)
 
    elif final_template == "knowledge_plus_analysis":
        text = _synthesize_knowledge_plus_analysis(execution_out)
    
    elif final_template == "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations":
        text = _synthesize_diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations(
            execution_out
        )
    elif final_template == "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations_plus_scenario":
        text = _synthesize_diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations_plus_scenario(
            execution_out
        )
    elif final_template == "diagnosis_plus_cost_driver_plus_recommendations":
        text = _synthesize_diagnosis_plus_cost_driver_plus_recommendations(
            execution_out
        )
    elif final_template == "diagnosis_plus_cost_driver_plus_recommendations_plus_scenario":
        text = _synthesize_diagnosis_plus_cost_driver_plus_recommendations_plus_scenario(
            execution_out
        )        
 
    else:
        # generic fallback: concatenate all step texts
        texts = [_extract_text(step) for step in step_results]
        text = "\n\n---\n\n".join(t for t in texts if t)
 
    return {
        "text": text,
        "plan": plan,
        "step_results": step_results,
    }



def synthesize_execution(execution_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize executed plan output into one final response.
 
    Parameters
    ----------
    execution_out : dict
        Output of analysis_executor.execute_plan(...)
 
    Returns
    -------
    dict
        {
            "text": ...,
            "plan": ...,
            "step_results": ...,
        }
    """
    if not isinstance(execution_out, dict):
        raise ValueError("execution_out must be a dict")
 
    plan = execution_out.get("plan")
    step_results = execution_out.get("step_results")
 
    if plan is None or step_results is None:
        raise ValueError("execution_out must contain 'plan' and 'step_results'")
 
    if not isinstance(step_results, list) or not step_results:
        raise ValueError("execution_out['step_results'] must be a non-empty list")
 
    if len(step_results) == 1:
        return _single_step_response(execution_out)
 
    final_template = plan.get("final_template")
 
    if final_template == "diagnosis_plus_cost_driver":
        text = _synthesize_diagnosis_plus_cost_driver(execution_out)
 
    elif final_template == "cost_driver_plus_shap":
        text = _synthesize_cost_driver_plus_shap(execution_out)
 
    elif final_template == "knowledge_plus_analysis":
        text = _synthesize_knowledge_plus_analysis(execution_out)
    
    elif final_template == "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations":
        text = _synthesize_diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations(
            execution_out
        )
    elif final_template == "diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations_plus_scenario":
        text = _synthesize_diagnosis_plus_cost_driver_plus_knowledge_plus_recommendations_plus_scenario(
            execution_out
        )
    elif final_template == "diagnosis_plus_cost_driver_plus_recommendations":
        text = _synthesize_diagnosis_plus_cost_driver_plus_recommendations(
            execution_out
        )
    elif final_template == "diagnosis_plus_cost_driver_plus_recommendations_plus_scenario":
        text = _synthesize_diagnosis_plus_cost_driver_plus_recommendations_plus_scenario(
            execution_out
        )        
 
    else:
        # generic fallback: concatenate all step texts
        texts = [_extract_text(step) for step in step_results]
        text = "\n\n---\n\n".join(t for t in texts if t)
 
    blocks = _collect_blocks_from_steps(step_results)

    return {
        "type": "orchestrated",
        "text": text,
        "figure": None,
        "data_frame": None,
        "blocks": blocks,
        "raw": execution_out,
        "plan": plan,
        "step_results": step_results,
    }