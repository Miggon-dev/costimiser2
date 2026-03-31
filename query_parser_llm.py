# query_parser_llm.py

from __future__ import annotations

import json
from typing import Any, Dict


ALLOWED_INTENTS = [
    "diagnosis",
    "process_data",
    "prediction",
    "shap",
    "cost_driver",
    "simulate_scenario",
    "knowledge",
    "unknown",
]

ALLOWED_LEVELS = [
    "overall",
    "grade",
    "grade_component",
]

ALLOWED_OBJECTS = [
    "cost",
    "overprocessing",
]

ALLOWED_INTERVENTION_MODES = [
    "relative",
    "delta",
    "absolute",
]


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


def _normalize_llm_parse(obj: Dict[str, Any]) -> Dict[str, Any]:
    intent = obj.get("intent")
    if intent not in ALLOWED_INTENTS:
        intent = "unknown"

    levels = obj.get("levels")
    if not isinstance(levels, list):
        levels = None
    else:
        levels = [x for x in levels if x in ALLOWED_LEVELS] or None

    objects = obj.get("objects")
    if not isinstance(objects, list):
        objects = None
    else:
        objects = [x for x in objects if x in ALLOWED_OBJECTS] or None

    interventions = obj.get("interventions")
    if not isinstance(interventions, list):
        interventions = []
    else:
        clean_itv = []
        for itv in interventions:
            if not isinstance(itv, dict):
                continue

            variable = itv.get("variable")
            mode = itv.get("mode")
            value = itv.get("value")

            if variable is None:
                continue
            if mode not in ALLOWED_INTERVENTION_MODES:
                continue
            if value is None:
                continue

            try:
                value = float(value)
            except Exception:
                continue

            clean_itv.append(
                {
                    "variable": str(variable),
                    "mode": mode,
                    "value": value,
                }
            )
        interventions = clean_itv

    return {
        "intent": intent,
        "cost_component": obj.get("cost_component"),
        "grade": obj.get("grade"),
        "target_range_text": obj.get("target_range_text"),
        "baseline_range_text": obj.get("baseline_range_text"),
        "levels": levels,
        "objects": objects,
        "interventions": interventions,
    }


def parse_query_llm(query: str) -> Dict[str, Any]:
    """
    LLM-assisted parser for flexible queries.
    Returns only structured query content.
    Does NOT set wants_explanation / wants_recommendations / etc.
    """
    import knowledge_retrieval as rag

    prompt = f"""
You are parsing a user query for an industrial analytics assistant.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations.

Allowed intent values:
{ALLOWED_INTENTS}

Allowed levels:
{ALLOWED_LEVELS}

Allowed objects:
{ALLOWED_OBJECTS}

Allowed intervention modes:
{ALLOWED_INTERVENTION_MODES}

Return this exact schema:
{{
  "intent": "...",
  "cost_component": null or string,
  "grade": null or string,
  "target_range_text": null or string,
  "baseline_range_text": null or string,
  "levels": null or list of strings,
  "objects": null or list of strings,
  "interventions": [
    {{
      "variable": "string",
      "mode": "relative|delta|absolute",
      "value": number
    }}
  ]
}}

Rules:
- Use target_range_text and baseline_range_text as natural-language phrases from the query.
- Do NOT invent exact dates.
- If uncertain, use null.
- If the user asks for diagnosis, breakdown, explanation, drivers, or understanding changes, prefer intent="diagnosis".
- If the user asks for process variables or plots, prefer intent="process_data".
- If the user asks about SHAP specifically, prefer intent="shap".
- If the user asks about recommendations, but the workflow is still diagnosis/recommendation, intent can still be "diagnosis".

Query:
{query}
""".strip()

    raw = rag.ask(prompt)
    if isinstance(raw, dict):
        raw = raw.get("answer") or raw.get("text") or str(raw)

    try:
        parsed = json.loads(_strip_json_code_fence(str(raw)))
    except Exception:
        return {
            "intent": "unknown",
            "cost_component": None,
            "grade": None,
            "target_range_text": None,
            "baseline_range_text": None,
            "levels": None,
            "objects": None,
            "interventions": [],
        }

    return _normalize_llm_parse(parsed)

def merge_rule_and_llm_parse(rule_parsed: dict, llm_parsed: dict) -> dict:
    out = dict(rule_parsed)

    for key in [
        "intent",
        "cost_component",
        "grade",
        "levels",
        "objects",
    ]:
        if out.get(key) in [None, [], "unknown"]:
            if llm_parsed.get(key) not in [None, [], "unknown"]:
                out[key] = llm_parsed[key]

    if not out.get("interventions"):
        out["interventions"] = llm_parsed.get("interventions", [])

    if out.get("target_range") is None and llm_parsed.get("target_range_text"):
        out["target_range_text"] = llm_parsed["target_range_text"]

    if out.get("baseline_range") is None and llm_parsed.get("baseline_range_text"):
        out["baseline_range_text"] = llm_parsed["baseline_range_text"]

    return out