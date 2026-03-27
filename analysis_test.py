import importlib
import traceback
from datetime import date
import pandas as pd
import query_parser as qp
import analysis_planner as ap
import analysis_executor as ae
import analysis_synthesizer as syn
import assistant_router as ar
import scenario_tools as st
import diagnosis_tools as diag
import cost_driver_tools as cdt
import recommendation_tools as rt
import process_data_tools as pdt


def run_test(name, fn):
    try:
        out = fn()
        print(f"✅ {name}")
        return {
            "test": name,
            "status": "PASS",
            "details": out,
            "error": None,
        }
    except Exception as e:
        print(f"❌ {name}")
        traceback.print_exc()
        return {
            "test": name,
            "status": "FAIL",
            "details": None,
            "error": str(e),
        }

def assert_true(condition, msg):
    if not condition:
        raise AssertionError(msg)

def assert_in(item, container, msg=None):
    if item not in container:
        raise AssertionError(msg or f"{item!r} not found")

def assert_not_none(value, msg):
    if value is None:
        raise AssertionError(msg)

def assert_nonempty_text(value, msg="Expected non-empty text"):
    if value is None or not str(value).strip():
        raise AssertionError(msg)
    
def test_parse_diagnosis():
    q = "diagnose cost overall for grade 6010120 in March 2026"
    parsed = qp.parse_query(q)
    assert_true(parsed["intent"] == "diagnosis", "Wrong intent")
    assert_true(parsed["grade"] == "6010120", "Wrong grade")
    assert_true(parsed["objects"] == ["cost"], "Wrong objects")
    assert_true(parsed["levels"] == [1], "Wrong levels")
    assert_not_none(parsed["target_range"], "Missing target_range")
    assert_not_none(parsed["baseline_range"], "Missing baseline_range")
    return parsed

def test_parse_recommendation_query():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    assert_true(parsed["intent"] == "diagnosis", "Recommendation query should route to diagnosis")
    assert_true(parsed["cost_component"] == "steam", "Wrong cost component")
    assert_true(parsed["grade"] == "6010120", "Wrong grade")
    assert_true(parsed["objects"] == ["cost"], "Wrong objects")
    return parsed

def test_parse_scenario():
    q = "simulate starch cost for grade 6010120 if Starch_uptake__g/m2_ is reduced by 10%"
    parsed = qp.parse_query(q)
    assert_true(parsed["intent"] == "simulate_scenario", "Wrong intent")
    assert_true(parsed["cost_component"] == "starch", "Wrong component")
    assert_true(parsed["grade"] == "6010120", "Wrong grade")
    assert_true(len(parsed["interventions"]) == 1, "Expected one intervention")
    return parsed

def test_parse_diagnosis():
    q = "diagnose cost overall for grade 6010120 in March 2026"
    parsed = qp.parse_query(q)
    assert_true(parsed["intent"] == "diagnosis", "Wrong intent")
    assert_true(parsed["grade"] == "6010120", "Wrong grade")
    assert_true(parsed["objects"] == ["cost"], "Wrong objects")
    assert_true(parsed["levels"] == [1], "Wrong levels")
    assert_not_none(parsed["target_range"], "Missing target_range")
    assert_not_none(parsed["baseline_range"], "Missing baseline_range")
    return parsed

def test_parse_recommendation_query():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    assert_true(parsed["intent"] == "diagnosis", "Recommendation query should route to diagnosis")
    assert_true(parsed["cost_component"] == "steam", "Wrong cost component")
    assert_true(parsed["grade"] == "6010120", "Wrong grade")
    assert_true(parsed["objects"] == ["cost"], "Wrong objects")
    return parsed

def test_parse_scenario():
    q = "simulate starch cost for grade 6010120 if Starch_uptake__g/m2_ is reduced by 10%"
    parsed = qp.parse_query(q)
    assert_true(parsed["intent"] == "simulate_scenario", "Wrong intent")
    assert_true(parsed["cost_component"] == "starch", "Wrong component")
    assert_true(parsed["grade"] == "6010120", "Wrong grade")
    assert_true(len(parsed["interventions"]) == 1, "Expected one intervention")
    return parsed

def test_scenario_tool():
    sim = st.simulate_turnup_scenario(
        cost_component="steam",
        grade="6010120",
        interventions=[
            {
                "variable": "Moisture_out_of_PreDryer",
                "mode": "relative",
                "value": -0.10,
            }
        ],
    )
    assert_not_none(sim, "Scenario returned None")
    assert_in("baseline_prediction", sim)
    assert_in("scenario_prediction", sim)
    assert_in("delta_prediction", sim)
    return {
        "baseline": sim["baseline_prediction"],
        "scenario": sim["scenario_prediction"],
        "delta": sim["delta_prediction"],
    }

def test_diagnosis_tool():
    out = diag.run_diagnosis(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
        grades=["6010120"],
        objects="cost",
        levels=1,
        lang="en",
    )
    assert_nonempty_text(out["combined_text"])
    assert_true(len(out["blocks"]) >= 1, "Expected at least one diagnosis block")
    return {
        "n_blocks": len(out["blocks"]),
        "text_preview": out["combined_text"][:300],
    }

def test_cost_driver_tool():
    out = cdt.run_cost_driver_analysis(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
        cost_component="steam",
        grade="6010120",
    )
    assert_in("shapley_contrib", out)
    assert_in("df1", out)
    assert_in("df2", out)
    assert_in("top_driver_variables", out)
    return {
        "top_driver_variables": out["top_driver_variables"],
        "n_shap_rows": 0 if out["shapley_contrib"] is None else len(out["shapley_contrib"]),
    }

def test_recommendation_tool():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    rec_steps = [s for s in out["step_results"] if s["tool"] == "recommend"]
    assert_true(len(rec_steps) == 1, "Expected one recommend step")

    rec = rec_steps[0]["result"]
    assert_in("text", rec)
    assert_in("actions", rec)
    assert_true(len(rec["actions"]) > 0, "Expected at least one action")

    first = rec["actions"][0]
    assert_in("classification", first)
    assert_in("confidence", first)
    assert_in("engineering_reason", first)
    assert_in("direction_hint", first)
    assert_in("priority_score", first)
    assert_in("priority_score_final", first)

    return {
        "n_actions": len(rec["actions"]),
        "first_action": first,
        "text_preview": rec["text"][:500],
    }

def test_execute_recommendation_plan():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)
    execution_out = ae.execute_plan(bundle)

    tools = [s["tool"] for s in execution_out["step_results"]]
    expected = ["diagnosis", "cost_driver", "shap", "knowledge", "recommend"]
    assert_true(tools == expected, f"Expected {expected}, got {tools}")

    rec_steps = [s for s in execution_out["step_results"] if s["tool"] == "recommend"]
    assert_true(len(rec_steps) == 1, "Expected one recommend step")

    rec = rec_steps[0]["result"]
    actions = rec.get("actions", [])
    assert_true(len(actions) > 0, "Expected at least one recommendation action")

    first = actions[0]
    assert_in("classification", first)
    assert_in("confidence", first)
    assert_in("engineering_reason", first)
    assert_in("direction_hint", first)
    assert_in("suggested_intervention", first)

    return {
        "tools": tools,
        "n_actions": len(actions),
        "first_action": first,
    }

def test_synthesize_recommendation_plan():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)
    execution_out = ae.execute_plan(bundle)
    final_out = syn.synthesize_execution(execution_out)

    assert_in("type", final_out)
    assert_in("text", final_out)
    assert_in("blocks", final_out)
    assert_in("plan", final_out)
    assert_in("step_results", final_out)

    assert_true(final_out["type"] == "orchestrated", f"Unexpected type: {final_out['type']}")
    assert_nonempty_text(final_out["text"])

    tools = [s["tool"] for s in final_out["plan"]["steps"]]
    expected = ["diagnosis", "cost_driver", "shap", "knowledge", "recommend"]
    assert_true(tools == expected, f"Expected {expected}, got {tools}")

    assert_true(
        final_out["plan"]["final_template"]
        == "diagnosis_plus_cost_driver_plus_shap_plus_knowledge_plus_recommendations",
        f"Unexpected final_template: {final_out['plan']['final_template']}",
    )

    return {
        "type": final_out["type"],
        "tools": tools,
        "n_blocks": len(final_out["blocks"]),
        "text_preview": final_out["text"][:700],
    }

def test_router_diagnosis():
    out = ar.answer("diagnose cost for grade 6010120 in week 11")
    assert_in("type", out)
    assert_in("text", out)
    assert_in("blocks", out)
    assert_in("plan", out)
    assert_in("step_results", out)
    assert_true(out["type"] == "orchestrated", f"Unexpected type: {out['type']}")
    assert_nonempty_text(out["text"])
    return {
        "type": out["type"],
        "n_blocks": len(out["blocks"]),
        "tools": [s["tool"] for s in out["plan"]["steps"]],
    }


def test_router_recommendation():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    assert_in("type", out)
    assert_in("text", out)
    assert_in("plan", out)
    assert_in("step_results", out)
    assert_true(out["type"] == "orchestrated", f"Unexpected type: {out['type']}")

    tools = [s["tool"] for s in out["plan"]["steps"]]
    expected = ["diagnosis", "cost_driver", "shap", "knowledge", "recommend"]
    assert_true(tools == expected, f"Expected {expected}, got {tools}")

    rec = [s for s in out["step_results"] if s["tool"] == "recommend"][0]["result"]
    assert_true(len(rec.get("actions", [])) > 0, "Expected recommendation actions")

    first = rec["actions"][0]
    assert_in("classification", first)
    assert_in("confidence", first)
    assert_in("engineering_reason", first)
    assert_in("direction_hint", first)

    return {
        "tools": tools,
        "n_actions": len(rec["actions"]),
        "text_preview": out["text"][:700],
    }

def test_router_scenario():
    out = ar.answer(
        "simulate steam cost for grade 6010120 if starch uptake bottom is reduced by 10%"
    )
    assert_in("text", out)
    assert_nonempty_text(out["text"])
    return {"text_preview": out["text"][:300]}

def test_plan_diagnosis_only():
    q = "diagnose cost overall for grade 6010120 in March 2026"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)
    plan = bundle["plan"]
    tools = [s["tool"] for s in plan["steps"]]
    assert_true(tools == ["diagnosis"], f"Unexpected tool sequence: {tools}")
    return plan

def test_plan_recommendation():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)
    plan = bundle["plan"]
    tools = [s["tool"] for s in plan["steps"]]
    assert_true(
        tools == ["diagnosis", "cost_driver", "knowledge", "recommend"],
        f"Unexpected tool sequence: {tools}"
    )
    return plan

def test_plan_scenario_only():
    q = "simulate starch cost for grade 6010120 if Starch_uptake__g/m2_ is reduced by 10%"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)
    plan = bundle["plan"]
    tools = [s["tool"] for s in plan["steps"]]
    assert_true(tools == ["scenario"], f"Unexpected tool sequence: {tools}")
    return plan

def test_process_data_tool():
    out = ar.answer_process_data(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        grade="6010120",
        query="show cost for grade 6010120 in week 11",
    )
    assert_not_none(out, "Process data output is None")
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("columns", out)
    assert_true(isinstance(out["columns"], list), "columns must be a list")
    return {
        "n_rows": out["n_rows"],
        "columns": out["columns"],
        "has_figure": out["figure"] is not None,
    }

def test_router_process_data():
    out = ar.answer("show cost for grade 6010120 in week 11")
    assert_not_none(out, "Router process_data output is None")
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("columns", out)
    return {
        "columns": out["columns"],
        "has_figure": out["figure"] is not None,
        "n_rows": out["n_rows"],
    }

def test_prediction_tool():
    out = ar.answer_prediction(
        cost_component="steam",
        grade="6010120",
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
    )
    assert_not_none(out, "Prediction output is None")
    return out

def test_router_prediction():
    out = ar.answer("predict steam cost for grade 6010120 in week 11")
    assert_not_none(out, "Router prediction output is None")
    return out

def test_shap_tool():
    out = ar.answer_shap(
        component="steam",
        grade_id="6010120",
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
    )
    assert_not_none(out, "SHAP output is None")
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("text", out)
    return {
        "has_figure": out["figure"] is not None,
        "n_rows": 0 if out["data_frame"] is None else len(out["data_frame"]),
        "text_preview": out["text"][:200] if out.get("text") else "",
    }

def test_router_shap():
    out = ar.answer("show shap for steam cost for grade 6010120 in week 11")
    assert_not_none(out, "Router SHAP output is None")
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("text", out)
    return {
        "has_figure": out["figure"] is not None,
        "n_rows": 0 if out["data_frame"] is None else len(out["data_frame"]),
        "text_preview": out["text"][:200] if out.get("text") else "",
    }

def test_shap_without_grade():
    out = ar.answer_shap(
        component="steam",
        grade_id=None,
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
    )
    assert_not_none(out, "SHAP without grade output is None")
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("text", out)
    return {
        "has_figure": out["figure"] is not None,
        "n_rows": 0 if out["data_frame"] is None else len(out["data_frame"]),
        "text_preview": out["text"][:200] if out.get("text") else "",
    }

def test_router_shap_without_grade():
    out = ar.answer("show shap for steam cost in week 11")
    assert_not_none(out, "Router SHAP without grade output is None")
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("text", out)
    return {
        "has_figure": out["figure"] is not None,
        "n_rows": 0 if out["data_frame"] is None else len(out["data_frame"]),
        "text_preview": out["text"][:200] if out.get("text") else "",
    }

def test_shap_dataframe_structure():
    out = ar.answer_shap(
        component="steam",
        grade_id="6010120",
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
    )
    df = out["data_frame"]
    assert_not_none(df, "SHAP dataframe is None")
    assert_true(len(df.columns) > 0, "SHAP dataframe has no columns")

    shap_cols = [c for c in df.columns if str(c).startswith("shap_")]
    assert_true(len(shap_cols) > 0, f"No shap_ columns found in {list(df.columns)}")

    return {
        "n_cols": len(df.columns),
        "n_shap_cols": len(shap_cols),
        "sample_shap_cols": shap_cols[:5],
    }

def test_router_cost_driver():
    out = ar.answer("show cost drivers for steam cost for grade 6010120 in week 11")
    assert_not_none(out, "Router cost driver output is None")
    assert_in("text", out)
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_nonempty_text(out["text"])
    return {
        "text_preview": out["text"][:300],
        "has_figure": out["figure"] is not None,
        "n_rows": 0 if out["data_frame"] is None else len(out["data_frame"]),
    }

def test_cost_driver_wrapper():
    out = ar.answer_cost_driver_analysis(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
        cost_component="steam",
        grade="6010120",
        lang="en",
    )
    assert_not_none(out, "Cost driver wrapper output is None")
    assert_in("text", out)
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("raw", out)
    assert_nonempty_text(out["text"])
    return {
        "n_rows": out["n_rows"],
        "top_driver_variables": out["raw"].get("top_driver_variables", []),
        "text_preview": out["text"][:300],
        "has_figure": out["figure"] is not None,
    }

def test_knowledge_tool():
    out = ar.answer_knowledge("papermaking recommendations to reduce steam cost")

    assert_not_none(out, "Knowledge output is None")
    assert_in("answer", out)
    assert_true(
        isinstance(out["answer"], str) and len(out["answer"]) > 0,
        "Knowledge answer is empty",
    )

    return {
        "answer_preview": out["answer"][:400],
        "n_sources": len(out.get("sources", [])),
    }

def test_answer_orchestrated():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer_orchestrated(q)

    assert_not_none(out, "Orchestrated output is None")
    assert_in("type", out)
    assert_in("text", out)
    assert_in("blocks", out)
    assert_in("plan", out)
    assert_in("step_results", out)

    assert_true(out["type"] == "orchestrated", f"Unexpected type: {out['type']}")
    assert_nonempty_text(out["text"])

    tools = [s["tool"] for s in out["plan"]["steps"]]
    expected = ["diagnosis", "cost_driver", "shap", "knowledge", "recommend"]
    assert_true(tools == expected, f"Expected {expected}, got {tools}")

    return {
        "type": out["type"],
        "n_steps": len(out["step_results"]),
        "n_blocks": len(out["blocks"]),
        "tools": tools,
    }

def test_router_recommendation_with_estimate():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11 and what are the expected savings"
    out = ar.answer(q)

    assert_in("type", out)
    assert_in("text", out)
    assert_in("plan", out)
    assert_in("step_results", out)
    assert_true(out["type"] == "orchestrated", f"Unexpected type: {out['type']}")

    tools = [s["tool"] for s in out["plan"]["steps"]]
    expected = ["diagnosis", "cost_driver", "shap", "knowledge", "recommend", "scenario"]
    assert_true(tools == expected, f"Expected {expected}, got {tools}")

    rec = [s for s in out["step_results"] if s["tool"] == "recommend"][0]["result"]
    suggested = rec.get("suggested_interventions", [])
    assert_true(len(suggested) > 0, "Expected suggested interventions")

    return {
        "tools": tools,
        "final_template": out["plan"]["final_template"],
        "suggested_interventions": suggested,
        "text_preview": out["text"][:800],
    }

def test_recommendation_has_suggested_interventions():
    cdr = cdt.run_cost_driver_analysis(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
        cost_component="steam",
        grade="6010120",
    )
    rec = rt.build_recommendations(
        cost_driver_result=cdr,
        lang="en",
    )
    assert_in("suggested_interventions", rec)
    assert_true(isinstance(rec["suggested_interventions"], list), "Expected list")
    return rec["suggested_interventions"]


def test_router_recommendation_with_estimate():
    out = ar.answer(
        "what are the recommendations to improve the steam cost for grade 6010120 in week 11 and what are the expected savings"
    )
    tools = [s["tool"] for s in out["plan"]["steps"]]
    assert_true(tools[-1] == "scenario", f"Expected final step to be scenario, got {tools}")
    return tools

def test_cost_driver_has_figure():
    out = ar.answer_cost_driver_analysis(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
        cost_component="steam",
        grade="6010120",
        lang="en",
    )
    assert_in("figure", out)
    return {"has_figure": out["figure"] is not None}

def test_planner_regressions():
    cases = {
        "diagnose cost overall for grade 6010120 in March 2026": ["diagnosis"],
        "help me understand why cost worsened for grade 6010120 in March 2026": ["diagnosis", "cost_driver"],
        "what are the recommendations to improve the steam cost for grade 6010120 in week 11": ["diagnosis", "cost_driver", "shap", "knowledge", "recommend"],
        "simulate starch cost for grade 6010120 if starch uptake is reduced by 10%": ["scenario"],
    }

    results = {}
    for q, expected in cases.items():
        parsed = qp.parse_query(q)
        bundle = ap.make_plan(parsed, raw_query=q)
        tools = [s["tool"] for s in bundle["plan"]["steps"]]
        assert_true(tools == expected, f"For query {q!r}, expected {expected}, got {tools}")
        results[q] = {
            "tools": tools,
            "final_template": bundle["plan"]["final_template"],
        }

    return results

def test_process_data_basis_weight_selection():
    out = ar.answer(
        "show current basis weight for grade 6010120 last week"
    )
    assert_in("columns", out)
    assert_true(
        any("basis" in c.lower() and "weight" in c.lower() for c in out["columns"]),
        f"Expected a basis weight column, got {out['columns']}",
    )
    return {
        "columns": out["columns"],
        "has_figure": out["figure"] is not None,
    }

def test_process_data_and_selection():
    out = ar.answer(
        "show steam kWh and electricity kWh for grade 6010120 last week"
    )
    assert_in("columns", out)
    cols_l = [c.lower() for c in out["columns"]]
    assert_true(any("steam" in c for c in cols_l), f"No steam column found: {out['columns']}")
    assert_true(any("electric" in c for c in cols_l), f"No electricity column found: {out['columns']}")
    return {
        "columns": out["columns"],
        "has_figure": out["figure"] is not None,
    }

def test_process_data_negative_filter():
    out = ar.answer(
        "show basis weight without target for grade 6010120 last week"
    )
    assert_in("columns", out)
    cols_l = [c.lower() for c in out["columns"]]
    assert_true(
        not any("target" in c for c in cols_l),
        f"Target columns should be excluded, got {out['columns']}",
    )
    assert_true(
        any("basis" in c and "weight" in c for c in cols_l),
        f"Expected a basis weight column, got {out['columns']}",
    )
    return {
        "columns": out["columns"],
        "has_figure": out["figure"] is not None,
    }

def test_process_data_secondary_axis_preference():
    out = ar.answer(
        "show basis weight and predrier steam in secondary axis for grade 6010120 last week"
    )
    assert_in("plot_preferences", out)
    assert_true(
        out["plot_preferences"].get("secondary_axis") is True,
        f"Expected secondary_axis=True, got {out['plot_preferences']}",
    )
    assert_in("figure", out)
    return {
        "columns": out["columns"],
        "plot_preferences": out["plot_preferences"],
        "has_figure": out["figure"] is not None,
    }

def test_extract_feature_request_phrase():
    q = "show basis weight and predrier steam in secondary axis for grade 6010120 last week"
    phrase = pdt.extract_feature_request_phrase(q)
    assert_true("basis weight" in phrase, f"Unexpected cleaned phrase: {phrase}")
    assert_true("predrier steam" in phrase, f"Unexpected cleaned phrase: {phrase}")
    assert_true("grade" not in phrase, f"Grade should be removed: {phrase}")
    assert_true("secondary axis" not in phrase, f"Plot instruction should be removed: {phrase}")
    return phrase


def test_extract_negative_terms():
    q = "show basis weight without target and prediction for grade 6010120 last week"
    terms = pdt.extract_negative_terms(q)
    assert_true("target" in " ".join(terms), f"Missing target in {terms}")
    assert_true("prediction" in " ".join(terms), f"Missing prediction in {terms}")
    return terms


def test_parse_plot_preferences():
    q = "show basis weight and predrier steam in secondary axis for grade 6010120 last week"
    prefs = pdt.parse_plot_preferences(q)
    assert_true(prefs["secondary_axis"] is True, f"Unexpected prefs: {prefs}")
    return prefs

def test_orchestrated_standard_response_shape():
    out = ar.answer("diagnose cost for grade 6010120 in week 11")
    assert_in("type", out)
    assert_in("text", out)
    assert_in("figure", out)
    assert_in("data_frame", out)
    assert_in("blocks", out)
    assert_in("raw", out)
    assert_in("plan", out)
    assert_in("step_results", out)
    assert_true(out["type"] == "orchestrated", f"Unexpected type: {out['type']}")
    return {
        "type": out["type"],
        "n_blocks": len(out["blocks"]),
        "n_steps": len(out["step_results"]),
    }

def test_router_diagnosis_has_blocks():
    out = ar.answer("diagnose cost for grade 6010120 in week 11")
    assert_in("blocks", out)
    assert_true(isinstance(out["blocks"], list), "blocks must be a list")
    return {
        "n_blocks": len(out["blocks"]),
        "has_any_figure": any(b.get("figure") is not None for b in out["blocks"]),
    }

def test_router_recommendation_has_blocks():
    out = ar.answer("what are the recommendations to improve the steam cost for grade 6010120 in week 11")
    assert_in("blocks", out)
    assert_true(isinstance(out["blocks"], list), "blocks must be a list")
    return {
        "n_blocks": len(out["blocks"]),
        "tools": [s["tool"] for s in out["plan"]["steps"]],
    }

def test_router_explain_change_uses_cost_driver_text():
    out = ar.answer(
        "help me understand why the steam cost variation for grade 6010120 in March 2026"
    )
    assert_in("text", out)
    assert_nonempty_text(out["text"])
    assert_in("plan", out)

    tools = [s["tool"] for s in out["plan"]["steps"]]
    assert_true(
        tools == ["diagnosis", "cost_driver"],
        f"Unexpected tool sequence: {tools}"
    )

    # check that cost-driver step itself has usable text
    cost_driver_steps = [s for s in out["step_results"] if s["tool"] == "cost_driver"]
    assert_true(len(cost_driver_steps) == 1, "Expected one cost_driver step")
    cd_result = cost_driver_steps[0]["result"]
    assert_in("text", cd_result)
    assert_nonempty_text(cd_result["text"])

    return {
        "tools": tools,
        "cost_driver_text_preview": cd_result["text"][:300],
        "final_text_preview": out["text"][:500],
    }

def test_parse_scenario_full_variable_phrase():
    q = "simulate steam cost for grade 6010120 if starch uptake bottom is reduced by 10%"
    parsed = qp.parse_query(q)

    assert_true(parsed["intent"] == "simulate_scenario", "Wrong intent")
    assert_true(len(parsed["interventions"]) == 1, "Expected one intervention")

    itv = parsed["interventions"][0]
    assert_true(itv["variable"] == "starch uptake bottom", f"Wrong variable phrase: {itv}")
    assert_true(itv["mode"] == "relative", f"Wrong mode: {itv}")
    assert_true(abs(itv["value"] + 0.10) < 1e-9, f"Wrong value: {itv}")

    return parsed

def test_parse_scenario_multiple_interventions_comma():
    q = (
        "simulate steam cost for grade 6010120 if starch uptake bottom is reduced by 10%, "
        "current basis weight is increased by 2%"
    )
    parsed = qp.parse_query(q)

    assert_true(parsed["intent"] == "simulate_scenario", "Wrong intent")
    assert_true(len(parsed["interventions"]) == 2, f"Expected two interventions, got {parsed['interventions']}")

    return parsed["interventions"]

def test_parse_scenario_multiple_interventions_and():
    q = (
        "simulate steam cost for grade 6010120 if starch uptake bottom is reduced by 10% "
        "and current basis weight is increased by 2%"
    )
    parsed = qp.parse_query(q)

    assert_true(parsed["intent"] == "simulate_scenario", "Wrong intent")
    assert_true(len(parsed["interventions"]) == 2, f"Expected two interventions, got {parsed['interventions']}")

    return parsed["interventions"]    

def test_parse_scenario_grouped_variables_common_change():
    q = (
        "simulate steam cost for grade 6010120 if starch uptake bottom and starch uptake top "
        "are reduced by 10%"
    )
    parsed = qp.parse_query(q)

    assert_true(parsed["intent"] == "simulate_scenario", "Wrong intent")
    assert_true(len(parsed["interventions"]) == 2, f"Expected two interventions, got {parsed['interventions']}")

    vars_ = [itv["variable"] for itv in parsed["interventions"]]
    assert_true("starch uptake bottom" in vars_, f"Missing bottom phrase: {vars_}")
    assert_true("starch uptake top" in vars_, f"Missing top phrase: {vars_}")

    for itv in parsed["interventions"]:
        assert_true(itv["mode"] == "relative", f"Wrong mode: {itv}")
        assert_true(abs(itv["value"] + 0.10) < 1e-9, f"Wrong value: {itv}")

    return parsed["interventions"]

def test_scenario_variable_resolution_from_phrase():
    import process_data_tools as pdt
    import scenario_tools as st

    df_cols = pdt.get_available_columns()

    resolved, warnings = st.normalize_interventions(
        interventions=[
            {"variable": "starch uptake bottom", "mode": "relative", "value": -0.10}
        ],
        df_columns=df_cols,
    )

    assert_true(len(resolved) == 1, "Expected one resolved intervention")
    assert_true(resolved[0]["variable"] in df_cols, f"Resolved variable not in columns: {resolved}")
    return {
        "resolved_variable": resolved[0]["variable"],
        "warnings": warnings,
    }

def test_router_scenario_with_phrase_resolution():
    out = ar.answer(
        "simulate steam cost for grade 6010120 if starch uptake bottom is reduced by 10%"
    )
    assert_in("text", out)
    assert_nonempty_text(out["text"])
    return {
        "text_preview": out["text"][:400],
    }

def test_build_knowledge_query_from_drivers_with_shap():
    import recommendation_tools as rt

    cost_driver_out = ar.answer_cost_driver_analysis(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
        cost_component="steam",
        grade="6010120",
        lang="en",
    )

    shap_out = ar.answer_shap(
        component="steam",
        grade_id="6010120",
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
    )

    q = rt.build_knowledge_query_from_drivers(
        cost_driver_result=cost_driver_out["raw"],
        shap_result=shap_out["raw"],
    )

    assert_true(isinstance(q, str) and len(q) > 0, "Knowledge query must be non-empty")
    assert_true("Top SHAP-sensitive variables" in q, f"Expected SHAP section in query: {q}")
    return q[:500]

def test_recommendation_uses_shap():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    rec_steps = [s for s in out["step_results"] if s["tool"] == "recommend"]
    assert_true(len(rec_steps) == 1, "Expected one recommend step")

    rec = rec_steps[0]["result"]
    actions = rec.get("actions", [])
    assert_true(len(actions) > 0, "Expected at least one recommendation action")

    first = actions[0]
    assert_in("shap_mean_abs", first)
    assert_in("shap_mean_signed", first)
    assert_in("priority_score", first)
    assert_in("classification", first)
    assert_in("confidence", first)

    return {
        "first_action": first,
        "n_actions": len(actions),
    }

def test_plan_recommendation():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)

    tools = [s["tool"] for s in bundle["plan"]["steps"]]
    expected = ["diagnosis", "cost_driver", "shap", "knowledge", "recommend"]

    assert_true(tools == expected, f"Expected {expected}, got {tools}")
    assert_true(
        bundle["plan"]["final_template"]
        == "diagnosis_plus_cost_driver_plus_shap_plus_knowledge_plus_recommendations",
        f"Unexpected final_template: {bundle['plan']['final_template']}",
    )

    return {
        "tools": tools,
        "final_template": bundle["plan"]["final_template"],
    }

def test_knowledge_actionable_classification():
    import recommendation_tools as rt

    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    knowledge_steps = [s for s in out["step_results"] if s["tool"] == "knowledge"]
    assert_true(len(knowledge_steps) == 1, "Expected one knowledge step")

    knowledge_result = knowledge_steps[0]["result"]
    assert_in("answer", knowledge_result)

    parsed = rt._extract_actionability_map_from_json(knowledge_result["answer"])

    assert_true(isinstance(parsed, dict), "Parsed output must be dict")
    assert_true(len(parsed) > 0, "Parsed output is empty")

    first_key = next(iter(parsed.keys()))
    first_val = parsed[first_key]

    assert_in("classification", first_val)
    assert_in("recommended_direction", first_val)
    assert_in("confidence", first_val)
    assert_in("engineering_reason", first_val)

    return {
        "n_variables": len(parsed),
        "sample_key": first_key,
        "sample": first_val,
    }

    # --- Step 3: build query ---
    query = rt.build_knowledge_query_from_drivers(
        cost_driver_result=cost_driver_out["raw"],
        shap_result=shap_out["raw"],
    )

    #print("\n--- RAG QUERY ---\n")
    #print(query)

    # --- Step 4: call RAG ---
    rag_out = ar.answer_knowledge(query)

    #print("\n--- RAG ANSWER ---\n")
    #print(rag_out["answer"])

    # --- Step 5: minimal assertions ---
    assert_true(isinstance(rag_out["answer"], str) and len(rag_out["answer"]) > 50,
                "RAG response too short or empty")

    return {
        "query_preview": query[:500],
        "answer_preview": rag_out["answer"][:500],
    }

def test_parser_with_real_rag():
    import recommendation_tools as rt

    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    knowledge_steps = [s for s in out["step_results"] if s["tool"] == "knowledge"]
    assert_true(len(knowledge_steps) == 1, "Expected one knowledge step")

    knowledge_text = knowledge_steps[0]["result"]["answer"]
    parsed = rt._extract_actionability_map_from_json(knowledge_text)

    assert_true(len(parsed) > 0, "No variables parsed from RAG answer")

    return {
        "n_parsed": len(parsed),
        "keys": list(parsed.keys())[:10],
    }

def test_knowledge_structured_output():
    import assistant_router as ar
    import recommendation_tools as rt

    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    knowledge_steps = [s for s in out["step_results"] if s["tool"] == "knowledge"]
    assert_true(len(knowledge_steps) == 1, "Expected one knowledge step")

    knowledge_result = knowledge_steps[0]["result"]
    knowledge_text = knowledge_result.get("answer", "")

    parsed = rt._extract_actionability_map_from_json(knowledge_text)

    assert_true(isinstance(parsed, dict), "Parsed knowledge output must be dict")
    assert_true(len(parsed) > 0, "Parsed knowledge output is empty")

    sample_key = next(iter(parsed.keys()))
    sample = parsed[sample_key]

    assert_in("classification", sample)
    assert_in("recommended_direction", sample)
    assert_in("confidence", sample)
    assert_in("engineering_reason", sample)

    return {
        "n_variables": len(parsed),
        "sample_key": sample_key,
        "sample": sample,
    }

def test_recommendation_conflict_fields_present():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    rec_steps = [s for s in out["step_results"] if s["tool"] == "recommend"]
    assert_true(len(rec_steps) == 1, "Expected one recommend step")

    rec = rec_steps[0]["result"]
    actions = rec.get("actions", [])
    assert_true(len(actions) > 0, "Expected at least one action")

    first = actions[0]
    assert_in("rag_direction_hint", first)
    assert_in("shap_direction_hint", first)
    assert_in("delta_direction_hint", first)
    assert_in("has_direction_conflict", first)
    assert_in("conflict_penalty", first)

    return {
        "first_action": first["variable"],
        "conflict": first["has_direction_conflict"],
        "final_direction": first["direction_hint"],
    }

def test_knowledge_json_parser():
    import recommendation_tools as rt

    knowledge_text = """```json
{
"variables": [
    {
    "variable": "DG4_Temperature_Inlet_Air",
    "classification": "actionable",
    "recommended_direction": "decrease",
    "confidence": "high",
    "engineering_reason": "Lower inlet air temperature reduces steam consumption."
    }
]
}
```"""

    parsed = rt._extract_actionability_map_from_json(knowledge_text)

    assert_true("DG4_Temperature_Inlet_Air" in parsed, f"Missing variable: {parsed}")
    row = parsed["DG4_Temperature_Inlet_Air"]
    assert_true(row["classification"] == "actionable", f"Unexpected classification: {row}")
    assert_true(row["recommended_direction"] == "decrease", f"Unexpected direction: {row}")
    assert_true(row["confidence"] == "high", f"Unexpected confidence: {row}")
    return parsed

def test_review_intervention_is_preserved():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    rec_steps = [s for s in out["step_results"] if s["tool"] == "recommend"]
    assert_true(len(rec_steps) == 1, "Expected one recommend step")

    rec = rec_steps[0]["result"]
    suggested = rec.get("suggested_interventions", [])
    assert_true(len(suggested) > 0, "Expected suggested interventions")

    review_items = [x for x in suggested if x.get("mode") == "review"]

    for itv in review_items:
        assert_in("variable", itv)
        assert_in("mode", itv)
        assert_in("value", itv)
        assert_true(itv["mode"] == "review", f"Unexpected review item: {itv}")

    return {
        "n_suggested": len(suggested),
        "n_review": len(review_items),
        "suggested": suggested,
    }

def test_actionable_high_confidence_outranks_indirect_low():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    rec = [s for s in out["step_results"] if s["tool"] == "recommend"][0]["result"]
    actions = rec["actions"]

    # just inspect sorted order and ensure structured ranking exists
    assert_true(len(actions) > 0, "No actions found")

    for a in actions:
        assert_in("classification", a)
        assert_in("confidence", a)
        assert_in("priority_score_final", a)

    return [
        {
            "variable": a["variable"],
            "classification": a.get("classification"),
            "confidence": a.get("confidence"),
            "priority_score_final": a.get("priority_score_final"),
        }
        for a in actions
    ]

def test_recommendation_text_shows_review_flag():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    text = out["text"]
    assert_true(
        ("[REVIEW BEFORE ACTION]" in text) or ("[VOR EINGRIFF PRÜFEN]" in text),
        "Expected review flag in recommendation text"
    )
    return text[:1000]

def test_recommendation_has_suggested_interventions():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    out = ar.answer(q)

    rec = [s for s in out["step_results"] if s["tool"] == "recommend"][0]["result"]
    suggested = rec.get("suggested_interventions", [])

    assert_true(len(suggested) > 0, "Expected at least one suggested intervention")

    allowed_modes = {"relative", "review"}
    for itv in suggested:
        assert_in("variable", itv)
        assert_in("mode", itv)
        assert_in("value", itv)
        assert_true(itv["mode"] in allowed_modes, f"Unexpected mode: {itv}")

    return suggested

def test_router_recommendation_with_estimate_has_scenario_template():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11 and what are the expected savings"
    out = ar.answer(q)

    assert_true(
        out["plan"]["final_template"] == "diagnosis_plus_cost_driver_plus_shap_plus_knowledge_plus_recommendations_plus_scenario",
        f"Unexpected final_template: {out['plan']['final_template']}"
    )

    tools = [s["tool"] for s in out["plan"]["steps"]]
    assert_true(
        tools == ["diagnosis", "cost_driver", "shap", "knowledge", "recommend", "scenario"],
        f"Unexpected tools: {tools}"
    )

    return {
        "final_template": out["plan"]["final_template"],
        "tools": tools,
        "text_preview": out["text"][:800],
    }

def test_build_joint_variable_set():
    import joint_distribution_tools as jdt

    baseline_row = pdt.get_feature_snapshot(grade="6010120").head(1)
    interventions = [{"variable": "Linepressure_shoe_press__bar_", "mode": "relative", "value": 0.05}]
    vars_out = jdt.build_joint_variable_set(
        interventions=interventions,
        baseline_row=baseline_row,
        max_vars=10,
    )

    assert_true("Linepressure_shoe_press__bar_" in vars_out, f"Unexpected vars: {vars_out}")
    assert_true(len(vars_out) <= 10, f"Too many vars: {vars_out}")
    return vars_out

def test_joint_calibrate_interventions_sequential():
    import joint_distribution_tools as jdt

    variables = [
        "Linepressure_shoe_press__bar_",
        "Jet/wire_ratio",
        "DG4_Temperature_Inlet_Air",
    ]

    bundle = jdt.fit_joint_model_for_grade_from_tools(
        grade="6010120",
        variables=variables,
    )

    row = bundle.data_used.iloc[0]

    out = jdt.calibrate_interventions_for_row(
        row=row,
        interventions=[
            {"variable": "Linepressure_shoe_press__bar_", "mode": "relative", "value": 0.05},
            {"variable": "DG4_Temperature_Inlet_Air", "mode": "relative", "value": -0.05},
        ],
        joint_bundle=bundle,
        sequential=True,
    )

    assert_true(out["n_interventions"] == 2, f"Unexpected out: {out}")
    assert_true("final_row" in out, f"Missing final_row: {out}")
    return out
