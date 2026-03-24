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
        cost_component="starch",
        grade="6010120",
        interventions=[
            {
                "variable": "Starch_uptake__g/m2_",
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
    assert_nonempty_text(rec["text"])
    assert_in("actions", rec)
    return {
        "n_actions": len(rec["actions"]),
        "text_preview": rec["text"][:300],
    }

def test_execute_recommendation_plan():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)
    execution_out = ae.execute_plan(bundle)
    tools = [s["tool"] for s in execution_out["step_results"]]
    assert_true(
        tools == ["diagnosis", "cost_driver", "knowledge", "recommend"],
        f"Unexpected executed tools: {tools}"
    )
    return {"executed_tools": tools}

def test_synthesize_recommendation_plan():
    q = "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    parsed = qp.parse_query(q)
    bundle = ap.make_plan(parsed, raw_query=q)
    execution_out = ae.execute_plan(bundle)
    final_out = syn.synthesize_execution(execution_out)
    assert_nonempty_text(final_out["text"])
    return {
        "text_preview": final_out["text"][:500],
        "plan": final_out["plan"],
    }

def test_router_diagnosis():
    out = ar.answer("diagnose cost overall for grade 6010120 in March 2026")
    assert_in("text", out)
    assert_nonempty_text(out["text"])
    return {"text_preview": out["text"][:300]}

def test_router_recommendation():
    out = ar.answer("what are the recommendations to improve the steam cost for grade 6010120 in week 11")
    assert_in("text", out)
    assert_nonempty_text(out["text"])
    assert_in("plan", out)
    return {
        "plan": out["plan"],
        "text_preview": out["text"][:500],
    }

def test_router_scenario():
    out = ar.answer("simulate starch cost for grade 6010120 if Starch_uptake__g/m2_ is reduced by 10%")
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
    return out

def test_router_shap():
    out = ar.answer("show shap for steam cost for grade 6010120 in week 11")
    assert_not_none(out, "Router SHAP output is None")
    return out

def test_router_cost_driver():
    out = ar.answer("show cost drivers for steam cost for grade 6010120 in week 11")
    assert_not_none(out, "Router cost driver output is None")
    return out

def test_cost_driver_wrapper():
    out = ar.answer_cost_driver_analysis(
        target_range=(date(2026, 3, 9), date(2026, 3, 14)),
        baseline_range=(date(2026, 2, 9), date(2026, 3, 9)),
        cost_component="steam",
        grade="6010120",
        lang="en",
    )
    assert_not_none(out, "Cost driver wrapper output is None")
    assert_in("raw", out)
    return {
        "n_rows": out["n_rows"],
        "top_driver_variables": out["raw"].get("top_driver_variables", []),
        "narrative_preview": out["narrative"][:300] if out.get("narrative") else "",
    }

def test_knowledge_tool():
    out = ar.answer_knowledge("What is retention?")
    assert_not_none(out, "Knowledge output is None")
    return out

def test_answer_orchestrated():
    out = ar.answer_orchestrated(
        "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    )
    assert_not_none(out, "Orchestrated output is None")
    assert_in("plan", out)
    assert_in("step_results", out)
    assert_nonempty_text(out["text"])
    return {
        "plan": out["plan"],
        "n_steps": len(out["step_results"]),
        "text_preview": out["text"][:400],
    }

def test_router_recommendation_with_estimate():
    out = ar.answer(
        "what are the recommendations to improve the steam cost for grade 6010120 in week 11 and what are the expected savings"
    )
    assert_in("plan", out)
    tools = [s["tool"] for s in out["plan"]["steps"]]
    assert_true(tools[-1] == "scenario", f"Expected scenario as final step, got {tools}")
    assert_nonempty_text(out["text"])
    return {
        "tools": tools,
        "text_preview": out["text"][:500],
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

def test_router_recommendation_without_estimate_no_scenario():
    out = ar.answer(
        "what are the recommendations to improve the steam cost for grade 6010120 in week 11"
    )
    tools = [s["tool"] for s in out["plan"]["steps"]]
    assert_true("scenario" not in tools, f"Did not expect scenario in tools: {tools}")
    return tools

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
        "what are the recommendations to improve the steam cost for grade 6010120 in week 11": ["diagnosis", "cost_driver", "knowledge", "recommend"],
        "simulate starch cost for grade 6010120 if Starch_uptake__g/m2_ is reduced by 10%": ["scenario"],
    }
    results = {}
    for q, expected in cases.items():
        parsed = qp.parse_query(q)
        bundle = ap.make_plan(parsed, raw_query=q)
        tools = [s["tool"] for s in bundle["plan"]["steps"]]
        assert_true(tools == expected, f"For query {q!r}, expected {expected}, got {tools}")
        results[q] = tools
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
        "show steam and electricity for grade 6010120 last week"
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