"""
query_parser.py
 
Rule-based parser for the AI process assistant.
 
Extracts:
- intent
- cost_component
- grade
- reel_id
- timestamp
- target_range
- baseline_range
- interventions
"""
 
from typing import Dict, Any, Optional, List
import re
 
 
COMPONENT_ALIASES = {
    "fibre": ["fibre", "fiber"],
    "steam": ["steam"],
    "electricity": ["electricity", "power"],
    "starch": ["starch"],
}
 
 
def parse_cost_component(query: str) -> Optional[str]:
    q = query.lower()
 
    for canonical, aliases in COMPONENT_ALIASES.items():
        for alias in aliases:
            if alias in q:
                return canonical
 
    return None
 
 
def parse_grade(query: str) -> Optional[str]:
    """
    Extract grade-like numeric codes such as 6010120.
    """
    m = re.search(r"\b\d{6,10}\b", query)
    if m:
        return m.group(0)
    return None
 
 
def parse_reel_id(query: str) -> Optional[str]:
    """
    Extract reel id from patterns like:
    - reel 123456
    - reel id 123456
    - MBS_Current_reel_ID 123456
    """
    patterns = [
        r"\breel\s+id\s+(\d+)\b",
        r"\breel\s+(\d+)\b",
        r"\bmbs_current_reel_id\s+(\d+)\b",
    ]
 
    q = query.lower()
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1)
    return None
 
 
def parse_reference_timestamp(query: str) -> Optional[str]:
    """
    Extract timestamp-like reference from patterns like:
    - at 2026-03-10 14:30
    - timestamp 2026-03-10 14:30
    - on 2026-03-10 14:30
 
    Returns ISO-like string if found.
    """
    q = query.lower()
 
    patterns = [
        r"\bat\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\b",
        r"\btimestamp\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\b",
        r"\bon\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\b",
    ]
 
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1)
 
    return None
 
 
def parse_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["what if", "simulate", "scenario", "if we change", "if i change"]):
        return "simulate_scenario"
    if any(k in q for k in [
        "diagnosis",
        "diagnose",
        "drilldown",
        "root cause",
        "help me understand",
        "understand why",
        "explain why",
        "why did",
        "why has",
        "why cost worsened",
        "why cost increased",
        "what is happening",
        "what's happening",
        "what is driving",
        "what's driving",
        "what is causing",
        "what's causing",
        "recommend",
        "recommendation",
        "recommendations",
        "improve",
        "improvement",
        "reduce cost",
        "what should we do",
        "what can we do",
        "actions",
    ]):
        return "diagnosis"
    if any(k in q for k in ["shap", "shapley", "feature importance"]):
        return "shap"
    if any(k in q for k in ["driver", "drivers", "what changed", "variation"]):
        return "cost_driver"
    if any(k in q for k in ["predict", "prediction", "estimate", "estimated cost"]):
        return "prediction"
    if any(k in q for k in ["data", "trend", "show", "compare"]):
        return "process_data"
    return "knowledge"
 
 
def parse_interventions(query: str):
    q = query.lower()

    if " if " not in q:
        return []

    tail = q.split(" if ", 1)[1].strip()

    interventions = []

    # -------------------------------------------------
    # 1. GROUPED VARIABLES (are reduced / increased / set)
    # -------------------------------------------------
    grouped_patterns = [
        # reduced by %
        (r"(.+?)\s+are\s+reduced\s+by\s+(\d+(?:\.\d+)?)\s*%", "relative", -1, "%"),
        # increased by %
        (r"(.+?)\s+are\s+increased\s+by\s+(\d+(?:\.\d+)?)\s*%", "relative", +1, "%"),
        # reduced absolute
        (r"(.+?)\s+are\s+reduced\s+by\s+(\d+(?:\.\d+)?)$", "delta", -1, None),
        # increased absolute
        (r"(.+?)\s+are\s+increased\s+by\s+(\d+(?:\.\d+)?)$", "delta", +1, None),
    ]

    consumed_spans = []

    for pattern, mode, sign, unit in grouped_patterns:
        for m in re.finditer(pattern, tail):
            var_block = m.group(1).strip()
            raw_value = float(m.group(2))

            if unit == "%":
                value = sign * raw_value / 100.0
            else:
                value = sign * raw_value

            # split variables inside the group
            variables = [
                v.strip()
                for v in re.split(r"\band\b|,", var_block)
                if v.strip()
            ]

            for var in variables:
                interventions.append({
                    "variable": var,
                    "mode": mode,
                    "value": value,
                })

            consumed_spans.append(m.span())

    # -------------------------------------------------
    # 2. REMOVE GROUPED PARTS FROM TEXT
    # -------------------------------------------------
    cleaned_tail = tail
    for start, end in reversed(consumed_spans):
        cleaned_tail = cleaned_tail[:start] + cleaned_tail[end:]

    # -------------------------------------------------
    # 3. SINGLE VARIABLE CLAUSES (existing logic)
    # -------------------------------------------------
    clauses = [
        c.strip()
        for c in re.split(r",|\band\b", cleaned_tail)
        if c.strip()
    ]

    for clause in clauses:
        m = re.match(r"(.+?)\s+is\s+reduced\s+by\s+(\d+(?:\.\d+)?)\s*%", clause)
        if m:
            interventions.append({
                "variable": m.group(1).strip(),
                "mode": "relative",
                "value": -float(m.group(2)) / 100.0,
            })
            continue

        m = re.match(r"(.+?)\s+is\s+increased\s+by\s+(\d+(?:\.\d+)?)\s*%", clause)
        if m:
            interventions.append({
                "variable": m.group(1).strip(),
                "mode": "relative",
                "value": float(m.group(2)) / 100.0,
            })
            continue

        m = re.match(r"(.+?)\s+is\s+reduced\s+by\s+(\d+(?:\.\d+)?)$", clause)
        if m:
            interventions.append({
                "variable": m.group(1).strip(),
                "mode": "delta",
                "value": -float(m.group(2)),
            })
            continue

        m = re.match(r"(.+?)\s+is\s+increased\s+by\s+(\d+(?:\.\d+)?)$", clause)
        if m:
            interventions.append({
                "variable": m.group(1).strip(),
                "mode": "delta",
                "value": float(m.group(2)),
            })
            continue

        m = re.match(r"(.+?)\s+is\s+set\s+to\s+(\d+(?:\.\d+)?)$", clause)
        if m:
            interventions.append({
                "variable": m.group(1).strip(),
                "mode": "absolute",
                "value": float(m.group(2)),
            })
            continue

    return interventions

def parse_query(query: str) -> Dict[str, Any]:
    import pandas as pd

    # ----------------------------
    # RULE-BASED PARSING (your current logic)
    # ----------------------------
    intent = parse_intent(query)

    target_range, baseline_range = parse_date_ranges(query)
    target_kind = None

    if target_range is None:
        from_to = parse_from_to_range(query)
        if from_to is not None:
            target_range = from_to
            target_kind = "single"

    if target_range is None:
        day_range = parse_day_range(query)
        if day_range is not None:
            target_range = day_range
            target_kind = "day"

    if target_range is None:
        week_range = parse_week_range(query)
        if week_range is not None:
            target_range = week_range
            target_kind = "week"

    if target_range is None:
        month_range = parse_month_range(query)
        if month_range is not None:
            target_range = month_range
            target_kind = "month"

    if target_range is not None and baseline_range is None:
        start = pd.to_datetime(target_range[0])

        if target_kind == "month":
            baseline_end = start
            baseline_start = start - pd.offsets.MonthBegin(1)
            baseline_range = (baseline_start.date(), baseline_end.date())
        elif target_kind in ("day", "week", "single"):
            baseline_end = start
            baseline_start = start - pd.Timedelta(weeks=4)
            baseline_range = (baseline_start.date(), baseline_end.date())

    parsed = {
        "intent": intent,
        "cost_component": parse_cost_component(query),
        "grade": parse_grade(query),
        "reel_id": parse_reel_id(query),
        "timestamp": parse_reference_timestamp(query),
        "target_range": target_range,
        "baseline_range": baseline_range,
        "interventions": parse_interventions(query) if intent == "simulate_scenario" else [],
        "raw_query": query,
        "levels": parse_levels(query) if intent == "diagnosis" else None,
        "objects": parse_diagnosis_objects(query) if intent == "diagnosis" else None,
    }

    # ----------------------------
    # LLM FALLBACK (only if needed)
    # ----------------------------
    needs_llm = (
        parsed["intent"] in [None, "unknown"]
        or parsed["grade"] is None
        or parsed["cost_component"] is None
        or (parsed["levels"] is None and "diagnos" in query.lower())
        or (parsed["objects"] is None and "cost" in query.lower())
        or (parsed["interventions"] == [] and contains_intervention_language(query))
    )

    if not needs_llm:
        return parsed

    print("Needs LLM")
    from query_parser_llm import parse_query_llm, merge_rule_and_llm_parse

    llm_parsed = parse_query_llm(query)
    parsed = merge_rule_and_llm_parse(parsed, llm_parsed)

    # ----------------------------
    # RESOLVE TIME FROM LLM TEXT (optional but important)
    # ----------------------------
    if parsed.get("target_range") is None and parsed.get("target_range_text"):
        tr = resolve_time_range_text(parsed["target_range_text"])
        if tr is not None:
            parsed["target_range"] = tr

    if parsed.get("baseline_range") is None and parsed.get("baseline_range_text"):
        br = resolve_time_range_text(parsed["baseline_range_text"])
        if br is not None:
            parsed["baseline_range"] = br

    return parsed
 
def parse_query_TOREMOVE(query: str) -> Dict[str, Any]:
    import pandas as pd
 
    intent = parse_intent(query)
 
    target_range, baseline_range = parse_date_ranges(query)
    target_kind = None
 
    if target_range is None:
        from_to = parse_from_to_range(query)
        if from_to is not None:
            target_range = from_to
            target_kind = "single"
 
    if target_range is None:
        day_range = parse_day_range(query)
        if day_range is not None:
            target_range = day_range
            target_kind = "day"
 
    if target_range is None:
        week_range = parse_week_range(query)
        if week_range is not None:
            target_range = week_range
            target_kind = "week"
 
    if target_range is None:
        month_range = parse_month_range(query)
        if month_range is not None:
            target_range = month_range
            target_kind = "month"
 
    if target_range is not None and baseline_range is None:
        start = pd.to_datetime(target_range[0])
 
        if target_kind == "month":
            baseline_end = start
            baseline_start = start - pd.offsets.MonthBegin(1)
            baseline_range = (baseline_start.date(), baseline_end.date())
        elif target_kind in ("day", "week", "single"):
            baseline_end = start
            baseline_start = start - pd.Timedelta(weeks=4)
            baseline_range = (baseline_start.date(), baseline_end.date())
 
    return {
        "intent": intent,
        "cost_component": parse_cost_component(query),
        "grade": parse_grade(query),
        "reel_id": parse_reel_id(query),
        "timestamp": parse_reference_timestamp(query),
        "target_range": target_range,
        "baseline_range": baseline_range,
        "interventions": parse_interventions(query) if intent == "simulate_scenario" else [],
        "raw_query": query,
        "levels": parse_levels(query) if intent == "diagnosis" else None,
        "objects": parse_diagnosis_objects(query) if intent == "diagnosis" else None,
    }
 
 
def parse_date_ranges(query: str):
    """
    Extract two explicit date ranges from a query.
 
    Expected format:
    - between YYYY-MM-DD and YYYY-MM-DD compared with YYYY-MM-DD and YYYY-MM-DD
    """
    pattern = (
        r"between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})"
        r".*?(?:compared with|vs|versus|against)\s+"
        r"(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})"
    )
 
    m = re.search(pattern, query.lower())
    if not m:
        return None, None
 
    target_range = (m.group(1), m.group(2))
    baseline_range = (m.group(3), m.group(4))
 
    return target_range, baseline_range
 
 
def parse_day_range(query: str):
    """
    Extract standalone ISO date and convert into one-day interval [day, next_day).
    """
    import pandas as pd
 
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if not m:
        return None
 
    day = pd.to_datetime(m.group(1))
    next_day = day + pd.Timedelta(days=1)
 
    return (day.date(), next_day.date())
 
 
def parse_month_range(query: str):
    """
    Extract month name and convert into [month_start, next_month_start).
    """
    import pandas as pd
    from datetime import datetime
 
    month_names = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
 
    q = query.lower()
 
    month_num = None
    for name, num in month_names.items():
        if re.search(rf"\b{name}\b", q):
            month_num = num
            break
 
    if month_num is None:
        return None
 
    year_match = re.search(r"\b(20\d{2})\b", q)
    if year_match:
        year = int(year_match.group(1))
    else:
        year = datetime.today().year
 
    start = pd.Timestamp(year=year, month=month_num, day=1)
    end = start + pd.offsets.MonthBegin(1)
 
    return (start.date(), end.date())
 
 
def parse_week_range(query: str):
    """
    Extract 'week N' and convert to date range.
    """
    import pandas as pd
    from datetime import datetime
 
    m = re.search(r"week\s+(\d{1,2})", query.lower())
    if not m:
        return None
 
    week = int(m.group(1))
    year = datetime.today().year
 
    start = pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u")
    end_exclusive = start + pd.Timedelta(days=5)
 
    return (start.date(), end_exclusive.date())
 
 
def parse_from_to_range(query: str):
    """
    Extract 'from YYYY-MM-DD to YYYY-MM-DD' as a single interval.
    """
    m = re.search(
        r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
        query.lower(),
    )
    if not m:
        return None
 
    return (m.group(1), m.group(2))

def parse_levels(query: str):
    """
    Diagnosis levels:
    1 = overall
    2 = by grade
    3 = by grade and cost component
    """
    q = query.lower()
    levels = []
 
    if "by grade and cost component" in q:
        levels.append(3)
 
    q_without_l3 = q.replace("by grade and cost component", "")
 
    if "overall" in q_without_l3 or "total" in q_without_l3:
        levels.append(1)
 
    if "grade" in q_without_l3 :
        levels.append(2)
 
    if re.search(r"\blevel\s*1\b", q):
        levels.append(1)
    if re.search(r"\blevel\s*2\b", q):
        levels.append(2)
    if re.search(r"\blevel\s*3\b", q):
        levels.append(3)
 
    matches = re.findall(r"\blevels?\s+([123](?:\s*(?:,|and)\s*[123])*)", q)
    for m in matches:
        nums = re.findall(r"[123]", m)
        levels.extend(int(x) for x in nums)
 
    out = []
    seen = set()
    for x in levels:
        if x not in seen:
            out.append(x)
            seen.add(x)
 
    return out or None

def parse_diagnosis_objects(query: str):
    """
    Diagnosis objects:
    - cost = diagnose variation of cost
    - overprocessing = diagnose variation of overprocessing
    """
    q = query.lower()
    objects = []
 
    if "overprocessing" in q:
        objects.append("overprocessing")
 
    if "cost" in q:
        objects.append("cost")
 
    out = []
    seen = set()
    for x in objects:
        if x not in seen:
            out.append(x)
            seen.add(x)
 
    return out or None


def contains_intervention_language(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in [
        "if",
        "increase",
        "decrease",
        "reduce",
        "raise",
        "lower",
    ])

def resolve_time_range_text(range_text: str):
    """
    Resolve a natural-language time phrase into a date range using the
    existing rule-based parsers.
    """
    if not range_text:
        return None

    out = parse_date_ranges(range_text)
    if out is not None and out[0] is not None:
        return out[0]

    from_to = parse_from_to_range(range_text)
    if from_to is not None:
        return from_to

    day_range = parse_day_range(range_text)
    if day_range is not None:
        return day_range

    week_range = parse_week_range(range_text)
    if week_range is not None:
        return week_range

    month_range = parse_month_range(range_text)
    if month_range is not None:
        return month_range

    return None