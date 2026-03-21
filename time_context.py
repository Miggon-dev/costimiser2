"""
time_context.py
 
Shared time-range policy for the AI process assistant.
 
Rule:
If a workflow needs a time interval and the user did not specify one,
use the default business interval and make that explicit in the answer.
"""
 
from datetime import timedelta
from typing import Optional, Tuple, Dict, Any
 
import pandas as pd
 
from utility import _turnup_data, setpoint_df
 
 
DateRange = Tuple[object, object]
 
 
def get_data_time_bounds() -> Dict[str, Any]:
    turnup_data = _turnup_data(True, "", None, setpoint_df, False).copy()
 
    if "Wedge_Time" not in turnup_data.columns:
        raise KeyError("turnup_data must contain 'Wedge_Time'")
 
    turnup_data["Wedge_Time"] = pd.to_datetime(turnup_data["Wedge_Time"], errors="coerce")
    turnup_data = turnup_data.dropna(subset=["Wedge_Time"])
 
    if turnup_data.empty:
        raise ValueError("turnup_data is empty after parsing Wedge_Time")
 
    return {
        "min_date": turnup_data["Wedge_Time"].min(),
        "max_date": turnup_data["Wedge_Time"].max(),
    }
 
 
def get_default_interval_ranges() -> Dict[str, Any]:
    """
    Default business intervals:
 
    target:
      last complete Monday-Friday window based on the latest available Wedge_Time
 
    baseline:
      previous 4 weeks ending the day before target_start
    """
    bounds = get_data_time_bounds()
    max_date = bounds["max_date"]
    min_date = bounds["min_date"]
 
    last_friday = max_date - pd.to_timedelta((max_date.weekday() - 4) % 7, unit="D")
    target_start = last_friday - pd.Timedelta(days=4)
    target_end_exclusive = last_friday + pd.Timedelta(days=1)
 
    target_range = (target_start.date(), target_end_exclusive.date())
 
    end_date = min(target_start.date() - timedelta(days=1), max_date.date())
    baseline_start = end_date - pd.Timedelta(weeks=4)
    baseline_end = end_date
    baseline_range = (baseline_start, baseline_end)
 
    return {
        "target_range": target_range,
        "baseline_range": baseline_range,
        "min_date": min_date,
        "max_date": max_date,
    }
 
 
def resolve_ranges(
    target_range: Optional[DateRange] = None,
    baseline_range: Optional[DateRange] = None,
) -> Dict[str, Any]:
    defaults = get_default_interval_ranges()
 
    return {
        "target_range": target_range if target_range is not None else defaults["target_range"],
        "baseline_range": baseline_range if baseline_range is not None else defaults["baseline_range"],
        "used_default_target": target_range is None,
        "used_default_baseline": baseline_range is None,
        "used_any_default": (target_range is None or baseline_range is None),
        "min_date": defaults["min_date"],
        "max_date": defaults["max_date"],
    }
 
 
def build_default_ranges_message(
    target_range,
    baseline_range,
    used_any_default: bool,
    lang: str = "en",
) -> str:
    if not used_any_default:
        return ""
 
    if (lang or "en").lower() == "de":
        return (
            f"Es wurden keine Zeitintervalle angegeben, daher wurden die Standardzeiträume verwendet: "
            f"Zielzeitraum={target_range[0]} bis {target_range[1]}, "
            f"Baseline={baseline_range[0]} bis {baseline_range[1]}."
        )
 
    return (
        f"No time intervals were specified, so the default periods were used: "
        f"target={target_range[0]} to {target_range[1]}, "
        f"baseline={baseline_range[0]} to {baseline_range[1]}."
    )

def get_next_week_range():
    """
    Default prediction interval:
    next full Monday–Friday week after the latest available data.
    """
    bounds = get_data_time_bounds()
    max_date = bounds["max_date"]
 
    import pandas as pd
 
    # next Monday after max_date
    days_to_monday = (7 - max_date.weekday()) % 7
    if days_to_monday == 0:
        days_to_monday = 7  # ensure "next" week
 
    next_monday = max_date + pd.Timedelta(days=days_to_monday)
    next_friday = next_monday + pd.Timedelta(days=4)
    next_saturday = next_friday + pd.Timedelta(days=1)  # exclusive end
 
    return (next_monday.date(), next_saturday.date())

def resolve_prediction_range(target_range=None):
    """
    Prediction uses:
    - user-specified range if provided
    - otherwise next-week default
    """
    if target_range is not None:
        return {
            "target_range": target_range,
            "used_default": False,
        }
 
    default_range = get_next_week_range()
 
    return {
        "target_range": default_range,
        "used_default": True,
    }

def build_prediction_range_message(target_range, used_default, lang="en"):
    if not used_default:
        return ""
 
    if (lang or "en").lower() == "de":
        return (
            f"Es wurde kein Zeitraum angegeben, daher wurde die nächste Woche verwendet: "
            f"{target_range[0]} bis {target_range[1]}."
        )
 
    return (
        f"No time interval was specified, so the next week was used: "
        f"{target_range[0]} to {target_range[1]}."
    )

def resolve_time_context(target_range=None, baseline_range=None):
    resolved = resolve_ranges(
        target_range=target_range,
        baseline_range=baseline_range,
    )
 
    return {
        "target_range": resolved["target_range"],
        "baseline_range": resolved["baseline_range"],
        "combined_range": combine_ranges(
            resolved["target_range"],
            resolved["baseline_range"],
        ),
        "used_default_target": resolved["used_default_target"],
        "used_default_baseline": resolved["used_default_baseline"],
        "used_any_default": resolved["used_any_default"],
        "min_date": resolved["min_date"],
        "max_date": resolved["max_date"],
    }

def combine_ranges(target_range=None, baseline_range=None):
    """
    Combine target and baseline into one interval covering both.
    If only one exists, return that one.
    """
    if target_range is None and baseline_range is None:
        resolved = resolve_ranges(None, None)
        target_range = resolved["target_range"]
        baseline_range = resolved["baseline_range"]
 
    elif target_range is None or baseline_range is None:
        resolved = resolve_ranges(target_range, baseline_range)
        target_range = resolved["target_range"]
        baseline_range = resolved["baseline_range"]
 
    starts = [r[0] for r in [target_range, baseline_range] if r is not None]
    ends = [r[1] for r in [target_range, baseline_range] if r is not None]
 
    return (min(starts), max(ends))

def get_default_process_data_range():
    """
    Default process-data interval:
    last complete Monday-Friday window based on latest available Wedge_Time.
    """
    defaults = get_default_interval_ranges()
    return defaults["target_range"]

def resolve_single_period_range(target_range=None):
    """
    Resolve one single interval for process_data.
    """
    if target_range is not None:
        return {
            "target_range": target_range,
            "used_default": False,
        }
 
    default_range = get_default_process_data_range()
 
    return {
        "target_range": default_range,
        "used_default": True,
    }

def build_single_period_message(target_range, used_default: bool, lang: str = "en") -> str:
    if not used_default:
        return ""
 
    if (lang or "en").lower() == "de":
        return (
            f"Es wurde kein Zeitraum angegeben, daher wurde standardmäßig die letzte Woche verwendet: "
            f"{target_range[0]} bis {target_range[1]}."
        )
 
    return (
        f"No time interval was specified, so the last week was used by default: "
        f"{target_range[0]} to {target_range[1]}."
    )

def build_prediction_message(
    target_range,
    used_default: bool,
    used_proxy_latest: bool,
    lang: str = "en",
) -> str:
    lang = (lang or "en").lower()
 
    if lang == "de":
        parts = []
 
        if used_default:
            parts.append(
                f"Es wurde kein Zeitraum angegeben, daher wurde standardmäßig die nächste Woche verwendet: "
                f"{target_range[0]} bis {target_range[1]}."
            )
 
        if used_proxy_latest:
            parts.append(
                "Da für diesen Zeitraum keine beobachteten Prozessdaten verfügbar sind, "
                "wird für die Vorhersage der zuletzt verfügbare Betriebszustand dieser Sorte als Näherung verwendet."
            )
        else:
            parts.append(
                "Die Vorhersage basiert auf einem repräsentativen Betriebszustand aus dem ausgewählten Zeitraum."
            )
 
        return "\n\n".join(parts)
 
    parts = []
 
    if used_default:
        parts.append(
            f"No time interval was specified, so the next week was used by default: "
            f"{target_range[0]} to {target_range[1]}."
        )
 
    if used_proxy_latest:
        parts.append(
            "Because no observed process data is available for that interval, "
            "the prediction uses the latest available operating state for this grade as a proxy."
        )
    else:
        parts.append(
            "The prediction is based on a representative operating state from the selected interval."
        )
 
    return "\n\n".join(parts)