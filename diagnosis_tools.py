"""
diagnosis_tools.py
"""
 
from __future__ import annotations
 
from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import date
 
import process_data_tools as pdt
from utility import (
    _process_data,
    drilldown_df,
    _drilldown_analysis_plot,
    mix_effect,
    build_drilldown_text,
)
 
 
DEFAULT_LEVELS = (1, 2, 3)
DEFAULT_OBJECTS = ("cost", "overprocessing")
DEFAULT_GRADES = [3200115, 6010085, 6010100, 6010120]


def _normalize_grades(grades: Optional[Union[int, Sequence[int]]]) -> List[str]:
    """
    Normalize grades to list of string (consistent with process_data usage).
    """
    if grades is None:
        out = DEFAULT_GRADES
    elif isinstance(grades, (int, str)):
        out = [grades]
    else:
        out = list(grades)
 
    # convert to string (your pipeline uses str comparison)
    out = [str(g) for g in out]
 
    # remove duplicates preserving order
    seen = set()
    result = []
    for g in out:
        if g not in seen:
            result.append(g)
            seen.add(g)
 
    return result
 
 
def _validate_date_range(name, range_value):
    if range_value is None:
        raise ValueError(f"{name} cannot be None")
 
    if not isinstance(range_value, tuple) or len(range_value) != 2:
        raise ValueError(f"{name} must be a tuple: (start_date, end_date)")
 
    start, end = range_value
    if not isinstance(start, date) or not isinstance(end, date):
        raise ValueError(f"{name} must contain datetime.date values")
 
 
def _normalize_levels(levels: Optional[Union[int, Sequence[int]]]) -> List[int]:
    if levels is None:
        out = list(DEFAULT_LEVELS)
    elif isinstance(levels, int):
        out = [levels]
    else:
        out = [int(x) for x in levels]
 
    valid = {1, 2, 3}
    invalid = [x for x in out if x not in valid]
    if invalid:
        raise ValueError(f"Invalid levels: {invalid}. Supported levels are [1, 2, 3]")
 
    # preserve order, remove duplicates
    seen = set()
    result = []
    for x in out:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result
 
 
def _normalize_objects(
    objects: Optional[Union[str, Sequence[str]]]
) -> List[str]:
    if objects is None:
        out = list(DEFAULT_OBJECTS)
    elif isinstance(objects, str):
        out = [objects.lower()]
    else:
        out = [str(x).lower() for x in objects]
 
    valid = {"cost", "overprocessing"}
    invalid = [x for x in out if x not in valid]
    if invalid:
        raise ValueError(
            f"Invalid object_drilldown values: {invalid}. "
            f"Supported values are ['cost', 'overprocessing']"
        )
 
    # preserve order, remove duplicates
    seen = set()
    result = []
    for x in out:
        if x not in seen:
            result.append(x)
            seen.add(x)
    return result
 
 
def _infer_grades_summary(process_data, grade_col: str = "AB_Grade_ID") -> List[Any]:
    if grade_col not in process_data.columns:
        raise ValueError(f"Missing grade column: {grade_col}")
    return process_data[grade_col].dropna().astype(str).unique().tolist()
 
 
def prepare_diagnosis_process_data(
    target_range,
    baseline_range,
    use_proxy: bool = False,
):
    turnup_data = pdt.load_turnup_data()
    process_data = _process_data(turnup_data, target_range, baseline_range, use_proxy)
    return process_data
 
 
def build_single_diagnosis_block(
    process_data,
    grades_summary,
    level: int,
    object_drilldown: str,
    lang: str = "en",
    reference: str = "baseline",
) -> Dict[str, Any]:
    df_scope = process_data[
        process_data.AB_Grade_ID.astype(str).isin([str(g) for g in grades_summary])
    ]
 
    drilldown = drilldown_df(df_scope, level, object_drilldown, reference)
 
    if level == 1 and object_drilldown == "cost":
        mix_contribution = mix_effect(
            object_drilldown,
            reference,
            grades_summary,
            process_data,
        )
    else:
        mix_contribution = None
 
    description_md = build_drilldown_text(
        drilldown.rename(columns={"cost": object_drilldown}),
        mix_contribution,
        lang=lang,
    )
 
    fig = _drilldown_analysis_plot(
        drilldown,
        mix_contribution,
        level,
        object_drilldown,
        reference,
    )
    fig = fig.update_layout(title="")
 
    return {
        "level": level,
        "object_drilldown": object_drilldown,
        "reference": reference,
        "drilldown": drilldown,
        "mix_contribution": mix_contribution,
        "description_md": description_md,
        "figure": fig,
    }
 
 
def run_diagnosis(
    target_range,
    baseline_range,
    grades: Optional[Union[int, Sequence[int]]] = None,
    levels: Optional[Union[int, Sequence[int]]] = None,
    objects: Optional[Union[str, Sequence[str]]] = None,
    lang: str = "en",
    reference: str = "baseline",
    use_proxy: bool = False,
) -> Dict[str, Any]:
    """
    Time-aware diagnosis runner.
 
    Parameters
    ----------
    target_range : tuple(date, date)
    baseline_range : tuple(date, date)
    grades : int or list[int], optional
        Grades to analyse. Defaults to predefined set.
    levels : int or list[int], optional
        Drilldown levels. Supported: [1, 2, 3]
    objects : str or list[str], optional
        Diagnosis objects. Supported:
            - "cost"
            - "overprocessing"
    lang : str
        "en" or "de"
    reference : str
        Usually "baseline"
    use_proxy : bool
        Passed into _process_data()
 
    Returns
    -------
    dict
    """
 
    # ----------------------------
    # Validate date ranges
    # ----------------------------
    _validate_date_range("target_range", target_range)
    _validate_date_range("baseline_range", baseline_range)
 
    # ----------------------------
    # Normalize inputs
    # ----------------------------
    levels = _normalize_levels(levels)
    objects = _normalize_objects(objects)
    grades = _normalize_grades(grades)
 
    # ----------------------------
    # Load + prepare process_data
    # ----------------------------
    process_data = prepare_diagnosis_process_data(
        target_range=target_range,
        baseline_range=baseline_range,
        use_proxy=use_proxy,
    )
 
    if process_data is None or process_data.empty:
        raise ValueError("Diagnosis process_data is empty")
 
    # ----------------------------
    # Filter grades (CRITICAL STEP)
    # ----------------------------
    process_data = process_data[
        process_data["AB_Grade_ID"].astype(str).isin(grades)
    ]
 
    if process_data.empty:
        raise ValueError(f"No data found for selected grades: {grades}")
 
    grades_summary = grades  # now controlled
 
    # ----------------------------
    # Build diagnosis blocks
    # ----------------------------
    blocks = []
 
    for level in levels:
        for object_drilldown in objects:
            block = build_single_diagnosis_block(
                process_data=process_data,
                grades_summary=grades_summary,
                level=level,
                object_drilldown=object_drilldown,
                lang=lang,
                reference=reference,
            )
            blocks.append(block)
 
    # ----------------------------
    # Combine text
    # ----------------------------
    combined_text = "\n\n".join(
        block["description_md"]
        for block in blocks
        if block["description_md"]
    )
 
    # ----------------------------
    # Final output
    # ----------------------------
    return {
        "target_range": target_range,
        "baseline_range": baseline_range,
        "grades_summary": grades_summary,
        "levels": levels,
        "objects": objects,
        "lang": lang,
        "reference": reference,
        "process_data": process_data,
        "blocks": blocks,
        "combined_text": combined_text,
    }