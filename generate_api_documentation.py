from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_FILE = Path("api_documentation.pdf")

def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        name="TitleX", parent=s["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=24, textColor=colors.HexColor("#17365D"),
        alignment=TA_CENTER, spaceAfter=8
    ))
    s.add(ParagraphStyle(
        name="SubX", parent=s["Normal"], fontSize=9.5, leading=13,
        textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=14
    ))
    s.add(ParagraphStyle(
        name="H1X", parent=s["Heading1"], fontName="Helvetica-Bold",
        fontSize=14.5, leading=18, textColor=colors.HexColor("#17365D"),
        spaceBefore=9, spaceAfter=6
    ))
    s.add(ParagraphStyle(
        name="H2X", parent=s["Heading2"], fontName="Helvetica-Bold",
        fontSize=11.3, leading=14, textColor=colors.HexColor("#2F5597"),
        spaceBefore=6, spaceAfter=4
    ))
    s.add(ParagraphStyle(
        name="BodyX", parent=s["BodyText"], fontSize=9.2, leading=13.4,
        spaceAfter=5
    ))
    s.add(ParagraphStyle(
        name="SmallX", parent=s["BodyText"], fontSize=8.1, leading=10.5
    ))
    s.add(ParagraphStyle(
        name="CodeX", parent=s["Code"], fontName="Courier",
        fontSize=7.4, leading=9.6, leftIndent=7, rightIndent=7,
        backColor=colors.HexColor("#F4F6F8"),
        borderColor=colors.HexColor("#D9E2F3"), borderWidth=0.5,
        borderPadding=6, spaceBefore=3, spaceAfter=7
    ))
    s.add(ParagraphStyle(
        name="NoteX", parent=s["BodyText"], fontName="Helvetica-Bold",
        fontSize=9.0, leading=13, textColor=colors.HexColor("#7F6000"),
        leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=7
    ))
    return s


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(18*mm, 15*mm, 192*mm, 15*mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(18*mm, 10*mm, "Costimiser AI Analytics Engine - API Documentation v1.1 - 23/07/2026")
    canvas.drawRightString(192*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()


def esc(text):
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace(" ", "&#160;")
                .replace("\n", "<br/>"))


def code(text, s):
    return Paragraph(esc(text), s["CodeX"])


def table(data, widths, s):
    formatted = []
    for r, row in enumerate(data):
        formatted.append([
            Paragraph(f"<b>{cell}</b>", s["SmallX"]) if r == 0
            else Paragraph(str(cell), s["SmallX"])
            for cell in row
        ])
    t = Table(formatted, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#D9E2F3")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#B4C6E7")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FC")]),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return t


def bullets(story, items, s):
    for item in items:
        story.append(Paragraph(f"&#8226;&#160;{item}", s["BodyX"]))


def build():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_FILE), pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm, bottomMargin=20*mm
    )
    story = [
        Paragraph("Costimiser AI Analytics Engine", s["TitleX"]),
        Paragraph(
            "API Documentation<br/>"
            "Version: 1.1<br/>"
            "Date: 23/07/2026<br/>"
            "Status: Draft based on the current notebook usage<br/>"
            "Base URL: http://127.0.0.1:5000",
            s["SubX"]
        )
    ]

    story += [
        Paragraph("1. Overview", s["H1X"]),
        Paragraph(
            "The API exposes endpoints for health checks, process-data retrieval, snapshot prediction, "
            "card-based natural-language analysis, SHAP explanations, diagnosis, cost drivers, scenarios, "
            "recommendations, optimisation, asynchronous jobs, and downloadable artifacts.",
            s["BodyX"]
        ),
        Paragraph(
            "The API distinguishes between <b>analytical functions</b> and <b>process variables</b>. "
            "Analytical functions represent cost, consumption, or strength calculations. They use the following public names:",
            s["BodyX"]
        )
    ]
    bullets(story, ["fibre", "steam", "electricity", "starch uptake", "starch", "total",
                    "SCT CD", "SCT MD", "Burst", "CMT30"], s)
    story += [
        Paragraph(
            "Process variables are the actual model and process-data fields. For structured endpoints, variable names "
            "must match the canonical names exposed by <b>GET /process-data/variables</b>. Examples include "
            "<font name='Courier'>Current_basis_weight</font> and "
            "<font name='Courier'>Current_reel_moisture_average(reel)</font>, but clients should retrieve the current list "
            "rather than rely on examples copied into this document.",
            s["BodyX"]
        ),
        Paragraph(
            "The exception is <b>POST /ask-card</b>: because its input is natural language, users may refer to variables "
            "using friendly expressions such as “current basis weight”, “starch uptake bottom”, or "
            "“starch uptake not top bottom”. The card parser resolves these expressions to canonical variables internally. "
            "This friendly-name resolution does not apply to JSON fields in the raw analytical endpoints.",
            s["NoteX"]
        ),

        Paragraph("2. Common request options", s["H1X"]),
        Paragraph("Asynchronous mode", s["H2X"]),
        Paragraph("Long-running POST endpoints accept:", s["BodyX"]),
        code('{\n  "async": true\n}', s),
        Paragraph(
            "When accepted asynchronously, the endpoint returns HTTP 202 and a job identifier:",
            s["BodyX"]
        ),
        code('{\n  "job_id": "generated-job-id",\n  "job_type": "scenario"\n}', s),
        Paragraph("When async is false, the endpoint returns the final result directly.", s["BodyX"]),
        Paragraph("Artifact mode", s["H2X"]),
        Paragraph("Analytical endpoints accept:", s["BodyX"]),
        code('{\n  "download_artifacts": true\n}', s),
        Paragraph(
            "When true, tables and figures contain artifact URLs. When false, tables and figures are returned inline.",
            s["BodyX"]
        ),
        Paragraph("Typical completed response:", s["BodyX"]),
        code('{\n  "text": "Markdown-formatted explanation",\n  "tables": [],\n  "figures": []\n}', s),
        Paragraph(
            "Downloaded tables are Parquet files. Downloaded figures are Plotly-compatible JSON.",
            s["BodyX"]
        ),

        Paragraph("3. Health", s["H1X"]),
        Paragraph("GET /health", s["H2X"]),
        Paragraph("Checks whether the service is available.", s["BodyX"]),
        code(
            'import requests\n\n'
            'response = requests.get(\n'
            '    "http://127.0.0.1:5000/health",\n'
            '    timeout=30,\n'
            ')\n'
            'print(response.status_code)\n'
            'print(response.text)',
            s
        ),

        Paragraph("4. Process-data endpoints", s["H1X"]),
        Paragraph("4.1 GET /process-data/reels", s["H2X"]),
        Paragraph("Returns reels within a period.", s["BodyX"]),
        table([
            ["Parameter", "Type", "Required"],
            ["start", "datetime string", "Yes"],
            ["end", "datetime string", "Yes"],
        ], [48*mm, 70*mm, 40*mm], s),
        code(
            'response = requests.get(\n'
            '    "http://127.0.0.1:5000/process-data/reels",\n'
            '    params={\n'
            '        "start": "2026-03-01T00:00:00",\n'
            '        "end": "2026-03-31T23:59:59",\n'
            '    },\n'
            ')\n'
            'items = response.json()["items"]',
            s
        ),

        Paragraph("4.2 GET /process-data/snapshot", s["H2X"]),
        Paragraph("Returns a process snapshot selected by either timestamp or reel.", s["BodyX"]),
        table([
            ["Parameter", "Type", "Required"],
            ["timestamp", "datetime string", "Conditional"],
            ["reel_id", "string or integer", "Conditional"],
        ], [48*mm, 70*mm, 40*mm], s),
        code(
            'response = requests.get(\n'
            '    "http://127.0.0.1:5000/process-data/snapshot",\n'
            '    params={"reel_id": "12601843"},\n'
            ')\n'
            'snapshot = response.json()["snapshot"]',
            s
        ),

        Paragraph("4.3 GET /process-data/grades", s["H2X"]),
        Paragraph("Returns the available grades.", s["BodyX"]),
        code(
            'grades = requests.get(\n'
            '    "http://127.0.0.1:5000/process-data/grades"\n'
            ').json()["grades"]',
            s
        ),

        Paragraph("4.4 GET /process-data/variables", s["H2X"]),
        Paragraph(
            "Returns the canonical process-variable names accepted by structured endpoints. "
            "The optional functions parameter filters the list to variables relevant to one or more analytical functions.",
            s["BodyX"]
        ),
        Paragraph(
            "Clients should use this endpoint as the source of truth before supplying variables to "
            "/process-data/parquet, /process-data/variable-bounds, /process-data/grouped, /shap-values, /scenario, or /optimize.",
            s["NoteX"]
        ),
        code(
            'variables = requests.get(\n'
            '    "http://127.0.0.1:5000/process-data/variables",\n'
            '    params={"functions": ["SCT CD", "steam"]},\n'
            ').json()["variables"]',
            s
        ),
        Paragraph("Examples of valid function filters include steam, SCT CD, and starch uptake.", s["BodyX"]),

        Paragraph("4.5 GET /process-data/variable-bounds", s["H2X"]),
        Paragraph("Returns percentile bounds for selected internal process variables.", s["BodyX"]),
        table([
            ["Parameter", "Type", "Required"],
            ["variables", "list of strings", "Yes"],
            ["grade", "string", "Conditional"],
            ["reel_id", "string or integer", "Conditional"],
            ["lower_percentile", "float", "Yes"],
            ["upper_percentile", "float", "Yes"],
        ], [48*mm, 70*mm, 40*mm], s),
        Paragraph("Supply either grade or reel_id.", s["BodyX"]),
        code(
            'bounds = requests.get(\n'
            '    "http://127.0.0.1:5000/process-data/variable-bounds",\n'
            '    params={\n'
            '        "variables": [\n'
            '            "Current_basis_weight",\n'
            '            "Current_reel_moisture_average(reel)",\n'
            '        ],\n'
            '        "grade": "6010120",\n'
            '        "lower_percentile": 0.05,\n'
            '        "upper_percentile": 0.95,\n'
            '    },\n'
            ').json()["bounds"]',
            s
        ),

        Paragraph("4.6 POST /process-data/snapshot-predictions", s["H2X"]),
        Paragraph(
            "Evaluates friendly analytical functions using either a stored reel or supplied reference data.",
            s["BodyX"]
        ),
        table([
            ["Field", "Type", "Required"],
            ["reel_id", "string or integer", "Conditional"],
            ["reference_data", "object", "Conditional"],
            ["functions", "list of strings", "Yes"],
            ["cost_per_m2", "Boolean", "No"],
        ], [48*mm, 70*mm, 40*mm], s),
        Paragraph("Supply either reel_id or reference_data.", s["BodyX"]),
        code(
            '{\n'
            '  "reel_id": 12602792,\n'
            '  "functions": ["SCT CD", "steam", "electricity"],\n'
            '  "cost_per_m2": true\n'
            '}',
            s
        ),
        Paragraph("Response:", s["BodyX"]),
        code(
            '{\n'
            '  "predictions": [\n'
            '    {\n'
            '      "function": "steam",\n'
            '      "prediction": 10.5\n'
            '    }\n'
            '  ]\n'
            '}',
            s
        ),

        Paragraph("4.7 GET /process-data/parquet", s["H2X"]),
        Paragraph("Returns filtered process data directly as a Parquet file.", s["BodyX"]),
        Paragraph("Confirmed query parameters include:", s["BodyX"]),
        table([
            ["Parameter", "Type"],
            ["grade", "string"],
            ["start", "date or datetime string"],
            ["end", "date or datetime string"],
            ["variables", "comma-separated canonical variable names"],
        ], [60*mm, 108*mm], s),
        code(
            'response = requests.get(\n'
            '    "http://127.0.0.1:5000/process-data/parquet",\n'
            '    params={\n'
            '        "grade": "6010120",\n'
            '        "start": "2026-03-01",\n'
            '        "end": "2026-03-10",\n'
            '        "variables": "MBS_SCT_CD,Combined_cost__€/T_",\n'
            '    },\n'
            ')',
            s
        ),

        Paragraph("4.8 POST /process-data/grouped", s["H2X"]),
        Paragraph(
            "Returns two pandas DataFrames: the prepared row-level process data and the grouped summary. "
            "The endpoint supports inline JSON delivery or a ZIP containing two Parquet files.",
            s["BodyX"]
        ),
        table([
            ["Field", "Type", "Required"],
            ["y_variable_summary", "canonical process-variable name", "Yes"],
            ["y_variable_summary_secondary", "canonical process-variable name", "No"],
            ["x_variable_summary", "supported summary name", "Yes"],
            ["color_variable_summary", "supported summary name or null", "No"],
            ["grades", "list of grade identifiers", "No"],
            ["target_range", "two-element date list", "Yes"],
            ["baseline_range", "two-element date list", "Yes"],
            ["output_format", "json or parquet", "No"],
        ], [48*mm, 78*mm, 32*mm], s),
        Paragraph(
            "The endpoint uses internal defaults for the aggregated cost label, the cost components to consider, "
            "and the overprocessing variables. These values are not supplied in the request.",
            s["NoteX"]
        ),
        Paragraph("Valid x_variable_summary values", s["H2X"]),
        table([
            ["Value", "Grouping"],
            ["grade", "AB_Grade_ID; the grouped result also preserves grammage and paper_type"],
            ["day", "Wedge_Date"],
            ["week", "Wedge_Year and ISO Wedge_Week"],
            ["month", "Wedge_Year and Wedge_Month"],
            ["year", "Wedge_Year"],
            ["target", "target"],
        ], [42*mm, 126*mm], s),
        Paragraph("Valid color_variable_summary values", s["H2X"]),
        table([
            ["Value", "Grouping or transformation"],
            ["null or none", "No color grouping"],
            ["grade", "Group and color by AB_Grade_ID"],
            ["target", "Group and color by target"],
            ["target_grade", "Group by target and AB_Grade_ID"],
            ["cost", "Reshape the default cost components into a long-form cost column"],
            ["cost_grade", "Reshape cost components and group by cost and AB_Grade_ID"],
            ["overprocessing", "Reshape the default overprocessing variables into percentage values"],
            ["overprocessing_grade", "Reshape overprocessing values and group by cost and AB_Grade_ID"],
        ], [48*mm, 120*mm], s),
        Paragraph("Weekly cost with Speed as a secondary variable", s["H2X"]),
        Paragraph(
            "The optional y_variable_summary_secondary field preserves and aggregates a second process variable. "
            "In this example, the grouped result contains the weekly mean cost, weekly mean Speed, and observation count.",
            s["BodyX"]
        ),
        code(
            '{\n'
            '  "y_variable_summary": "Combined_cost__€/T_",\n'
            '  "y_variable_summary_secondary": "Speed",\n'
            '  "x_variable_summary": "week",\n'
            '  "color_variable_summary": "cost",\n'
            '  "grades": ["6010120"],\n'
            '  "target_range": ["2026-05-04", "2026-05-10"],\n'
            '  "baseline_range": ["2026-04-01", "2026-05-03"],\n'
            '  "output_format": "parquet"\n'
            '}',
            s
        ),
        Paragraph("JSON output", s["H2X"]),
        Paragraph(
            "When output_format is json, the response contains data and grouped objects using pandas split orientation. "
            "Each object contains columns, index, and data, allowing the client to reconstruct both DataFrames.",
            s["BodyX"]
        ),
        code(
            '{\n'
            '  "data": {\n'
            '    "columns": ["Wedge_Time", "Speed"],\n'
            '    "index": [0],\n'
            '    "data": [["2026-05-04T08:00:00", 1115.0]]\n'
            '  },\n'
            '  "grouped": {\n'
            '    "columns": ["Wedge_Year", "Wedge_Week", "cost", "Cost__€/T_", "Speed", "n"],\n'
            '    "index": [0],\n'
            '    "data": [[2026, 19, "Steam__€/T_", 10.8, 1115.0, 42]]\n'
            '  },\n'
            '  "metadata": {\n'
            '    "x_variable": "Wedge_Week",\n'
            '    "color_variable": "cost"\n'
            '  }\n'
            '}',
            s
        ),
        Paragraph("Parquet output", s["H2X"]),
        Paragraph(
            "When output_format is parquet, the HTTP response is a ZIP archive containing:",
            s["BodyX"]
        )
    ]
    bullets(story, [
        "process_data.parquet - prepared row-level DataFrame",
        "process_grouped.parquet - grouped summary DataFrame",
        "metadata.json - resolved x variable, color variable, color map, and row counts",
    ], s)
    story += [
        Paragraph(
            "The grouped endpoint returns the DataFrames directly and is therefore called synchronously by the current client helper.",
            s["NoteX"]
        ),

        Paragraph("5. Card-based natural-language endpoint", s["H1X"]),
        Paragraph("POST /ask-card", s["H2X"]),
        Paragraph(
            "This is the documented natural-language entry point. /ask is intentionally excluded.",
            s["BodyX"]
        ),
        table([
            ["Field", "Type", "Required"],
            ["query", "string", "Yes"],
            ["download_artifacts", "Boolean", "No"],
            ["diagnosis_summary", "Boolean", "No"],
            ["cost_driver_summary", "Boolean", "No"],
        ], [55*mm, 65*mm, 38*mm], s),
        code(
            '{\n'
            '  "query": "show steam cost for grade 6010120 for week 11",\n'
            '  "download_artifacts": true,\n'
            '  "diagnosis_summary": true,\n'
            '  "cost_driver_summary": false\n'
            '}',
            s
        ),
        Paragraph("Variable names in /ask-card", s["H2X"]),
        Paragraph(
            "Variable references inside query are natural-language expressions, not structured API identifiers. "
            "They may therefore be friendly and do not have to match the exact canonical field name.",
            s["BodyX"]
        )
    ]
    bullets(story, ["current basis weight", "starch uptake bottom", "starch uptake top",
                    "starch uptake not top bottom"], s)
    story += [
        Paragraph(
            "The parser uses the surrounding wording and modifiers to resolve the intended process variable. "
            "By contrast, a JSON request to /scenario or /optimize must use the exact canonical variable name returned by "
            "/process-data/variables.",
            s["NoteX"]
        ),
        Paragraph("Demonstrated query patterns include:", s["BodyX"])
    ]
    bullets(story, [
        "explain model for SCT CD for grade 6010120 in April 2026",
        "steam cost drivers for grade 6010120 in week 18",
        "Diagnose cost for grade 6010120 in week 18",
        "simulate SCT CD for reel id 12602792 if starch uptake bottom is reduced by 10% and current basis weight is increased by 1%",
        "what are the recommendations for steam, grade 6010120 and week 18",
        "maximize SCT CD for reel id 12602391",
        "minimize steam cost subject to SCT CD >= 2.1 for reel id 12602391",
    ], s)

    story += [
        Paragraph("6. SHAP values", s["H1X"]),
        Paragraph("POST /shap-values", s["H2X"]),
        Paragraph("Generates SHAP explanations for a friendly target.", s["BodyX"]),
        table([
            ["Field", "Type", "Required"],
            ["target", "string", "Yes"],
            ["grade", "string", "No"],
            ["start", "date string", "No"],
            ["end", "date string", "No"],
            ["max_rows", "integer", "No"],
            ["background_rows", "integer", "No"],
            ["max_features", "integer", "No"],
            ["variables", "list of canonical names from /process-data/variables", "No"],
            ["async", "Boolean", "No"],
            ["download_artifacts", "Boolean", "No"],
        ], [48*mm, 78*mm, 32*mm], s),
        code(
            '{\n'
            '  "target": "SCT CD",\n'
            '  "grade": "6010120",\n'
            '  "start": "2026-03-01",\n'
            '  "end": "2026-03-10",\n'
            '  "max_rows": 100,\n'
            '  "background_rows": 50,\n'
            '  "async": true\n'
            '}',
            s
        ),
        Paragraph("A returned table may use the identifier shap_values.", s["BodyX"]),

        Paragraph("7. Diagnosis", s["H1X"]),
        Paragraph("POST /diagnosis", s["H2X"]),
        Paragraph("Compares a target period with a baseline period.", s["BodyX"]),
        table([
            ["Field", "Type", "Required"],
            ["grade", "string or null", "No"],
            ["target_range", "two-element date list", "Yes"],
            ["baseline_range", "two-element date list", "Yes"],
            ["levels", "list of integers", "No"],
            ["objects", "list of strings", "No"],
            ["secondary_objects", "list of strings", "No"],
            ["summary", "Boolean", "No"],
            ["async", "Boolean", "No"],
            ["download_artifacts", "Boolean", "No"],
        ], [48*mm, 78*mm, 32*mm], s),
        code(
            '{\n'
            '  "grade": null,\n'
            '  "target_range": ["2026-04-01", "2026-04-30"],\n'
            '  "baseline_range": ["2026-03-01", "2026-03-31"],\n'
            '  "levels": [1, 2, 3, 4],\n'
            '  "objects": ["cost"],\n'
            '  "secondary_objects": ["chemicals", "steam", "electricity"],\n'
            '  "summary": true,\n'
            '  "async": true\n'
            '}',
            s
        ),

        Paragraph("8. Cost drivers", s["H1X"]),
        Paragraph("POST /cost-drivers", s["H2X"]),
        Paragraph("Explains drivers by comparing a target period with a baseline period.", s["BodyX"]),
        table([
            ["Field", "Type", "Required"],
            ["grade", "string", "Yes"],
            ["cost_component", "friendly function name", "Yes"],
            ["target_range", "two-element date list", "Yes"],
            ["baseline_range", "two-element date list", "Yes"],
            ["async", "Boolean", "No"],
            ["download_artifacts", "Boolean", "No"],
        ], [48*mm, 78*mm, 32*mm], s),
        code(
            '{\n'
            '  "grade": "6010120",\n'
            '  "cost_component": "steam",\n'
            '  "target_range": ["2026-04-01", "2026-04-30"],\n'
            '  "baseline_range": ["2026-03-01", "2026-03-31"],\n'
            '  "async": true\n'
            '}',
            s
        ),
        Paragraph("Demonstrated values include steam, starch uptake, and SCT CD.", s["BodyX"]),

        Paragraph("9. What-if scenario", s["H1X"]),
        Paragraph("POST /scenario", s["H2X"]),
        Paragraph("Evaluates interventions against one or more friendly analytical functions.", s["BodyX"]),
        table([
            ["Field", "Type", "Required"],
            ["reel_id", "string or integer", "Conditional"],
            ["reference_data", "object", "Conditional"],
            ["actions", "mapping of canonical variable names to proposed values", "Yes"],
            ["functions", "list of friendly function names", "Yes"],
            ["cost_per_m2", "Boolean", "No"],
            ["async", "Boolean", "No"],
            ["download_artifacts", "Boolean", "No"],
        ], [48*mm, 78*mm, 32*mm], s),
        Paragraph("Supply either reel_id or reference_data.", s["BodyX"]),
        code(
            '{\n'
            '  "reel_id": "12604448",\n'
            '  "actions": {\n'
            '    "Current_basis_weight": 115.0\n'
            '  },\n'
            '  "functions": ["SCT CD", "SCT MD", "steam", "total"],\n'
            '  "cost_per_m2": true,\n'
            '  "async": true\n'
            '}',
            s
        ),
        Paragraph("Known returned table identifiers include:", s["BodyX"])
    ]
    bullets(story, ["scenario_full_snapshot", "scenario_function_evaluation"], s)

    story += [
        Paragraph("10. Recommendations", s["H1X"]),
        Paragraph("POST /recommendations", s["H2X"]),
        Paragraph("Generates recommendations for a grade and cost component.", s["BodyX"]),
        table([
            ["Field", "Type", "Required"],
            ["grade", "string", "Yes"],
            ["cost_component", "friendly function name", "Yes"],
            ["target_range", "two-element date list", "Yes"],
            ["baseline_range", "two-element date list", "Yes"],
            ["async", "Boolean", "No"],
            ["download_artifacts", "Boolean", "No"],
        ], [48*mm, 78*mm, 32*mm], s),
        code(
            '{\n'
            '  "grade": "6010120",\n'
            '  "cost_component": "starch uptake",\n'
            '  "target_range": ["2026-04-01", "2026-05-03"],\n'
            '  "baseline_range": ["2026-05-04", "2026-05-10"],\n'
            '  "async": true\n'
            '}',
            s
        ),
        Paragraph("Demonstrated values include steam, starch, and starch uptake.", s["BodyX"]),

        Paragraph("11. Optimisation", s["H1X"]),
        Paragraph("POST /optimize", s["H2X"]),
        Paragraph(
            "Optimises a friendly analytical function from either a reel or supplied reference data.",
            s["BodyX"]
        ),
        table([
            ["Field", "Type", "Required"],
            ["reel_id", "string or integer", "Conditional"],
            ["reference_data", "object", "Conditional"],
            ["objective_function", "friendly function name", "Yes"],
            ["direction", "minimize or maximize", "Yes"],
            ["constraints", "object, list, or null", "No"],
            ["candidate_features", "canonical variable-name list", "No"],
            ["exclude_features", "canonical variable-name list", "No"],
            ["max_interventions", "integer", "No"],
            ["overprocessing", "Boolean", "No"],
            ["cost_per_m2", "Boolean", "No"],
            ["async", "Boolean", "No"],
            ["download_artifacts", "Boolean", "No"],
        ], [48*mm, 78*mm, 32*mm], s),
        Paragraph("Compact constraint form:", s["H2X"]),
        code('{\n  "SCT CD": 2.05,\n  "SCT MD": 4\n}', s),
        Paragraph("Explicit constraint form:", s["H2X"]),
        code(
            '[\n'
            '  {\n'
            '    "function": "SCT CD",\n'
            '    "operator": ">=",\n'
            '    "value": 1.95\n'
            '  },\n'
            '  {\n'
            '    "function": "Burst",\n'
            '    "operator": ">=",\n'
            '    "value": 240\n'
            '  }\n'
            ']',
            s
        ),
        Paragraph("Full example:", s["H2X"]),
        code(
            '{\n'
            '  "reel_id": "12604077",\n'
            '  "objective_function": "total",\n'
            '  "direction": "minimize",\n'
            '  "constraints": [\n'
            '    {\n'
            '      "function": "SCT CD",\n'
            '      "operator": ">=",\n'
            '      "value": 1.95\n'
            '    },\n'
            '    {\n'
            '      "function": "Burst",\n'
            '      "operator": ">=",\n'
            '      "value": 240\n'
            '    }\n'
            '  ],\n'
            '  "candidate_features": [\n'
            '    "Current_basis_weight",\n'
            '    "Starch_uptake_by_paper_Bottom_Roll__g/m2_"\n'
            '  ],\n'
            '  "max_interventions": 2,\n'
            '  "overprocessing": true,\n'
            '  "cost_per_m2": true,\n'
            '  "async": true\n'
            '}',
            s
        ),
        Paragraph(
            "Demonstrated objective functions include steam, total, starch, starch uptake, and SCT CD.",
            s["BodyX"]
        ),
        Paragraph(
            "Returned optimisation table IDs may be generic names such as block_4, block_5, or block_6. "
            "Clients should use the IDs returned by the API rather than assume fixed names.",
            s["NoteX"]
        ),

        Paragraph("12. Job status", s["H1X"]),
        Paragraph("GET /jobs/{job_id}", s["H2X"]),
        Paragraph("Recognised states are:", s["BodyX"])
    ]
    bullets(story, ["queued", "running", "completed", "failed"], s)
    story += [
        Paragraph("Queued/running response:", s["H2X"]),
        code('{\n  "job_id": "generated-job-id",\n  "status": "running"\n}', s),
        Paragraph("Completed response:", s["H2X"]),
        code(
            '{\n'
            '  "job_id": "generated-job-id",\n'
            '  "status": "completed",\n'
            '  "result": {\n'
            '    "text": "Analysis completed",\n'
            '    "tables": [],\n'
            '    "figures": []\n'
            '  }\n'
            '}',
            s
        ),
        Paragraph("Failed response:", s["H2X"]),
        code(
            '{\n'
            '  "job_id": "generated-job-id",\n'
            '  "status": "failed",\n'
            '  "error": "Description of the error"\n'
            '}',
            s
        ),
        Paragraph("A failed response may also contain a partial or diagnostic result.", s["BodyX"]),

        Paragraph("13. Current job-management limitations", s["H1X"]),
        Paragraph(
            "The current asynchronous interface provides submission and polling, but not a complete job-management system.",
            s["BodyX"]
        )
    ]
    bullets(story, [
        "Clients must retain the returned job_id.",
        "Completion is detected by polling GET /jobs/{job_id}.",
        "No callback or webhook endpoint is documented.",
        "No job-cancellation endpoint is documented.",
        "No endpoint for listing jobs is documented.",
        "No endpoint for deleting jobs is documented.",
        "No percentage-complete or detailed progress field is documented.",
        "Progress is limited to queued, running, completed, and failed.",
        "No job-priority mechanism is documented.",
        "No idempotency key is documented; repeated submissions may create independent jobs.",
        "Job-retention and artifact-retention periods are not part of the documented contract.",
        "Authentication, ownership, and per-user job access are not represented in the demonstrated schemas.",
    ], s)
    story += [
        Paragraph(
            "The notebook does not establish whether job state survives a service restart or whether it is shared across "
            "several pods. These points should remain documented as unknown until the Flask implementation is checked.",
            s["NoteX"]
        ),

        Paragraph("14. Artifact retrieval", s["H1X"]),
        Paragraph(
            "When download_artifacts is true, use the returned artifact URL exactly as supplied.",
            s["BodyX"]
        ),
        Paragraph("Table:", s["H2X"]),
        code(
            'import io\n'
            'import pandas as pd\n'
            'import requests\n\n'
            'response = requests.get(\n'
            '    base_url + table["artifact"]["url"],\n'
            '    timeout=120,\n'
            ')\n'
            'response.raise_for_status()\n'
            'df = pd.read_parquet(io.BytesIO(response.content))',
            s
        ),
        Paragraph("Figure:", s["H2X"]),
        code(
            'import plotly.graph_objects as go\n'
            'import requests\n\n'
            'response = requests.get(\n'
            '    base_url + figure["artifact"]["url"],\n'
            '    timeout=120,\n'
            ')\n'
            'response.raise_for_status()\n'
            'fig = go.Figure(response.json())',
            s
        ),

        Paragraph("15. Naming rules summary", s["H1X"]),
        table([
            ["Context", "Name type required", "Example"],
            ["target, functions, cost_component, objective_function, constraint function",
             "Analytical function name", "steam, starch uptake, SCT CD"],
            ["Raw endpoint variable fields such as variables, actions, candidate_features, and exclude_features",
             "Exact canonical process-variable name from /process-data/variables",
             "Current_basis_weight"],
            ["Natural-language text in /ask-card",
             "Friendly variable wording accepted and resolved by the card parser",
             "current basis weight, starch uptake bottom"],
        ], [58*mm, 66*mm, 44*mm], s),
        Paragraph(
            "Do not use friendly variable wording in structured JSON fields unless the endpoint explicitly documents that behavior.",
            s["NoteX"]
        ),

        Paragraph("16. Endpoint summary", s["H1X"]),
        table([
            ["Method", "Endpoint", "Purpose"],
            ["GET", "/health", "Service health"],
            ["GET", "/process-data/reels", "List reels"],
            ["GET", "/process-data/snapshot", "Retrieve a process snapshot"],
            ["GET", "/process-data/grades", "List grades"],
            ["GET", "/process-data/variables", "List variables"],
            ["GET", "/process-data/variable-bounds", "Retrieve percentile bounds"],
            ["POST", "/process-data/snapshot-predictions", "Predict functions for a snapshot"],
            ["GET", "/process-data/parquet", "Download process data"],
            ["POST", "/process-data/grouped", "Prepared and grouped process DataFrames"],
            ["POST", "/ask-card", "Card-based natural-language analysis"],
            ["POST", "/shap-values", "SHAP explanations"],
            ["POST", "/diagnosis", "Diagnosis"],
            ["POST", "/cost-drivers", "Cost-driver analysis"],
            ["POST", "/scenario", "What-if scenario"],
            ["POST", "/recommendations", "Recommendations"],
            ["POST", "/optimize", "Optimisation"],
            ["GET", "/jobs/{job_id}", "Job status and result"],
            ["GET", "Returned artifact URL", "Download a table or figure"],
        ], [20*mm, 70*mm, 78*mm], s),

        Paragraph("17. Items not established by the notebook", s["H1X"]),
        Paragraph("The notebook does not establish:", s["BodyX"])
    ]
    bullets(story, [
        "Authentication or authorisation headers",
        "Rate limits",
        "Maximum request sizes",
        "Formal OpenAPI schemas",
        "Exact validation rules for every optional parameter",
        "Job and artifact retention periods",
        "Restart recovery",
        "Multi-pod job-state behaviour",
        "Complete error-code mappings",
    ], s)
    story.append(Paragraph(
        "These items should not be described as implemented behavior without checking the Flask source.",
        s["NoteX"]
    ))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    build()
