from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


OUTPUT_FILE = Path("ask_card_orchestration_explanation.pdf")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=24,
        textColor=colors.HexColor("#17365D"),
        alignment=TA_CENTER,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="SubtitleCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=14,
    ))

    styles.add(ParagraphStyle(
        name="H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14.5,
        leading=18,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=9,
        spaceAfter=6,
    ))

    styles.add(ParagraphStyle(
        name="H2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.2,
        leading=14,
        textColor=colors.HexColor("#2F5597"),
        spaceBefore=6,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.3,
        leading=13.4,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.7,
    ))

    styles.add(ParagraphStyle(
        name="CalloutCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-BoldOblique",
        fontSize=9.1,
        leading=13.2,
        textColor=colors.HexColor("#2F5597"),
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="QuoteCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9.4,
        leading=13.5,
        textColor=colors.HexColor("#2F5597"),
        leftIndent=18,
        rightIndent=18,
        spaceBefore=4,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="SummaryCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.1,
        leading=13.3,
        backColor=colors.HexColor("#EAF2F8"),
        borderPadding=7,
        spaceBefore=5,
        spaceAfter=7,
    ))

    return styles


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawCentredString(
        105 * mm,
        10 * mm,
        "Costimiser AI Analytics Engine - Ask-Card orchestration",
    )
    canvas.restoreState()


def make_table(data, widths, styles, header_color="#D9E2F3"):
    rows = []
    for row_index, row in enumerate(data):
        rows.append([
            Paragraph(
                f"<b>{cell}</b>" if row_index == 0 else str(cell),
                styles["SmallCustom"],
            )
            for cell in row
        ])

    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#666666")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#F7F9FC"),
        ]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def add_numbered_steps(story, steps, styles):
    for number, text in enumerate(steps, start=1):
        story.append(
            Paragraph(
                f"{number}.&#160;&#160;{text}",
                styles["BodyCustom"],
            )
        )


def add_route(story, card_name, tool_text, dependency_text, styles):
    story.append(Paragraph(card_name, styles["H2Custom"]))
    story.append(
        Paragraph(
            f"<b>{card_name.replace(' card', '')} tool:</b> {tool_text}",
            styles["BodyCustom"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Dependencies managed by the tool:</b> {dependency_text}",
            styles["BodyCustom"],
        )
    )


def build_document():
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
    )

    story = [
        Paragraph("ASK-CARD ORCHESTRATION", styles["TitleCustom"]),
        Paragraph(
            "Companion explanation for the Costimiser analytical-engine diagram",
            styles["SubtitleCustom"],
        ),
        Paragraph("Architecture note", styles["H1Custom"]),
        Paragraph(
            "<b>Core principle:</b> /ask-card interprets and routes a natural-language "
            "request. The selected analytical card delegates the work to a specialised "
            "tool. That tool contains the main analytical logic, manages access to its "
            "dependencies, and returns a result compatible with the common card-response format.",
            styles["SummaryCustom"],
        ),
        Paragraph("Scope", styles["H1Custom"]),
        Paragraph(
            "This document explains the orchestration represented in the accompanying "
            "Draw.io diagram. It focuses only on the natural-language entry point "
            "<b>POST /ask-card</b> and the analytical paths that it can select.",
            styles["BodyCustom"],
        ),

        Paragraph("1. Purpose of the orchestration layer", styles["H1Custom"]),
        Paragraph(
            "The /ask-card endpoint provides a natural-language interface over the analytical "
            "capabilities of the Costimiser engine. It does not duplicate the business logic "
            "already implemented in the analytical tools. Instead, it understands the request, "
            "normalises the extracted parameters, selects the appropriate card and delegates "
            "execution to the corresponding tool.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "Only one primary analytical path is selected for a request. The parallel branches "
            "in the diagram represent alternative routes, not operations that are executed simultaneously.",
            styles["CalloutCustom"],
        ),
        Paragraph("The end-to-end flow is:", styles["BodyCustom"]),
    ]

    add_numbered_steps(story, [
        "Natural-language query",
        "POST /ask-card",
        "Query interpretation",
        "Friendly-name resolution",
        "Intent and card selection",
        "Specialised tool execution",
        "Optional enrichment",
        "Common response assembly",
    ], styles)

    story += [
        Paragraph("2. Request interpretation and routing", styles["H1Custom"]),
        Paragraph("2.1 Request entry", styles["H2Custom"]),
        Paragraph(
            "A user, frontend, notebook or internal service sends a natural-language query "
            "to POST /ask-card. The request can also include response options such as "
            "<font name='Courier'>download_artifacts</font>, "
            "<font name='Courier'>diagnosis_summary</font> and "
            "<font name='Courier'>cost_driver_summary</font>.",
            styles["BodyCustom"],
        ),

        Paragraph("2.2 Query interpretation", styles["H2Custom"]),
        Paragraph(
            "The interpretation stage identifies the analytical intent and extracts the "
            "parameters required to execute it. Depending on the request, these parameters "
            "may include grade, reel identifier, date range, target period, baseline period, "
            "cost or strength function, process variables, optimisation direction, constraints "
            "and chart requirements.",
            styles["BodyCustom"],
        ),

        Paragraph("2.3 Friendly-name resolution", styles["H2Custom"]),
        Paragraph(
            "The natural-language interface distinguishes analytical function names from "
            "process-variable names. Functions such as steam, electricity, starch, starch uptake, "
            "total, SCT CD and Burst are public analytical names. Process variables used by "
            "structured endpoints must match their canonical names, which are available through "
            "the process-data variable endpoint.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "Inside /ask-card, users may refer to process variables using friendly expressions. "
            "The resolution logic translates those expressions into the canonical names required "
            "by the selected tool. For example, “current basis weight” can be resolved to "
            "<font name='Courier'>Current_basis_weight</font>.",
            styles["BodyCustom"],
        ),

        Paragraph("2.4 Intent and card router", styles["H2Custom"]),
        Paragraph(
            "After interpretation and normalisation, the router selects one primary analytical "
            "card. The card represents the request type and provides the bridge between the "
            "orchestration layer and the corresponding tool.",
            styles["BodyCustom"],
        ),

        Paragraph("3. Responsibilities of cards, tools and dependencies", styles["H1Custom"]),
    ]

    responsibilities = [
        ["Component", "Primary responsibility", "What it does not do"],
        [
            "Card",
            "Represents the selected analytical intent, receives normalised parameters, "
            "invokes its tool and exposes the result in the common card format.",
            "It does not own the detailed analytical implementation.",
        ],
        [
            "Tool",
            "Contains the main analytical logic, validates and prepares inputs, accesses "
            "required dependencies, performs the analysis and builds the card result.",
            "It does not require the router to manage data or model access on its behalf.",
        ],
        [
            "Dependency",
            "Provides data, models, metadata, storage or external services required by a tool.",
            "It is not an orchestration entry point and is not called directly by the card router.",
        ],
    ]
    story.append(
        make_table(
            responsibilities,
            [38 * mm, 72 * mm, 58 * mm],
            styles,
        )
    )
    story.append(
        Paragraph(
            "Architectural rule: Card -&gt; Tool -&gt; Dependencies. "
            "The tool owns both the main logic and dependency access.",
            styles["CalloutCustom"],
        )
    )

    story.append(Paragraph("4. Analytical routes", styles["H1Custom"]))

    add_route(
        story,
        "Process-data card",
        "Retrieves, filters, compares and visualises process data.",
        "Process-data repository; reel and grade information; canonical variable catalogue.",
        styles,
    )
    add_route(
        story,
        "SHAP card",
        "Retrieves the relevant data and target model, computes SHAP contributions "
        "and builds explanation tables and figures.",
        "Process data; function registry; selected prediction model.",
        styles,
    )
    add_route(
        story,
        "Diagnosis card",
        "Compares a target period with a baseline period and applies the diagnostic "
        "hierarchy and summary logic.",
        "Target and baseline data; diagnosis rules; relevant cost and process components.",
        styles,
    )
    add_route(
        story,
        "Cost-drivers card",
        "Explains the variables or components responsible for a change between target "
        "and baseline periods.",
        "Target and baseline data; function resolution; driver-decomposition logic.",
        styles,
    )
    add_route(
        story,
        "Scenario card",
        "Retrieves or receives a reference snapshot, applies requested interventions, "
        "evaluates functions before and after and builds scenario outputs.",
        "Snapshots; canonical variables; function registry; cost and strength models.",
        styles,
    )
    add_route(
        story,
        "Recommendations card",
        "Combines analytical evidence with actionable-variable logic and process knowledge "
        "to produce operational recommendations.",
        "Cost-driver outputs; process knowledge; model information; optional optimisation results.",
        styles,
    )
    add_route(
        story,
        "Optimisation card",
        "Interprets the objective and constraints, retrieves the reference point and bounds, "
        "evaluates candidate interventions and returns the best feasible result.",
        "Snapshots; variable bounds; canonical variable catalogue; cost and strength models.",
        styles,
    )
    add_route(
        story,
        "Knowledge / RAG card",
        "Embeds the question, retrieves relevant document chunks and generates a grounded answer.",
        "Papermaking documents; vector database or FAISS; embedding model; language model.",
        styles,
    )

    story += [
        Paragraph("5. Selected result and optional enrichment", styles["H1Custom"]),
        Paragraph(
            "All alternative analytical routes converge conceptually into a selected card result. "
            "This collector does not combine the outputs of every branch. It represents the result "
            "returned by the single card chosen for the current request.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "After the primary tool completes, /ask-card may append optional diagnosis or "
            "cost-driver summaries when the corresponding request options are enabled. "
            "These summaries enrich the primary result; they do not replace the selected analytical operation.",
            styles["BodyCustom"],
        ),

        Paragraph("6. Common response assembly", styles["H1Custom"]),
        Paragraph(
            "The selected result and any optional enrichment are converted into a common "
            "card-response structure. This allows clients to handle all analytical cards consistently.",
            styles["BodyCustom"],
        ),
    ]

    response_table = [
        ["Response element", "Purpose"],
        ["text", "Markdown-formatted explanation or recommendation."],
        [
            "tables",
            "Structured tabular results, returned inline or as downloadable Parquet artifacts.",
        ],
        [
            "figures",
            "Plotly figures, returned inline or through downloadable artifact URLs.",
        ],
    ]
    story.append(
        make_table(
            response_table,
            [82 * mm, 86 * mm],
            styles,
            header_color="#D9EAD3",
        )
    )

    story += [
        Paragraph("6.1 Inline delivery", styles["H2Custom"]),
        Paragraph(
            "When download_artifacts is false, tables and figures are embedded directly "
            "in the JSON response.",
            styles["BodyCustom"],
        ),
        Paragraph("6.2 Artifact delivery", styles["H2Custom"]),
        Paragraph(
            "When download_artifacts is true, tables and figures are stored as artifacts "
            "and the response contains URLs that the client can retrieve.",
            styles["BodyCustom"],
        ),

        Paragraph("7. Example orchestration", styles["H1Custom"]),
        Paragraph("For the request:", styles["BodyCustom"]),
        Paragraph(
            "“Minimize steam cost subject to SCT CD &gt;= 2.1 for reel 12602391.”",
            styles["QuoteCustom"],
        ),
    ]

    add_numbered_steps(story, [
        "The query interpreter identifies an optimisation request.",
        "The friendly-name resolver recognises steam as the objective function and SCT CD as a constraint function.",
        "The router selects the Optimisation card.",
        "The card invokes the Optimisation tool with the normalised reel, objective, direction and constraint parameters.",
        "The tool retrieves the reference snapshot and variable bounds, accesses the relevant cost and strength models, and executes the optimisation logic.",
        "The tool returns the feasible optimum and supporting tables or figures.",
        "The common response builder returns the result inline or through artifact URLs.",
    ], styles)

    story += [
        Paragraph("8. Architectural summary", styles["H1Custom"]),
        Paragraph(
            "The design separates natural-language orchestration from specialised analytical "
            "implementation. /ask-card is responsible for understanding and routing the request. "
            "Cards represent supported analytical intents. Tools contain the main logic and manage "
            "all access to data, models, registries and external services. A common response builder "
            "standardises the final output for the client.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "In one sentence: /ask-card interprets the request, selects one analytical card, "
            "delegates execution to a specialised tool that manages its own dependencies, and "
            "returns a standardised result containing text, tables and figures.",
            styles["SummaryCustom"],
        ),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    build_document()