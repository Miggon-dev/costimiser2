from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


OUTPUT_FILE = Path("product_cost_quality_scoring.pdf")


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
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=8,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.6,
    ))

    styles.add(ParagraphStyle(
        name="FormulaCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=12.5,
        leading=17,
        alignment=TA_CENTER,
        spaceBefore=5,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="CalloutCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.1,
        leading=13.1,
        backColor=colors.HexColor("#EAF2F8"),
        borderColor=colors.HexColor("#5B9BD5"),
        borderWidth=0.8,
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=9,
    ))

    styles.add(ParagraphStyle(
        name="WarningCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.0,
        leading=13,
        backColor=colors.HexColor("#FFF2CC"),
        borderColor=colors.HexColor("#D6B656"),
        borderWidth=0.6,
        borderPadding=8,
        spaceBefore=5,
        spaceAfter=6,
    ))

    return styles


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(
        18 * mm,
        10 * mm,
        "Product Cost-Quality Scoring Framework",
    )
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
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
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B4C6E7")),
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


def build_document():
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
    )

    story = [
        Paragraph("Product Cost-Quality Scoring Framework", styles["TitleCustom"]),
        Paragraph(
            "A practical method to rank products so that low cost and high quality receive "
            "the best scores, while products below specification - especially expensive ones - "
            "receive the lowest scores.",
            styles["SubtitleCustom"],
        ),
        Paragraph(
            "<b>Recommended design</b><br/>"
            "Combine a normalized cost score with a capped quality score, then apply a strong "
            "penalty whenever quality falls below the minimum specification.",
            styles["CalloutCustom"],
        ),

        Paragraph("1. Variables", styles["H1Custom"]),
    ]

    variables = [
        ["Symbol", "Meaning"],
        ["c<sub>i</sub>", "Cost of product i"],
        ["q<sub>i</sub>", "Quality of product i"],
        ["q<sub>min</sub>", "Minimum acceptable quality specification"],
        [
            "q<sub>target</sub>",
            "Quality level above which extra quality receives no additional score",
        ],
        [
            "c<sub>min</sub>, c<sub>max</sub>",
            "Reference minimum and maximum cost values",
        ],
    ]
    story.append(make_table(variables, [38 * mm, 130 * mm], styles))

    story += [
        Paragraph("2. Cost score", styles["H1Custom"]),
        Paragraph(
            "Normalize cost so that the cheapest product receives 1 and the most expensive receives 0:",
            styles["BodyCustom"],
        ),
        Paragraph(
            "CostScore<sub>i</sub> = "
            "(c<sub>max</sub> - c<sub>i</sub>) / "
            "(c<sub>max</sub> - c<sub>min</sub>)",
            styles["FormulaCustom"],
        ),
        Paragraph(
            "Lower cost therefore always improves the score.",
            styles["BodyCustom"],
        ),

        Paragraph("3. Quality score", styles["H1Custom"]),
        Paragraph(
            "Reward quality above specification, but cap the benefit at a meaningful target. "
            "This prevents excessive quality or overprocessing from being rewarded indefinitely.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "QualityScore<sub>i</sub> = clip["
            "(q<sub>i</sub> - q<sub>min</sub>) / "
            "(q<sub>target</sub> - q<sub>min</sub>), 0, 1]",
            styles["FormulaCustom"],
        ),
    ]

    quality_table = [
        ["Quality position", "Quality score"],
        [
            "Below q<sub>min</sub>",
            "0 before applying the out-of-specification penalty",
        ],
        ["At q<sub>min</sub>", "0"],
        [
            "Between q<sub>min</sub> and q<sub>target</sub>",
            "Increases linearly from 0 to 1",
        ],
        ["At or above q<sub>target</sub>", "1"],
    ]
    story.append(
        make_table(
            quality_table,
            [62 * mm, 106 * mm],
            styles,
            header_color="#E2F0D9",
        )
    )

    story += [
        Paragraph("4. Out-of-specification penalty", styles["H1Custom"]),
        Paragraph(
            "Meeting specification is normally non-negotiable. A separate penalty ensures "
            "that a cheap but non-compliant product does not outrank a compliant product.",
            styles["BodyCustom"],
        ),
        Paragraph(
            "Penalty<sub>i</sub> = 0, if q<sub>i</sub> >= q<sub>min</sub>",
            styles["FormulaCustom"],
        ),
        Paragraph(
            "Penalty<sub>i</sub> = P<sub>0</sub> + "
            "P<sub>1</sub>(q<sub>min</sub> - q<sub>i</sub>) / "
            "q<sub>min</sub>, if q<sub>i</sub> &lt; q<sub>min</sub>",
            styles["FormulaCustom"],
        ),
        Paragraph(
            "P<sub>0</sub> is a fixed penalty for any specification failure. "
            "P<sub>1</sub> increases the penalty according to the severity of the quality shortfall.",
            styles["BodyCustom"],
        ),

        Paragraph("5. Final recommended score", styles["H1Custom"]),
        Paragraph(
            "Using equal weights as a starting point:",
            styles["BodyCustom"],
        ),
        Paragraph(
            "<b>Score<sub>i</sub> = 50 x CostScore<sub>i</sub> + "
            "50 x QualityScore<sub>i</sub> - Penalty<sub>i</sub></b>",
            styles["CalloutCustom"],
        ),
        Paragraph(
            "The weights can be adjusted. For example, use 60% quality and 40% cost when "
            "quality should dominate, or 60% cost and 40% quality when all evaluated products "
            "are already reliably compliant.",
            styles["BodyCustom"],
        ),

        Paragraph("6. Expected ranking behaviour", styles["H1Custom"]),
    ]

    ranking = [
        ["Product profile", "Expected score"],
        ["High quality, low cost", "Highest"],
        ["Acceptable quality, low cost", "High"],
        ["High quality, high cost", "Intermediate"],
        ["Below-specification quality, low cost", "Low"],
        ["Below-specification quality, high cost", "Lowest"],
    ]
    story.append(make_table(ranking, [112 * mm, 56 * mm], styles))

    story += [
        Paragraph("7. Suggested starting parameters", styles["H1Custom"]),
    ]

    parameters = [
        ["Parameter", "Starting recommendation"],
        ["Cost weight", "50"],
        ["Quality weight", "50"],
        [
            "q<sub>target</sub>",
            "A meaningful quality target above specification, not the observed maximum",
        ],
        [
            "P<sub>0</sub>",
            "50 points, making any specification breach immediately visible",
        ],
        [
            "P<sub>1</sub>",
            "50 points, distinguishing minor from severe quality failures",
        ],
        [
            "Cost reference range",
            "Prefer robust limits such as the 5th and 95th percentiles rather than raw extremes",
        ],
    ]
    story.append(
        make_table(
            parameters,
            [50 * mm, 118 * mm],
            styles,
            header_color="#FCE4D6",
        )
    )

    story += [
        Paragraph("8. Important implementation note", styles["H1Custom"]),
        Paragraph(
            "Use stable reference values for normalization. If c<sub>min</sub> and "
            "c<sub>max</sub> are recalculated from every small batch, the same product may "
            "receive a different score depending on the comparison set. For operational use, "
            "define the cost limits from a sufficiently long historical period and review them periodically.",
            styles["WarningCustom"],
        ),
        Paragraph(
            "<b>Interpretation:</b> The resulting score is a ranking index, not a physical "
            "measurement. Its parameters should be validated against business priorities and "
            "known examples of good and bad products.",
            styles["BodyCustom"],
        ),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    build_document()