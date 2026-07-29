from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle


OUTPUT_FILE = Path("pareto_cost_quality_scoring.pdf")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18.5,
        leading=23,
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
        spaceAfter=12,
    ))

    styles.add(ParagraphStyle(
        name="H1Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13.2,
        leading=16.5,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=7,
        spaceAfter=4,
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.1,
        leading=13,
        spaceAfter=4.5,
    ))

    styles.add(ParagraphStyle(
        name="SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.1,
        leading=10.4,
    ))

    styles.add(ParagraphStyle(
        name="FormulaCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11.5,
        leading=16,
        alignment=TA_CENTER,
        spaceBefore=3,
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="CalloutCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.0,
        leading=13,
        backColor=colors.HexColor("#EAF2F8"),
        borderColor=colors.HexColor("#5B9BD5"),
        borderWidth=0.8,
        borderPadding=7,
        spaceBefore=3,
        spaceAfter=7,
    ))

    styles.add(ParagraphStyle(
        name="WarningCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8.9,
        leading=12.8,
        backColor=colors.HexColor("#FFF2CC"),
        borderColor=colors.HexColor("#D6B656"),
        borderWidth=0.6,
        borderPadding=7,
        spaceBefore=4,
        spaceAfter=5,
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
        "Cost-Quality Ranking using Multiobjective Optimisation",
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
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_document():
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=20 * mm,
    )

    story = [
        Paragraph(
            "Cost-Quality Ranking using Multiobjective<br/>Optimisation",
            styles["TitleCustom"],
        ),
        Paragraph(
            "A normalization-free framework to identify efficient products: low cost, high quality, "
            "and strict handling of minimum quality specifications.",
            styles["SubtitleCustom"],
        ),
        Paragraph(
            "<b>Recommended idea:</b> treat quality specification as a constraint, then rank "
            "products using Pareto dominance. This avoids forcing cost and quality into one arbitrary normalized scale.",
            styles["CalloutCustom"],
        ),

        Paragraph("1. Problem formulation", styles["H1Custom"]),
        Paragraph(
            "For each product i, you observe cost c<sub>i</sub> and quality q<sub>i</sub>. "
            "The natural multiobjective problem is:",
            styles["BodyCustom"],
        ),
        Paragraph("minimize c<sub>i</sub>", styles["FormulaCustom"]),
        Paragraph("maximize q<sub>i</sub>", styles["FormulaCustom"]),
        Paragraph("subject to q<sub>i</sub> &gt;= q<sub>min</sub>", styles["FormulaCustom"]),
        Paragraph(
            "This formulation keeps the original units. Cost is cost, quality is quality, and "
            "the minimum specification is not treated as something that can be compensated by low cost.",
            styles["BodyCustom"],
        ),

        Paragraph("2. Feasibility first", styles["H1Custom"]),
        Paragraph(
            "Separate products into two groups before ranking:",
            styles["BodyCustom"],
        ),
    ]

    feasibility = [
        ["Group", "Condition", "Treatment"],
        ["Compliant", "q<sub>i</sub> &gt;= q<sub>min</sub>", "Rank using low cost and high quality."],
        [
            "Non-compliant",
            "q<sub>i</sub> &lt; q<sub>min</sub>",
            "Place below compliant products; rank by quality shortfall and cost.",
        ],
    ]
    story.append(make_table(feasibility, [38 * mm, 48 * mm, 82 * mm], styles))

    story += [
        Paragraph("3. Pareto dominance", styles["H1Custom"]),
        Paragraph(
            "For compliant products, product A dominates product B when A is no more expensive "
            "and no worse in quality, with at least one strict improvement:",
            styles["BodyCustom"],
        ),
        Paragraph(
            "c<sub>A</sub> &lt;= c<sub>B</sub> and q<sub>A</sub> &gt;= q<sub>B</sub>",
            styles["FormulaCustom"],
        ),
        Paragraph(
            "The Pareto-optimal products are those not dominated by any other product. "
            "They represent the efficient cost-quality trade-offs.",
            styles["BodyCustom"],
        ),

        Paragraph("4. Example interpretation", styles["H1Custom"]),
    ]

    example = [
        ["Product", "Cost", "Quality", "Interpretation"],
        ["A", "100", "2.10", "Pareto candidate"],
        ["B", "105", "2.20", "Pareto candidate"],
        ["C", "110", "2.05", "Dominated by A: higher cost and lower quality"],
        ["D", "95", "1.90", "Below specification, therefore infeasible"],
    ]
    story.append(
        make_table(
            example,
            [28 * mm, 30 * mm, 35 * mm, 75 * mm],
            styles,
            header_color="#E2F0D9",
        )
    )

    story += [
        Paragraph("5. From Pareto set to ranking", styles["H1Custom"]),
        Paragraph(
            "Pareto optimisation gives a set of efficient products, not automatically a single "
            "best product. If you need an ordered ranking, use non-dominated sorting:",
            styles["BodyCustom"],
        ),
    ]

    fronts = [
        ["Layer", "Meaning", "Priority"],
        [
            "Pareto front 1",
            "Products not dominated by any other compliant product.",
            "Best group",
        ],
        [
            "Pareto front 2",
            "Products dominated only by products in front 1.",
            "Second group",
        ],
        [
            "Pareto front 3+",
            "Progressively less efficient products.",
            "Lower groups",
        ],
    ]
    story.append(
        make_table(
            fronts,
            [42 * mm, 94 * mm, 32 * mm],
            styles,
            header_color="#FCE4D6",
        )
    )
    story += [
        Paragraph(
            "Within each Pareto front, apply a business tie-breaker: cost-first if quality above "
            "specification has limited value, or quality-margin-first if extra quality is valuable.",
            styles["BodyCustom"],
        ),

        Paragraph("6. Non-compliant products", styles["H1Custom"]),
        Paragraph(
            "For products below specification, define the quality shortfall:",
            styles["BodyCustom"],
        ),
        Paragraph(
            "s<sub>i</sub> = q<sub>min</sub> - q<sub>i</sub>",
            styles["FormulaCustom"],
        ),
        Paragraph(
            "Then rank non-compliant products by minimizing both shortfall and cost:",
            styles["BodyCustom"],
        ),
        Paragraph(
            "minimize s<sub>i</sub> and minimize c<sub>i</sub>",
            styles["FormulaCustom"],
        ),
        Paragraph(
            "This ensures the worst products are those with both large quality failure and high cost.",
            styles["BodyCustom"],
        ),

        Paragraph("7. Recommended operational rule", styles["H1Custom"]),
    ]

    rules = [
        ["Step", "Rule"],
        ["1", "Separate compliant and non-compliant products using q<sub>i</sub> &gt;= q<sub>min</sub>."],
        ["2", "For compliant products, compute Pareto fronts using cost minimisation and quality maximisation."],
        ["3", "Rank front 1 above front 2, front 2 above front 3, and so on."],
        ["4", "Within each front, use a clear tie-breaker such as lower cost first."],
        ["5", "Place all non-compliant products below compliant products."],
        ["6", "For non-compliant products, rank by quality shortfall first, then cost."],
    ]
    story.append(
        make_table(
            rules,
            [22 * mm, 146 * mm],
            styles,
            header_color="#E2F0D9",
        )
    )

    story.append(
        Paragraph(
            "<b>Important note:</b> This gives an ordinal decision ranking, not a smooth economic "
            "score. That is often an advantage: it avoids arbitrary weights and makes clear which "
            "products are objectively dominated.",
            styles["WarningCustom"],
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    build_document()

