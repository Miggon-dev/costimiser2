from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


OUTPUT_FILE = Path("backend_technology_stack.pdf")


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=25,
            textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SubtitleCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )

    styles.add(
        ParagraphStyle(
            name="H1Custom",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=7,
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.3,
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.5,
        )
    )

    return styles


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2F3"))
    canvas.line(18 * mm, 15 * mm, 192 * mm, 15 * mm)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(
        18 * mm,
        10 * mm,
        "Costimiser AI Analytic Engine - Backend Technology Stack",
    )
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def make_summary_table(styles):
    data = [
        ["Layer", "Technologies"],
        ["Application", "Python 3.11, Flask 3.0.3"],
        ["Analytics", "pandas, NumPy, SciPy, scikit-learn, SHAP"],
        [
            "RAG",
            "Amazon Bedrock, Amazon Titan Text Embeddings V2, FAISS CPU, "
            "Anthropic Claude Sonnet 4.5",
        ],
        ["Output", "PyArrow, Fastparquet, Plotly"],
        [
            "AWS",
            "boto3, s3fs, Amazon S3, AWS STS, Systems Manager Parameter Store, "
            "IAM Roles for Service Accounts",
        ],
        ["Deployment", "Docker, Amazon EKS, Helm 3"],
        ["Infrastructure and delivery", "Terraform 1.15.8, AWS CodeBuild"],
    ]

    formatted = []
    for row_index, row in enumerate(data):
        formatted.append([
            Paragraph(f"<b>{cell}</b>", styles["SmallCustom"]) if row_index == 0
            else Paragraph(str(cell), styles["SmallCustom"])
            for cell in row
        ])

    table = Table(formatted, colWidths=[42 * mm, 126 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E2F3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B4C6E7")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F7F9FC")],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


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
        Paragraph("Backend Technology Stack", styles["TitleCustom"]),
        Paragraph(
            "Costimiser AI Analytic Engine<br/>"
            "Concise description of the application, analytics, RAG, cloud, "
            "and deployment stack",
            styles["SubtitleCustom"],
        ),
        make_summary_table(styles),
        Spacer(1, 8),
    ]

    sections = [
        (
            "Application runtime",
            "The backend is implemented in <b>Python 3.11</b> and exposes its endpoints "
            "through <b>Flask 3.0.3</b>. The application is packaged as a Docker container "
            "and deployed to Amazon EKS.",
        ),
        (
            "Runtime consideration",
            "The current container starts the application with Flask's built-in server. "
            "For higher concurrency and stronger worker management, a dedicated WSGI "
            "server such as Gunicorn could be considered.",
        ),
        (
            "Application architecture",
            "The backend follows a layered structure with separate modules for API endpoints, "
            "orchestration, services, tools, configuration, and shared utilities. The "
            "<b>/ask-card</b> endpoint interprets natural-language requests, selects the "
            "appropriate analytical card, and delegates execution to the corresponding tool. "
            "Each tool contains the main analytical logic and manages access to its required "
            "dependencies.",
        ),
        (
            "Data and analytical processing",
            "The main data and numerical libraries are <b>pandas 2.3.3</b>, "
            "<b>NumPy 2.3.3</b>, and <b>SciPy</b>. They support process-data preparation, "
            "aggregation, statistical computation, scenario analysis, and optimization.",
        ),
        (
            "Machine learning and explainability",
            "The backend uses <b>scikit-learn 1.7.2</b> for prediction models and analytical "
            "pipelines, and <b>SHAP</b> for model explainability. These capabilities support "
            "prediction, diagnosis, scenario evaluation, recommendations, and optimization.",
        ),
        (
            "Retrieval-Augmented Generation",
            "The backend includes a RAG subsystem for papermaking knowledge retrieval and for "
            "enriching analytical recommendations. It uses <b>Amazon Bedrock</b>, "
            "<b>Amazon Titan Text Embeddings V2</b>, <b>FAISS CPU</b>, and "
            "<b>Anthropic Claude Sonnet 4.5</b>. Titan converts document chunks and user "
            "questions into embeddings, FAISS retrieves the most relevant papermaking content, "
            "and Claude generates the grounded answer or recommendation.",
        ),
        (
            "Data serialization and analytical output",
            "The backend uses <b>PyArrow</b>, <b>Fastparquet</b>, and "
            "<b>Plotly 5.24.1</b> to support tabular data exchange, downloadable analytical "
            "results, and interactive figures.",
        ),
        (
            "AWS integration",
            "The service uses <b>boto3</b>, <b>s3fs</b>, <b>Amazon S3</b>, "
            "<b>AWS STS</b>, <b>AWS Systems Manager Parameter Store</b>, and "
            "<b>IAM Roles for Service Accounts</b>. These services support data and model "
            "access, artifact handling, secure configuration, and AWS permissions.",
        ),
        (
            "MLflow status",
            "MLflow integration is planned but is not yet part of the operational backend stack.",
        ),
        (
            "Containerization and deployment",
            "The application is packaged with <b>Docker</b> and deployed to "
            "<b>Amazon EKS</b>. The Kubernetes resources are generated using <b>Helm 3</b>.",
        ),
        (
            "Infrastructure as code",
            "The backend infrastructure is managed with <b>Terraform 1.15.8</b>.",
        ),
        (
            "CI/CD",
            "The build and deployment process runs in <b>AWS CodeBuild</b>. The pipeline "
            "performs application validation, container build, security checks, infrastructure "
            "deployment, and rollout to EKS.",
        ),
        (
            "Security and quality controls",
            "The delivery process includes controls for static code analysis, Python style "
            "validation, dependency and container vulnerability scanning, Kubernetes manifest "
            "validation, code coverage, XML test reporting, non-root container execution, "
            "restricted container privileges, read-only container filesystems, and IAM-based "
            "access control.",
        ),
        (
            "Summary",
            "The backend is a Python 3.11 Flask analytical service that combines data processing, "
            "machine-learning inference, explainability, optimization, and Retrieval-Augmented "
            "Generation. The RAG subsystem uses Amazon Bedrock with Amazon Titan Text Embeddings "
            "V2, FAISS CPU, and Anthropic Claude Sonnet 4.5. The service is containerized with "
            "Docker, deployed to Amazon EKS using Helm, managed with Terraform, and delivered "
            "through AWS CodeBuild.",
        ),
    ]

    for title, body in sections:
        story.append(Paragraph(title, styles["H1Custom"]))
        story.append(Paragraph(body, styles["BodyCustom"]))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    build_document()