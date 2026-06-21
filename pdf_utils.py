from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from datetime import datetime


def generate_incident_report(
    event_type,
    closure_risk,
    duration,
    severity,
    officers,
    barricades,
    diversion,
    map_path=None,
    filename="MITRA_Incident_Report.pdf",
):

    doc = SimpleDocTemplate(
        filename,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=24,
        leading=28
    )

    subtitle_style = ParagraphStyle(
        "SubTitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=11,
        textColor=colors.grey
    )

    heading_style = styles["Heading2"]

    elements = []

    # ====================================================
    # HEADER
    # ====================================================

    elements.append(
        Paragraph("MITRA", title_style)
    )

    elements.append(
        Paragraph(
            "Model-driven Insights for Traffic & Routing Assistance",
            subtitle_style
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            "Traffic Event Response Report",
            styles["Heading1"]
        )
    )

    elements.append(Spacer(1, 15))

    generated_time = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    elements.append(
        Paragraph(
            f"<b>Generated:</b> {generated_time}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    # ====================================================
    # EXECUTIVE SUMMARY
    # ====================================================

    elements.append(
        Paragraph(
            "Executive Summary",
            heading_style
        )
    )

    summary = f"""
    A traffic event classified as <b>{event_type}</b> has been analysed by
    the MITRA decision-support platform.

    The predicted road-closure risk is <b>{closure_risk}</b>,
    with an estimated clearance duration of <b>{duration}</b>.

    The event has been categorised as a
    <b>{severity}</b> severity incident and resource deployment
    recommendations have been generated.
    """

    elements.append(
        Paragraph(summary, styles["BodyText"])
    )

    elements.append(Spacer(1, 15))

    # ====================================================
    # RISK TABLE
    # ====================================================

    elements.append(
        Paragraph(
            "Risk Assessment",
            heading_style
        )
    )

    risk_data = [
        ["Metric", "Value"],
        ["Road Closure Risk", str(closure_risk)],
        ["Expected Clearance Time", str(duration)],
        ["Severity Tier", str(severity)],
    ]

    table = Table(risk_data, colWidths=[220, 220])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    elements.append(table)

    elements.append(Spacer(1, 20))

    # ====================================================
    # DEPLOYMENT PLAN
    # ====================================================

    elements.append(
        Paragraph(
            "Recommended Deployment Plan",
            heading_style
        )
    )

    deploy_data = [
        ["Resource", "Recommendation"],
        ["Police Officers", str(officers)],
        ["Barricades", str(barricades)],
        ["Traffic Diversion", str(diversion)],
    ]

    deploy_table = Table(
        deploy_data,
        colWidths=[220, 220]
    )

    deploy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkgreen),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    elements.append(deploy_table)

    elements.append(Spacer(1, 20))

    # ====================================================
    # MAP
    # ====================================================

    if map_path:
        try:
            elements.append(
                Paragraph(
                    "Event Location & Corridor Context",
                    heading_style
                )
            )

            elements.append(
                Image(
                    map_path,
                    width=5.5 * inch,
                    height=4 * inch
                )
            )

            elements.append(Spacer(1, 15))

        except Exception:
            pass

    # ====================================================
    # FOOTER TEXT
    # ====================================================

    elements.append(
        Paragraph(
            """
            This report was automatically generated by MITRA.
            Recommendations are intended to assist traffic
            management personnel and should be reviewed
            alongside operational judgement.
            """,
            styles["Italic"]
        )
    )

    doc.build(elements)

    return filename