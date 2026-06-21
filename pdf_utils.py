"""PDF report generator for MITRA incident reports.

Generates a professional single-page (or two-page with map) traffic-event
response report using ReportLab.  All cell content is wrapped in Paragraph
objects to prevent text overflow.
"""
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from datetime import datetime

# ── page geometry ──────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 40
CONTENT_W = PAGE_W - 2 * MARGIN          # usable width

# ── colour palette ─────────────────────────────────────────────────────
BRAND_DARK   = colors.HexColor("#1a2332")
BRAND_ACCENT = colors.HexColor("#2563eb")
HEADER_BG    = colors.HexColor("#0f172a")
HEADER_FG    = colors.white
ROW_ALT      = colors.HexColor("#f8fafc")
ROW_WHITE    = colors.white
GRID_CLR     = colors.HexColor("#cbd5e1")
LIGHT_BLUE   = colors.HexColor("#eff6ff")
GREEN_BG     = colors.HexColor("#14532d")
SEVERITY_CLR = {
    "Critical": colors.HexColor("#dc2626"),
    "High":     colors.HexColor("#ea580c"),
    "Moderate": colors.HexColor("#ca8a04"),
    "Low":      colors.HexColor("#16a34a"),
}


def _cell(text, style):
    """Wrap plain text in a Paragraph so it word-wraps inside table cells."""
    return Paragraph(str(text), style)


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
        pagesize=A4,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    styles = getSampleStyleSheet()

    # ── custom styles ──────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=28,
        leading=34,
        textColor=BRAND_DARK,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "SubTitleCustom",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=4,
    )
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=BRAND_DARK,
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6,
    )
    cell_label = ParagraphStyle(
        "CellLabel",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
        fontName="Helvetica-Bold",
    )
    cell_value = ParagraphStyle(
        "CellValue",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
    )
    cell_header = ParagraphStyle(
        "CellHeader",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=HEADER_FG,
        fontName="Helvetica-Bold",
    )
    footer_style = ParagraphStyle(
        "FooterCustom",
        parent=styles["Italic"],
        fontSize=7,
        leading=10,
        textColor=colors.HexColor("#94a3b8"),
        alignment=TA_CENTER,
    )

    elements = []

    # ═══════════════════════════════════════════════════════════════════
    # HEADER
    # ═══════════════════════════════════════════════════════════════════
    elements.append(Paragraph("MITRA", title_style))
    elements.append(
        Paragraph(
            "Model-driven Insights for Traffic &amp; Routing Assistance",
            subtitle_style,
        )
    )
    elements.append(
        HRFlowable(
            width="100%", thickness=1.5,
            color=BRAND_ACCENT, spaceBefore=4, spaceAfter=8,
        )
    )

    # Report title + timestamp
    elements.append(
        Paragraph("Traffic Event Response Report", styles["Heading1"])
    )
    generated_time = datetime.now().strftime("%d %B %Y, %H:%M")
    elements.append(
        Paragraph(
            f"<b>Generated:</b> {generated_time}",
            body_style,
        )
    )
    elements.append(Spacer(1, 8))

    # ═══════════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    elements.append(Paragraph("Executive Summary", heading_style))

    sev_color = SEVERITY_CLR.get(severity, colors.grey)
    sev_hex = f"#{sev_color.hexval()[2:]}" if hasattr(sev_color, 'hexval') else "#64748b"

    summary = (
        f'A traffic event classified as <b>{event_type}</b> has been analysed '
        f'by the MITRA decision-support platform. '
        f'The predicted road-closure risk is <b>{closure_risk}</b>, '
        f'with an estimated clearance duration of <b>{duration}</b>. '
        f'The event has been categorised as '
        f'<font color="{sev_hex}"><b>{severity}</b></font> severity '
        f'and resource deployment recommendations have been generated.'
    )
    elements.append(Paragraph(summary, body_style))
    elements.append(Spacer(1, 6))

    # ═══════════════════════════════════════════════════════════════════
    # RISK ASSESSMENT TABLE
    # ═══════════════════════════════════════════════════════════════════
    elements.append(Paragraph("Risk Assessment", heading_style))

    col_w = [CONTENT_W * 0.40, CONTENT_W * 0.60]
    risk_data = [
        [_cell("Metric", cell_header), _cell("Value", cell_header)],
        [_cell("Event Type", cell_label), _cell(str(event_type), cell_value)],
        [_cell("Road Closure Risk", cell_label), _cell(str(closure_risk), cell_value)],
        [_cell("Expected Clearance Time", cell_label), _cell(str(duration), cell_value)],
        [_cell("Severity Tier", cell_label),
         _cell(f'<font color="{sev_hex}"><b>{severity}</b></font>', cell_value)],
    ]
    table = Table(risk_data, colWidths=col_w)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("BACKGROUND", (0, 1), (-1, 1), ROW_WHITE),
        ("BACKGROUND", (0, 2), (-1, 2), ROW_ALT),
        ("BACKGROUND", (0, 3), (-1, 3), ROW_WHITE),
        ("BACKGROUND", (0, 4), (-1, 4), ROW_ALT),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID_CLR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 10))

    # ═══════════════════════════════════════════════════════════════════
    # DEPLOYMENT PLAN TABLE
    # ═══════════════════════════════════════════════════════════════════
    elements.append(Paragraph("Recommended Deployment Plan", heading_style))

    deploy_data = [
        [_cell("Resource", cell_header), _cell("Recommendation", cell_header)],
        [_cell("Police Officers", cell_label), _cell(str(officers), cell_value)],
        [_cell("Barricades", cell_label), _cell(str(barricades), cell_value)],
        [_cell("Traffic Diversion", cell_label), _cell(str(diversion), cell_value)],
    ]
    deploy_table = Table(deploy_data, colWidths=col_w)
    deploy_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREEN_BG),
        ("BACKGROUND", (0, 1), (-1, 1), ROW_WHITE),
        ("BACKGROUND", (0, 2), (-1, 2), ROW_ALT),
        ("BACKGROUND", (0, 3), (-1, 3), ROW_WHITE),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID_CLR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(deploy_table)
    elements.append(Spacer(1, 10))

    # ═══════════════════════════════════════════════════════════════════
    # MAP — Event Location & Corridor Context
    # ═══════════════════════════════════════════════════════════════════
    if map_path:
        try:
            map_section = []
            map_section.append(
                Paragraph("Event Location &amp; Corridor Context", heading_style)
            )
            # Scale image to fill content width while keeping aspect ratio
            img_w = CONTENT_W
            img_h = img_w * 0.65      # ~3:2 aspect ratio
            map_section.append(
                Image(map_path, width=img_w, height=img_h)
            )
            map_section.append(Spacer(1, 6))
            map_section.append(
                Paragraph(
                    "<i>Map shows the incident location (red marker) on "
                    "OpenStreetMap tiles. Road network and nearby landmarks "
                    "are visible for context.</i>",
                    ParagraphStyle(
                        "MapCaption", parent=body_style,
                        fontSize=7, textColor=colors.HexColor("#94a3b8"),
                        alignment=TA_CENTER,
                    ),
                )
            )
            elements.extend(map_section)
        except Exception:
            pass

    elements.append(Spacer(1, 12))

    # ═══════════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════════
    elements.append(
        HRFlowable(
            width="100%", thickness=0.5,
            color=GRID_CLR, spaceBefore=6, spaceAfter=6,
        )
    )
    elements.append(
        Paragraph(
            "This report was automatically generated by MITRA. "
            "Recommendations are intended to assist traffic management "
            "personnel and should be reviewed alongside operational judgement.",
            footer_style,
        )
    )

    doc.build(elements)
    return filename