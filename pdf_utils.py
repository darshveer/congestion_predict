from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet

def generate_incident_report(
    event_type,
    closure_risk,
    duration,
    severity,
    officers,
    barricades,
    diversion,
    filename="MITRA_Incident_Report.pdf",
):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("MITRA Incident Response Report",
                  styles["Title"])
    )

    elements.append(Spacer(1, 12))

    fields = {
        "Event Type": event_type,
        "Closure Risk": closure_risk,
        "Expected Duration (mins)": duration,
        "Severity": severity,
        "Recommended Officers": officers,
        "Recommended Barricades": barricades,
        "Diversion Required": diversion,
    }

    for key, value in fields.items():
        elements.append(
            Paragraph(
                f"<b>{key}</b>: {value}",
                styles["BodyText"]
            )
        )

        elements.append(Spacer(1, 6))

    doc.build(elements)

    return filename