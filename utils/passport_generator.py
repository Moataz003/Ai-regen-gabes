"""
passport_generator.py ─── Gabès Regenerate AI ─── Soil Passport Export
"""
import io
import json
import qrcode
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# QR Code Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_qr_bytes(zone_id: str, passport_data: dict, farmer_name: str) -> bytes:
    """Generate QR code containing passport verification data."""
    qr_data = {
        "zone_id": zone_id,
        "farmer": farmer_name,
        "timestamp": datetime.now().isoformat(),
        "cd_mg_kg": passport_data.get("input", {}).get("cd_mg_kg", 0),
        "pb_mg_kg": passport_data.get("input", {}).get("pb_mg_kg", 0),
        "zone_color": passport_data.get("zone_color", "UNKNOWN"),
    }
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(json.dumps(qr_data))
    qr.make(fit=True)

    img = qr.make_image(fill_color="#2E7D32", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# PDF Passport Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_passport_pdf(zone_id: str, passport_data: dict, farmer_name: str, qr_bytes: bytes = None) -> bytes:
    """Generate PDF soil passport certificate with embedded QR code."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, cm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        return None

    # ── FIX: Properly access nested input data ───────────────────────────────
    inp = passport_data.get("input", passport_data)

    ec_val = inp.get("ec_ds_m", inp.get("ec", 0))
    ph_val = inp.get("ph", 7.0)
    cd_val = inp.get("cd_mg_kg", inp.get("cd", 0))
    pb_val = inp.get("pb_mg_kg", inp.get("pb", 0))
    zn_val = inp.get("zn_mg_kg", 0)
    dist_val = inp.get("dist_km", 0)

    zone_color = passport_data.get("zone_color", "GREEN")
    months_to_safe = passport_data.get("planting_cycles", 0) * 6
    plant_mix = passport_data.get("plant_mix", "— ")
    safe_for_fodder = passport_data.get("safe_for_fodder", zone_color != "RED")

    # ── Document Setup ───────────────────────────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("CustomTitle", parent=styles["Heading1"], fontSize=18, textColor="#1B5E20", alignment=TA_CENTER, spaceAfter=12)
    subtitle_style = ParagraphStyle("CustomSubtitle", parent=styles["Normal"], fontSize=10, textColor="#666666", alignment=TA_CENTER, spaceAfter=20)
    section_style = ParagraphStyle("CustomSection", parent=styles["Heading2"], fontSize=12, textColor="#2E7D32", spaceAfter=8, spaceBefore=12)

    # ── Header ───────────────────────────────────────────────────────────────
    elements.append(Paragraph("SOIL PASSPORT — GABÈS REGENERATE AI", title_style))
    elements.append(Paragraph(f"Zone ID: {zone_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))

    # ── Zone Status Badge ────────────────────────────────────────────────────
    zone_labels = {
        "GREEN": ("REGENERATIVE CERTIFIED", "#2E7D32", "#E8F5E9"),
        "ORANGE": ("MONITORING REQUIRED", "#E65100", "#FFF3E0"),
        "RED": ("TOXIC BIOMASS", "#B71C1C", "#FFEBEE"),
    }
    zone_txt, zone_fg, zone_bg = zone_labels.get(zone_color, ("UNKNOWN", "#666", "#EEE"))

    status_table = Table([[zone_txt]], colWidths=[4*inch])
    status_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(zone_bg)),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(zone_fg)),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    elements.append(status_table)
    elements.append(Spacer(1, 0.3*inch))

    # ── Farmer Info ──────────────────────────────────────────────────────────
    elements.append(Paragraph("FARMER INFORMATION", section_style))
    farmer_data = [
        ["Farmer Name", farmer_name],
        ["Zone Coordinates", f"{passport_data.get('lat', 'N/A')}, {passport_data.get('lon', 'N/A')}"],
        ["Distance from Factory", f"{dist_val} km"],
    ]
    farmer_table = Table(farmer_data, colWidths=[2*inch, 3*inch])
    farmer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F5F5")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
    ]))
    elements.append(farmer_table)
    elements.append(Spacer(1, 0.3*inch))

    # ── Soil Analysis ────────────────────────────────────────────────────────
    elements.append(Paragraph("SOIL ANALYSIS RESULTS", section_style))

    CD_SAFE = 1.0
    CD_EU = 3.0
    PB_MOD = 50
    PB_HIGH = 85
    EC_STRESS = 6.0

    def get_status(value, threshold_high, threshold_low=None):
        if threshold_low and value < threshold_low:
            return "SAFE", "#2E7D32"
        if value > threshold_high:
            return "EXCEEDS LIMIT", "#B71C1C"
        return "MODERATE", "#E65100"

    cd_status, cd_color = get_status(cd_val, CD_EU, CD_SAFE)
    pb_status, pb_color = get_status(pb_val, PB_HIGH, PB_MOD)
    ec_status, ec_color = get_status(ec_val, EC_STRESS)

    soil_data = [
        ["Parameter", "Measured Value", "Threshold", "Status"],
        ["Cadmium (Cd)", f"{cd_val:.3f} mg/kg", f"< {CD_SAFE} safe, > {CD_EU} limit", cd_status],
        ["Lead (Pb)", f"{pb_val:.1f} mg/kg", f"< {PB_MOD} safe, > {PB_HIGH} high risk", pb_status],
        ["Zinc (Zn)", f"{zn_val:.1f} mg/kg", "—", "—"],
        ["Electrical Conductivity (EC)", f"{ec_val:.1f} dS/m", f"< {EC_STRESS} dS/m (crops)", ec_status],
        ["pH Level", f"{ph_val:.1f} ", "6.5–7.5 optimal", "—"],
    ]

    soil_table = Table(soil_data, colWidths=[2*inch, 1.5*inch, 2*inch, 1.5*inch])
    soil_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
        ("BACKGROUND", (3, 1), (3, 1), colors.HexColor(cd_color)),
        ("TEXTCOLOR", (3, 1), (3, 1), colors.white),
        ("BACKGROUND", (3, 2), (3, 2), colors.HexColor(pb_color)),
        ("TEXTCOLOR", (3, 2), (3, 2), colors.white),
        ("BACKGROUND", (3, 4), (3, 4), colors.HexColor(ec_color)),
        ("TEXTCOLOR", (3, 4), (3, 4), colors.white),
    ]))
    elements.append(soil_table)
    elements.append(Spacer(1, 0.3*inch))

    # ── Prescription ─────────────────────────────────────────────────────────
    elements.append(Paragraph("REMEDIATION PRESCRIPTION", section_style))
    rx_data = [
        ["Plant Mix", str(plant_mix)[:80]],
        ["Safe for Fodder", "Yes" if safe_for_fodder else "No"],
        ["Months to Safe Zone", f"{months_to_safe} months" if months_to_safe > 0 else "Already safe"],
        ["Planting Cycles", f"{months_to_safe // 6}" if months_to_safe > 0 else "N/A"],
    ]
    rx_table = Table(rx_data, colWidths=[2*inch, 4*inch])
    rx_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F5F5")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDDDDD")),
    ]))
    elements.append(rx_table)
    elements.append(Spacer(1, 0.2*inch))

    # ── Embedded QR Code ─────────────────────────────────────────────────────
    if qr_bytes:
        try:
            qr_image = Image(io.BytesIO(qr_bytes), width=1.8*inch, height=1.8*inch)
            qr_image.hAlign = 'CENTER'
            elements.append(qr_image)
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph("Scan to verify passport details", subtitle_style))
        except Exception as e:
            elements.append(Paragraph(f"QR code embedding failed: {str(e)}", subtitle_style))

    # ── Footer ───────────────────────────────────────────────────────────────
    footer_text = f"""
    <para alignment="center">
    <b>Gabès Regenerate AI · SoilRevive Engine</b><br/>
    This passport is verifiable via QR code. Valid for market access certification.<br/>
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </para>
    """
    elements.append(Paragraph(footer_text, subtitle_style))

    # Build PDF
    doc.build(elements)
    return buffer.getvalue()