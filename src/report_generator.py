"""
report_generator.py - Generates professional Agronomic Diagnostic Field Reports in PDF format.
Utilizes ReportLab to compile leaf photos, primary diagnoses, differential candidates,
and comprehensive cultural/organic/chemical remedies.
"""

import io
import datetime
from typing import List, Dict, Any, Union
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable
)


def generate_pdf_report(
    predictions: List[Dict[str, Any]],
    leaf_image: Image.Image,
    model_name: str = "MobileNetV3 (TFLite Edge)",
    latency_ms: float = 0.0,
    specimen_name: str = "Leaf Specimen"
) -> bytes:
    """
    Compiles an agronomic diagnostic report into PDF bytes.

    Args:
        predictions: List of top-k prediction dictionaries from predict_top_k().
        leaf_image: PIL Image of the inspected leaf.
        model_name: Name of active deep learning model engine.
        latency_ms: Inference runtime in milliseconds.
        specimen_name: Identifier or filename of the leaf specimen.

    Returns:
        Bytes containing the generated PDF document.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    primary = predictions[0]
    info = primary["details"]
    is_healthy = "healthy" in primary["class_name"].lower()

    # Styling Palette
    c_primary = colors.HexColor("#1b5e20")      # Forest Green
    c_secondary = colors.HexColor("#2e7d32")    # Leaf Green
    c_dark = colors.HexColor("#1e293b")         # Slate Dark
    c_gray = colors.HexColor("#64748b")         # Slate Gray
    c_bg_light = colors.HexColor("#f8fafc")     # Light Slate Tint
    c_accent = colors.HexColor("#166534") if is_healthy else colors.HexColor("#b91c1c")
    c_badge_bg = colors.HexColor("#dcfce7") if is_healthy else colors.HexColor("#fee2e2")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=c_primary,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=c_gray,
        spaceAfter=10
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=c_dark
    )

    bold_label_style = ParagraphStyle(
        "BoldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=c_dark
    )

    badge_text_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=c_accent,
        alignment=1  # Centered
    )

    footer_style = ParagraphStyle(
        "FooterText",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=c_gray,
        alignment=1
    )

    elements = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("🌱 <b>AGRIVISION AI DIAGNOSTIC REPORT</b>", title_style),
            Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}<br/><b>Report ID:</b> AGR-{datetime.datetime.now().strftime('%Y%m%d%H%M')}", ParagraphStyle("MetaRight", fontName="Helvetica", fontSize=8, leading=11, textColor=c_gray, alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    elements.append(header_table)
    elements.append(Paragraph("Automated Phytopathology Diagnostics & Agronomic Management Advisory", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=0, spaceAfter=8))

    # 2. Specimen & Engine Metadata Strip
    meta_table_data = [
        [
            Paragraph(f"<b>Specimen:</b> {specimen_name[:35]}", body_style),
            Paragraph(f"<b>Inference Engine:</b> {model_name}", body_style),
            Paragraph(f"<b>Latency:</b> {latency_ms:.1f} ms" if latency_ms > 0 else "<b>Latency:</b> Real-time", body_style),
            Paragraph("<b>Dataset:</b> PlantVillage 38-Class", body_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[140, 160, 90, 150])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_bg_light),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # 3. Two-Column Diagnostic Section (Photo + Primary Diagnosis)
    # Convert PIL Image to in-memory bytes for ReportLab
    img_byte_arr = io.BytesIO()
    leaf_image.convert("RGB").save(img_byte_arr, format="JPEG", quality=90)
    img_byte_arr.seek(0)
    rl_img = RLImage(img_byte_arr, width=150, height=150)

    status_label = "HEALTHY SPECIMEN" if is_healthy else "DISEASE DETECTED"
    badge_table = Table([[Paragraph(f"● {status_label}", badge_text_style)]], colWidths=[160])
    badge_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_badge_bg),
        ('BOX', (0, 0), (-1, -1), 1, c_accent),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    diag_info_data = [
        [badge_table],
        [Paragraph(f"<font size=14 color='#1e293b'><b>{primary['class_name']}</b></font>", body_style)],
        [Paragraph(f"<b>Statistical Confidence:</b> <font color='#166534' size=11><b>{primary['confidence_percentage']:.2f}%</b></font>", body_style)],
        [Paragraph(f"<b>Host Crop:</b> {info['crop']}", body_style)],
        [Paragraph(f"<b>Pathogen Category:</b> {info['category']}", body_style)],
        [Paragraph("<b>Status Note:</b> Active symptoms identified on leaf surface." if not is_healthy else "<b>Status Note:</b> Leaf exhibits normal vigor and healthy chlorophyll density.", body_style)]
    ]
    diag_info_table = Table(diag_info_data, colWidths=[360])
    diag_info_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
    ]))

    primary_block = Table([[rl_img, diag_info_table]], colWidths=[160, 380])
    primary_block.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(primary_block)
    elements.append(Spacer(1, 10))

    # 4. Top Differential Candidates Table
    elements.append(Paragraph("📊 Differential Diagnosis (Top Probability Candidates)", heading_style))
    diff_table_data = [
        [
            Paragraph("<b>Rank</b>", bold_label_style),
            Paragraph("<b>Condition / Disease</b>", bold_label_style),
            Paragraph("<b>Host Crop</b>", bold_label_style),
            Paragraph("<b>Confidence</b>", bold_label_style)
        ]
    ]
    for r_idx, item in enumerate(predictions[:4], start=1):
        diff_table_data.append([
            Paragraph(f"#{r_idx}", body_style),
            Paragraph(f"<b>{item['class_name']}</b>" if r_idx == 1 else item['class_name'], body_style),
            Paragraph(item["details"]["crop"], body_style),
            Paragraph(f"{item['confidence_percentage']:.2f}%", bold_label_style if r_idx == 1 else body_style)
        ])

    diff_table = Table(diff_table_data, colWidths=[40, 260, 120, 120])
    diff_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(diff_table)
    elements.append(Spacer(1, 10))

    # 5. Detailed Agronomic Prescriptions
    elements.append(Paragraph("📋 Field Symptoms & Prescribed Treatments", heading_style))

    remedy_data = [
        [
            Paragraph("<b>🔍 Characteristic Symptoms:</b>", bold_label_style),
            Paragraph(info["symptoms"], body_style)
        ],
        [
            Paragraph("<b>🌿 Organic & Biological Controls:</b>", bold_label_style),
            Paragraph(info["organic_treatment"], body_style)
        ],
        [
            Paragraph("<b>🧪 Targeted Chemical Options:</b>", bold_label_style),
            Paragraph(info["chemical_treatment"], body_style)
        ],
        [
            Paragraph("<b>🛡️ Cultural Prevention Practices:</b>", bold_label_style),
            Paragraph(info["prevention"], body_style)
        ]
    ]

    remedy_table = Table(remedy_data, colWidths=[150, 390])
    remedy_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(remedy_table)
    elements.append(Spacer(1, 14))

    # 6. Disclaimer & Footer
    elements.append(HRFlowable(width="100%", thickness=0.5, color=c_gray, spaceBefore=4, spaceAfter=6))
    elements.append(Paragraph(
        "<b>Notice:</b> This diagnostic report was synthesized using artificial intelligence computer vision models. "
        "It provides field guidance and phytopathological suggestions. Always adhere to chemical manufacturer label "
        "guidelines and local government agricultural extension advisories before administering pesticide or fungicide interventions.",
        footer_style
    ))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
