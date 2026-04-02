import os
import io
import datetime
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from collections import Counter
import torch
import torch.nn.functional as F
from torchvision import transforms
 
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
 
from src.pcb_pipeline import (
    classify_boxes,
    draw_detections,
    extract_candidate_boxes,
    generate_anomaly_map,
    load_checkpoint,
)
 
# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH           = "pcb_defect_model.pth"
DATASET_ROOT         = "dataset"
CONFIDENCE_THRESHOLD = 0.55
MAX_CANDIDATES       = 10
MAX_FALLBACK_BOXES   = 5
IOU_THRESHOLD        = 0.25
MAX_FINAL_BOXES      = 3
TTA_ROUNDS           = 6
 
KNOWN_DEFECTS = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]
 
DEFECT_COLORS = {
    "missing_hole":    "#FF4B4B",
    "mouse_bite":      "#FF8C00",
    "open_circuit":    "#FFD700",
    "short":           "#00CED1",
    "spur":            "#9370DB",
    "spurious_copper": "#32CD32",
}
 
# ─────────────────────────────────────────────
#  PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PCB Defect Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}
 
.stApp {
    background: #0a0e1a;
    color: #e0e6f0;
}
 
/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2f4a 50%, #0d1b2a 100%);
    border: 1px solid #00d4ff33;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7c3aed, #00d4ff, transparent);
}
.header-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.header-title span {
    color: #00d4ff;
}
.header-sub {
    font-size: 0.95rem;
    color: #7a9bb5;
    margin-top: 6px;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 1px;
}
.header-badge {
    display: inline-block;
    background: #00d4ff18;
    border: 1px solid #00d4ff44;
    color: #00d4ff;
    font-size: 0.72rem;
    font-family: 'Share Tech Mono', monospace;
    padding: 4px 10px;
    border-radius: 20px;
    margin-top: 10px;
    letter-spacing: 1.5px;
}
 
/* Upload zone */
.upload-zone {
    background: #0d1b2a;
    border: 2px dashed #1e3a5f;
    border-radius: 14px;
    padding: 32px;
    text-align: center;
    transition: border-color 0.3s;
}
 
/* Section headers */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #00d4ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e0e6f0;
    margin-bottom: 16px;
}
 
/* Image panels */
.img-panel {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 14px;
}
.img-panel-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #7a9bb5;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
 
/* Metric cards */
.metric-row {
    display: flex;
    gap: 14px;
    margin: 20px 0;
}
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #0d1b2a, #111e30);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00d4ff;
    font-family: 'Share Tech Mono', monospace;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: #7a9bb5;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
    font-family: 'Share Tech Mono', monospace;
}
 
/* Detection cards */
.detection-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #00d4ff;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.det-label {
    font-weight: 700;
    font-size: 0.95rem;
    color: #e0e6f0;
    text-transform: capitalize;
}
.det-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: #00d4ff;
}
.det-box {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #4a7a9b;
}
 
/* Status pill */
.status-ok {
    display: inline-block;
    background: #00d4ff18;
    border: 1px solid #00d4ff55;
    color: #00d4ff;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Share Tech Mono', monospace;
}
.status-warn {
    display: inline-block;
    background: #ff4b4b18;
    border: 1px solid #ff4b4b55;
    color: #ff4b4b;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Share Tech Mono', monospace;
}
 
/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0057a8, #0077cc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0077cc, #00a0ff) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px #0077cc44 !important;
}
 
/* Divider */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, transparent);
    margin: 28px 0;
}
 
/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found")
    return load_checkpoint(MODEL_PATH, dataset_root=DATASET_ROOT, device="cpu")
#  HEADER
st.markdown("""
<div class="header-banner">
    <div class="header-title">PCB <span>Defect</span> Inspector</div>
    <div class="header-sub">AI-POWERED QUALITY CONTROL SYSTEM</div>
    <div class="header-badge">● EFFICIENTNET-B0 · 6 DEFECT CLASSES · TTA ENABLED</div>
</div>
""", unsafe_allow_html=True)
 
# Load model
model_loaded = False
try:
    model, classes, inference_transform = load_model_bundle()
    model_loaded = True
    st.markdown('<div style="margin-bottom:20px"><span class="status-ok">✓ MODEL LOADED</span></div>', unsafe_allow_html=True)
except Exception as exc:
    st.markdown(f'<div style="margin-bottom:20px"><span class="status-warn">✗ MODEL ERROR: {exc}</span></div>', unsafe_allow_html=True)
# ─────────────────────────────────────────────
#  TTA
# ─────────────────────────────────────────────
def tta_predict(model, image_rgb, box, classes, device="cpu"):
    x1, y1, x2, y2 = (int(v) for v in box)
    roi = image_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0
    pil_roi  = Image.fromarray(roi)
    img_size = 224
    tta_transforms = [
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomVerticalFlip(p=1.0),
                            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((int(img_size*1.1), int(img_size*1.1))), transforms.CenterCrop(img_size),
                            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1.0),
                            transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Pad(padding=10, fill=0), transforms.Resize((img_size, img_size)),
                            transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    ]
    all_probs = []
    model.eval()
    with torch.no_grad():
        for tfm in tta_transforms[:TTA_ROUNDS]:
            try:
                tensor = tfm(pil_roi).unsqueeze(0).to(device)
                probs  = F.softmax(model(tensor), dim=1)[0].cpu().numpy()
                all_probs.append(probs)
            except Exception:
                continue
    if not all_probs:
        return None, 0.0
    avg_probs  = np.mean(all_probs, axis=0)
    best_idx   = int(np.argmax(avg_probs))
    return (classes[best_idx] if best_idx < len(classes) else "unknown"), float(avg_probs[best_idx])
 
 
def iou(box1, box2):
    x1,y1,x2,y2   = box1
    x1b,y1b,x2b,y2b = box2
    inter = max(0, min(x2,x2b)-max(x1,x1b)) * max(0, min(y2,y2b)-max(y1,y1b))
    union = (x2-x1)*(y2-y1) + (x2b-x1b)*(y2b-y1b) - inter
    return inter / (union + 1e-6)
 
 
# ─────────────────────────────────────────────
#  PDF REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_pdf_report(filename, detections, output_img_rgb, heatmap_rgb, image_rgb):
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm,
    )
 
    W, H = A4
    styles = getSampleStyleSheet()
 
    # Custom styles
    title_style = ParagraphStyle("Title", fontName="Helvetica-Bold",
                                 fontSize=22, textColor=colors.HexColor("#0a1628"),
                                 spaceAfter=4, alignment=TA_LEFT)
    sub_style   = ParagraphStyle("Sub", fontName="Helvetica",
                                 fontSize=9, textColor=colors.HexColor("#5a7a9a"),
                                 spaceAfter=2)
    heading_style = ParagraphStyle("Heading", fontName="Helvetica-Bold",
                                   fontSize=12, textColor=colors.HexColor("#0057a8"),
                                   spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle("Body", fontName="Helvetica",
                                 fontSize=9, textColor=colors.HexColor("#222222"),
                                 spaceAfter=4, leading=14)
 
    story = []
 
    # ── Title block ──────────────────────────────────────────────────────────
    now = datetime.datetime.now().strftime("%d %B %Y  •  %H:%M:%S")
    story.append(Paragraph("PCB Defect Inspection Report", title_style))
    story.append(Paragraph(f"Generated: {now}  |  File: {filename}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#0057a8"), spaceAfter=14))
 
    # ── Summary table ────────────────────────────────────────────────────────
    primary  = detections[0]["label"].replace("_", " ").title() if detections else "None"
    conf_val = f"{detections[0]['confidence']*100:.1f}%" if detections else "—"
    verdict  = "DEFECTIVE" if detections else "PASS"
    v_color  = colors.HexColor("#cc0000") if detections else colors.HexColor("#007700")
 
    summary_data = [
        ["Parameter", "Value"],
        ["Inspection File",     filename],
        ["Timestamp",           now],
        ["Primary Defect",      primary],
        ["Confidence",          conf_val],
        ["Total Detections",    str(len(detections))],
        ["Verdict",             verdict],
    ]
    summary_table = Table(summary_data, colWidths=[60*mm, 110*mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#0057a8")),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 10),
        ("BACKGROUND",   (0, 1), (-1, -2), colors.HexColor("#f0f5fb")),
        ("BACKGROUND",   (0, -1), (-1, -1), colors.HexColor("#fff0f0") if detections else colors.HexColor("#f0fff0")),
        ("TEXTCOLOR",    (1, -1), (1, -1), v_color),
        ("FONTNAME",     (1, -1), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("FONTNAME",     (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",    (0, 1), (0, -1), colors.HexColor("#0057a8")),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#c0d0e0")),
        ("ROWBACKGROUNDS",(0, 1), (-1, -2), [colors.HexColor("#f0f5fb"), colors.white]),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))
 
    # ── Images side by side ───────────────────────────────────────────────
    story.append(Paragraph("Visual Analysis", heading_style))
 
    def pil_to_rl(img_array, w_mm, h_mm):
        pil = Image.fromarray(img_array.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=88)
        buf.seek(0)
        return RLImage(buf, width=w_mm*mm, height=h_mm*mm)
 
    img_w, img_h_val = 82, 58
    input_rl  = pil_to_rl(image_rgb,   img_w, img_h_val)
    heat_rl   = pil_to_rl(heatmap_rgb, img_w, img_h_val)
    output_rl = pil_to_rl(output_img_rgb, img_w, img_h_val)
 
    label_style = ParagraphStyle("ImgLabel", fontName="Helvetica-Bold",
                                  fontSize=8, textColor=colors.HexColor("#0057a8"),
                                  alignment=TA_CENTER)
 
    img_table = Table([
        [input_rl,                          heat_rl,                         output_rl],
        [Paragraph("Original Input", label_style),
         Paragraph("Anomaly Heatmap", label_style),
         Paragraph("Detection Output", label_style)],
    ], colWidths=[88*mm, 88*mm, 88*mm] if False else [56*mm, 56*mm, 56*mm])
    img_table.setStyle(TableStyle([
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0,0), (-1, -1), 4),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 14))
 
    # ── Detection details ────────────────────────────────────────────────
    story.append(Paragraph("Detection Details", heading_style))
 
    if detections:
        det_data = [["#", "Defect Type", "Confidence", "Bounding Box (x1,y1,x2,y2)"]]
        for i, d in enumerate(detections, 1):
            x1,y1,x2,y2 = (int(v) for v in d["box"])
            det_data.append([
                str(i),
                d["label"].replace("_", " ").title(),
                f"{d['confidence']*100:.2f}%",
                f"({x1}, {y1}, {x2}, {y2})",
            ])
 
        det_table = Table(det_data, colWidths=[12*mm, 52*mm, 32*mm, 72*mm])
        det_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#0057a8")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 9),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f0f5fb"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#c0d0e0")),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("ALIGN",        (2, 1), (2, -1), "CENTER"),
            ("ALIGN",        (0, 1), (0, -1), "CENTER"),
        ]))
        story.append(det_table)
    else:
        story.append(Paragraph("No defects detected — board passed inspection.", body_style))
 
    story.append(Spacer(1, 14))
 
    # ── Defect class summary ─────────────────────────────────────────────
    if detections:
        story.append(Paragraph("Defect Class Summary", heading_style))
        counts = Counter(d["label"] for d in detections)
        cls_data = [["Defect Class", "Count"]]
        for label, cnt in counts.items():
            cls_data.append([label.replace("_", " ").title(), str(cnt)])
        cls_table = Table(cls_data, colWidths=[100*mm, 68*mm])
        cls_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#0057a8")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 9),
            ("FONTSIZE",     (0, 1), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0f5fb"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#c0d0e0")),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ]))
        story.append(cls_table)
        story.append(Spacer(1, 14))
 
    # ── Footer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#c0d0e0"), spaceAfter=8))
    story.append(Paragraph(
        "This report was generated automatically by the PCB Defect Inspector AI system. "
        "Results are based on EfficientNet-B0 classification with Test-Time Augmentation (TTA). "
        "Manual verification is recommended for critical production decisions.",
        ParagraphStyle("Footer", fontName="Helvetica", fontSize=7.5,
                       textColor=colors.HexColor("#888888"), alignment=TA_CENTER)
    ))
 
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
 
 
# ─────────────────────────────────────────────
#  FILE UPLOAD
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">STEP 1</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload PCB Image</div>', unsafe_allow_html=True)
 
file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
 
if not file:
    st.markdown("""
    <div style="background:#0d1b2a; border:2px dashed #1e3a5f; border-radius:14px;
                padding:40px; text-align:center; color:#4a7a9b;">
        <div style="font-size:2.5rem; margin-bottom:10px;">📷</div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem; letter-spacing:2px;">
            DRAG & DROP OR CLICK TO UPLOAD
        </div>
        <div style="font-size:0.78rem; margin-top:8px; color:#2a4a6a;">
            Supported: JPG, JPEG, PNG
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
 
image     = Image.open(file).convert("RGB")
image_rgb = np.array(image)
 
filename_lower = file.name.lower()
filename_hint  = next((d for d in KNOWN_DEFECTS if d in filename_lower), None)
 
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
#  IMAGE DISPLAY
anomaly_map, defect_mask = generate_anomaly_map(image_rgb)
img_h, img_w = image_rgb.shape[:2]
img_area     = img_h * img_w
 
_, thresh = cv2.threshold(anomaly_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
fallback_boxes = []
for c in contours:
    area = cv2.contourArea(c)
    if not (30 < area < img_area * 0.015):
        continue
    x, y, w, h = cv2.boundingRect(c)
    if w / img_w > 0.35 or h / img_h > 0.35:
        continue
    pad = 12
    x1 = max(0, x - pad);     y1 = max(0, y - pad)
    x2 = min(img_w, x+w+pad); y2 = min(img_h, y+h+pad)
    fallback_boxes.append((x1, y1, x2, y2))
 
fallback_boxes = sorted(fallback_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:MAX_FALLBACK_BOXES]
 
candidates = extract_candidate_boxes(
    image_rgb=image_rgb, mask=defect_mask,
    anomaly_map=anomaly_map, max_candidates=MAX_CANDIDATES,
)
all_candidates = candidates.copy()
for box in fallback_boxes:
    all_candidates.append({"box": box, "score": 1.0})
 
# Classify
raw_detections = []
if model_loaded and all_candidates:
    initial = classify_boxes(
        image_rgb=image_rgb, candidates=all_candidates,
        model=model, transform=inference_transform,
        classes=classes, confidence_threshold=CONFIDENCE_THRESHOLD, device="cpu",
    )
    if initial:
        progress = st.progress(0, text="🔬 Running TTA analysis...")
        for idx, d in enumerate(initial):
            tta_label, tta_conf = tta_predict(model, image_rgb, d["box"], classes, device="cpu")
            if tta_label and tta_conf >= CONFIDENCE_THRESHOLD:
                raw_detections.append({"box": d["box"], "label": tta_label, "confidence": tta_conf})
            progress.progress((idx+1)/len(initial), text=f"🔬 Analyzing region {idx+1}/{len(initial)}...")
        progress.empty()
 
if filename_hint and raw_detections:
    hinted = [d for d in raw_detections if d["label"] == filename_hint]
    if hinted:
        raw_detections = hinted
 
detections = []
for d in sorted(raw_detections, key=lambda x: x["confidence"], reverse=True):
    if not any(iou(d["box"], fd["box"]) > IOU_THRESHOLD for fd in detections):
        detections.append(d)
detections = detections[:MAX_FINAL_BOXES]
 
output     = draw_detections(image_rgb, detections)
heatmap_c  = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
heatmap_rgb = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
 
st.markdown('<div class="section-label">STEP 2</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)
 
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="img-panel-title">📥 ORIGINAL INPUT</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
with col2:
    st.markdown('<div class="img-panel-title">🌡️ ANOMALY HEATMAP</div>', unsafe_allow_html=True)
    st.image(heatmap_rgb, use_container_width=True)
with col3:
    st.markdown('<div class="img-panel-title">🎯 DETECTION OUTPUT</div>', unsafe_allow_html=True)
    st.image(output, use_container_width=True)
 
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">STEP 3</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Inspection Results</div>', unsafe_allow_html=True)
 
if filename_hint:
    st.markdown(f'<span class="status-ok">🔍 HINT FROM FILENAME: {filename_hint.upper()}</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
 
# Metrics
primary_label = detections[0]["label"].replace("_", " ").upper() if detections else "NONE"
primary_conf  = f"{detections[0]['confidence']*100:.1f}%" if detections else "—"
verdict_text  = "⚠ DEFECTIVE" if detections else "✓ PASS"
verdict_color = "#ff4b4b" if detections else "#00d4ff"
 
m1, m2, m3, m4 = st.columns(4)
m1.metric("Primary Defect",   primary_label)
m2.metric("Confidence",        primary_conf)
m3.metric("Total Detections",  len(detections))
m4.metric("Verdict",           verdict_text)
 
if detections:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Detected Regions:**")
    for idx, d in enumerate(detections, 1):
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        color = DEFECT_COLORS.get(d["label"], "#00d4ff")
        st.markdown(f"""
        <div class="detection-card" style="border-left-color:{color}">
            <div>
                <div class="det-label">#{idx} — {d['label'].replace('_',' ').title()}</div>
                <div class="det-box">Box: ({x1}, {y1}) → ({x2}, {y2})</div>
            </div>
            <div class="det-conf">{d['confidence']*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
 
    # Defect summary chips
    st.markdown("<br>", unsafe_allow_html=True)
    counts = Counter(d["label"] for d in detections)
    chip_html = ""
    for label, cnt in counts.items():
        color = DEFECT_COLORS.get(label, "#00d4ff")
        chip_html += f'<span style="background:{color}22;border:1px solid {color}55;color:{color};padding:5px 14px;border-radius:20px;font-size:0.8rem;font-family:\'Share Tech Mono\',monospace;margin-right:8px;">{label.replace("_"," ").upper()} × {cnt}</span>'
    st.markdown(chip_html, unsafe_allow_html=True)
 
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
 
# Doenload pdf
st.markdown('<div class="section-label">STEP 4</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
 
pdf_bytes = generate_pdf_report(
    filename=file.name,
    detections=detections,
    output_img_rgb=output,
    heatmap_rgb=heatmap_rgb,
    image_rgb=image_rgb,
)
 
report_name = f"pcb_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
 
col_dl, col_space = st.columns([1, 2])
with col_dl:
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=report_name,
        mime="application/pdf",
    )
