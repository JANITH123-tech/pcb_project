import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw
import cv2
import numpy as np
import io

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="PCB Detect", layout="wide")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.title {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: #0a3d62;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}
.result {
    font-size: 24px;
    font-weight: bold;
    color: #1e90ff;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown('<div class="title">🔍 PCB Detect</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced PCB Defect Detection System</div>', unsafe_allow_html=True)

# =========================
# CLASSES
# =========================
classes = [
    "Missing Hole",
    "Mouse Bite",
    "Open Circuit",
    "Short",
    "Spur",
    "Spurious Copper"
]

# =========================
# LOAD MODEL
# =========================
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 6)
model.load_state_dict(torch.load("pcb_defect_model.pth", map_location="cpu"))
model.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload PCB Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Convert to OpenCV
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # =========================
    # SUBTRACTION (Simple)
    # =========================
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    subtracted = cv2.absdiff(gray, blurred)

    # Threshold
    _, thresh = cv2.threshold(subtracted, 30, 255, cv2.THRESH_BINARY)

    # Find contours (defect regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    coordinates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter small noise
        if w > 20 and h > 20:
            coordinates.append((x, y, x+w, y+h))
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)

    # =========================
    # MODEL PREDICTION
    # =========================
    img_tensor = transform(image).unsqueeze(0)

    output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    # =========================
    # DISPLAY IMAGES
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📥 Input Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("📤 Output Image (Detected)")
        st.image(output_image, use_column_width=True)

    with col3:
        st.subheader("⚡ Subtraction Image")
        st.image(subtracted, use_column_width=True, clamp=True)

    # =========================
    # RESULTS
    # =========================
    st.markdown(f'<div class="result">Prediction: {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result">Confidence: {confidence_score:.2f}%</div>', unsafe_allow_html=True)

    # =========================
    # COORDINATES
    # =========================
    st.subheader("📍 Detected Coordinates")

    if coordinates:
        for i, coord in enumerate(coordinates):
            st.write(f"Defect {i+1}: {coord}")
    else:
        st.write("No significant defect regions detected")

    # =========================
    # DOWNLOAD OUTPUT
    # =========================
    buf = io.BytesIO()
    output_image.save(buf, format="PNG")

    st.download_button(
        label="📥 Download Output Image",
        data=buf.getvalue(),
        file_name="pcb_result.png",
        mime="image/png"
    )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 PCB Detect | Advanced AI Defect Detection")