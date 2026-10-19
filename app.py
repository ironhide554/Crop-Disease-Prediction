"""
app.py - Streamlit Web Application for Crop Disease & Pest Prediction.
Features:
1. Dual-Engine Inference: MobileNetV3 (TFLite Edge) & ResNet-50 (Keras)
2. Live Mobile/Webcam Camera Input (st.camera_input)
3. Drag-and-drop Image Upload & Benchmark Sample Gallery
4. Agronomic Prescription Engine (Symptoms, Organic, Chemical & Cultural)
5. Downloadable Agronomist Field Report (Branded PDF Export)
"""

import os
import time
import glob
import datetime
from PIL import Image
import numpy as np
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AgriVision - Crop Disease & Pest Prediction",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .disease-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-healthy {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .badge-diseased {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .remedy-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 12px;
        border-radius: 4px;
        margin-top: 8px;
    }
    .chemical-box {
        background-color: #fefce8;
        border-left: 4px solid #eab308;
        padding: 12px;
        border-radius: 4px;
        margin-top: 8px;
    }
    .download-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

from src.model import load_trained_model, build_crop_disease_model
from src.disease_data import CLASS_MAPPING, DISEASE_DETAILS, get_disease_info
from src.utils import predict_top_k
from src.report_generator import generate_pdf_report


@st.cache_resource
def get_cached_model(model_path: str):
    """Caches loaded model instance in memory."""
    if os.path.exists(model_path):
        try:
            model = load_trained_model(model_path)
            if model_path.lower().endswith(".tflite"):
                desc = "⚡ MobileNetV3 (TFLite Edge Optimized) - Loaded & Ready"
            else:
                desc = f"🧠 ResNet-50 Keras Model ({os.path.basename(model_path)}) - Loaded & Ready"
            return model, desc, True
        except Exception as e:
            return build_crop_disease_model(num_classes=38), f"Fallback: Base ResNet-50 (Error loading {model_path}: {e})", False
    else:
        return build_crop_disease_model(num_classes=38), f"Model file '{model_path}' not found. Initialized base ResNet-50.", False


def main():
    st.markdown('<div class="main-title">🌱 AgriVision AI — Crop Disease & Pest Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Intelligent agricultural diagnostic system powered by Deep Learning. Detects 38 crop disease conditions across 14 plant species with instant agronomic prescriptions.</div>', unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        available_models = {}
        if os.path.exists("MobileNetV3.tflite"):
            available_models["⚡ MobileNetV3 TFLite (Fast Edge Model, ~12 MB) [Default]"] = "MobileNetV3.tflite"
        
        resnet_path = os.path.join("models", "crop_disease_resnet50.h5")
        if os.path.exists(resnet_path):
            available_models["🧠 ResNet-50 Full Model (Keras, ~98 MB)"] = resnet_path
        
        if not available_models:
            available_models["MobileNetV3.tflite"] = "MobileNetV3.tflite"

        selected_model_name = st.selectbox(
            "Active Inference Engine:",
            list(available_models.keys()),
            index=0
        )
        selected_model_path = available_models[selected_model_name]

        top_k = st.slider("Top Candidates to Display", min_value=1, max_value=8, value=5)

        st.markdown("---")
        st.markdown("### 📊 Active Model Specifications")
        if "MobileNet" in selected_model_name:
            st.success("**Architecture:** MobileNetV3-Large\n\n**Format:** TensorFlow Lite (.tflite)\n\n**Footprint:** ~12 MB\n\n**Accuracy:** ~95%\n\n**Edge Ready:** Optimized for Raspberry Pi / Webcams")
        else:
            st.info("**Architecture:** ResNet-50\n\n**Format:** Keras HDF5 (.h5)\n\n**Footprint:** ~98 MB\n\n**Depth:** 50 Residual Layers")

        st.markdown("---")
        st.markdown("### 📚 Dataset Details")
        st.markdown("[PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)")

    # Load model
    model, status_msg, is_trained = get_cached_model(selected_model_path)
    if is_trained:
        st.success(f"✅ {status_msg}")
    else:
        st.warning(f"ℹ️ {status_msg}")

    # Tabs for image input
    tab_camera, tab_upload, tab_sample, tab_database = st.tabs([
        "📸 Live Camera Capture",
        "📤 Upload Leaf Image",
        "🖼️ Try Benchmark Samples",
        "📖 38-Class Reference Database"
    ])

    selected_image = None
    image_source_label = ""

    with tab_camera:
        st.markdown("##### 📷 Take a live photo of a crop leaf:")
        st.caption("Point your phone or webcam at the affected leaf with good lighting.")
        camera_photo = st.camera_input("Capture Leaf Photo")
        if camera_photo is not None:
            selected_image = Image.open(camera_photo)
            image_source_label = "Live Camera Photo"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a close-up photograph of the crop leaf:",
            type=["jpg", "jpeg", "png", "webp"],
            help="Best results are achieved with well-lit, in-focus photos of the upper or lower leaf surface."
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_source_label = f"Uploaded: {uploaded_file.name}"

    with tab_sample:
        sample_dir = os.path.join(os.path.dirname(__file__), "samples")
        sample_files = glob.glob(os.path.join(sample_dir, "*.jpg")) + glob.glob(os.path.join(sample_dir, "*.png"))

        if sample_files:
            sample_names = [os.path.basename(f) for f in sample_files]
            chosen_sample = st.selectbox("Select a benchmark sample image:", sample_names)
            if chosen_sample:
                sample_path = os.path.join(sample_dir, chosen_sample)
                selected_image = Image.open(sample_path)
                image_source_label = f"Sample: {chosen_sample}"
        else:
            st.info("No sample images in `samples/`. Generate sample images using `python generate_samples.py`.")

    with tab_database:
        st.subheader("📋 Supported Crop Disease & Health Classes (38 Total)")
        crop_filter = st.selectbox(
            "Filter by Crop Species:",
            ["All Crops", "Apple", "Blueberry", "Cherry", "Corn (maize)", "Grape", "Orange", "Peach", "Pepper Bell", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"]
        )

        displayed_classes = []
        for idx, details in DISEASE_DETAILS.items():
            if crop_filter == "All Crops" or details["crop"].lower() in crop_filter.lower():
                displayed_classes.append({
                    "Index": idx,
                    "Crop": details["crop"],
                    "Condition": details["disease"],
                    "Category": details["category"],
                    "Key Symptoms": details["symptoms"]
                })

        st.dataframe(displayed_classes, use_container_width=True)

    # Prediction Section
    if selected_image is not None:
        st.markdown("---")
        st.subheader(f"🔍 Diagnostic Analysis ({image_source_label})")

        col1, col2 = st.columns([1, 1.4], gap="large")

        with col1:
            st.image(selected_image, caption=f"Specimen: {image_source_label}", use_column_width=True)

        with col2:
            start_t = time.perf_counter()
            with st.spinner("Analyzing leaf morphology and spectral features..."):
                predictions = predict_top_k(model, selected_image, top_k=top_k)
            latency_ms = (time.perf_counter() - start_t) * 1000

            primary = predictions[0]
            is_healthy = "healthy" in primary["class_name"].lower()
            badge_class = "badge-healthy" if is_healthy else "badge-diseased"
            badge_label = "HEALTHY CROP" if is_healthy else "DISEASE DETECTED"

            st.markdown(f"""
            <div class="metric-card">
                <span class="disease-badge {badge_class}">{badge_label}</span>
                <span style="float: right; color: #64748b; font-size: 0.85rem;">⚡ Latency: <strong>{latency_ms:.1f} ms</strong></span>
                <h3 style="margin-top: 10px; margin-bottom: 4px;">{primary['class_name']}</h3>
                <p style="color: #64748b; margin-bottom: 8px;">Confidence: <strong>{primary['confidence_percentage']:.2f}%</strong> | Pathogen: <em>{primary['details']['category']}</em></p>
            </div>
            """, unsafe_allow_html=True)

            # Top candidates progress bars
            st.markdown("##### 📊 Top Probability Distributions:")
            for item in predictions:
                label = f"{item['class_name']} ({item['confidence_percentage']:.1f}%)"
                st.write(label)
                st.progress(float(item['confidence']))

        # Detailed Agronomic Prescription
        st.markdown("### 📋 Agronomic Diagnosis & Treatment Prescription")
        info = primary["details"]

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("#### 🔬 Symptoms & Pathology")
            st.write(f"**Identified Cause / Pathogen:** {info['category']}")
            st.write(f"**Field Symptoms:** {info['symptoms']}")

            st.markdown("#### 🛡️ Cultural & Preventative Practices")
            st.write(info['prevention'])

        with exp_col2:
            st.markdown("#### 🌿 Organic & Biological Remedies")
            st.markdown(f'<div class="remedy-box">{info["organic_treatment"]}</div>', unsafe_allow_html=True)

            st.markdown("#### 🧪 Chemical Control Options")
            st.markdown(f'<div class="chemical-box">{info["chemical_treatment"]}</div>', unsafe_allow_html=True)

        # PDF Field Report Download Section
        st.markdown("---")
        st.markdown("### 📄 Export Agronomist Field Report")
        
        pdf_col1, pdf_col2 = st.columns([2, 1])
        with pdf_col1:
            st.markdown("""
            Generate a standardized, printable **Agronomic Field Diagnostic Report (PDF)** 
            complete with the leaf photograph, primary diagnosis, confidence metrics, differential table, 
            and complete treatment prescriptions for farm records or extension consults.
            """)
        
        with pdf_col2:
            with st.spinner("Compiling PDF report..."):
                engine_name = "MobileNetV3 (TFLite Edge)" if "MobileNet" in selected_model_name else "ResNet-50 (Keras)"
                pdf_bytes = generate_pdf_report(
                    predictions=predictions,
                    leaf_image=selected_image,
                    model_name=engine_name,
                    latency_ms=latency_ms,
                    specimen_name=image_source_label
                )

            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"AgriVision_Report_{info['crop']}_{timestamp_str}.pdf"

            st.download_button(
                label="📥 Download Field Report (PDF)",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                help="Click to download the official diagnostic summary in PDF format."
            )


if __name__ == "__main__":
    main()
