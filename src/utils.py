"""
utils.py - Preprocessing, prediction helpers, and report formatting for Crop Disease Prediction.
Supports both MobileNetV3 (TFLite) and ResNet-50 (Keras) models.
"""

import os
from typing import Union, List, Dict, Any, Tuple
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from src.disease_data import CLASS_MAPPING, get_disease_info


def load_and_preprocess_image(
    image_input: Union[str, Image.Image],
    target_size: Tuple[int, int] = (224, 224),
    model_type: str = "auto"
) -> Tuple[np.ndarray, Image.Image]:
    """
    Loads an image from path or PIL Image, validates RGB color mode,
    resizes to target dimensions, and applies model-specific preprocessing.

    Args:
        image_input: File path (str) or PIL.Image instance.
        target_size: Desired (height, width) tuple (default 224x224).
        model_type: 'mobilenet', 'tflite', 'resnet', or 'auto'.
                    - MobileNetV3 uses raw [0, 255] float32 (internal rescaling layer).
                    - ResNet-50 uses zero-centered BGR normalization.

    Returns:
        Tuple of (preprocessed_batch_array (1, 224, 224, 3), original_pil_image).
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        # File-like object (e.g., Streamlit UploadedFile)
        pil_img = Image.open(image_input).convert("RGB")

    # Resize with high-quality resampling
    resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    # Normalize based on model architecture
    if model_type in ["mobilenet", "tflite"]:
        # MobileNetV3 expects raw [0, 255] float32 pixels
        preprocessed_batch = img_batch
    elif model_type == "resnet":
        preprocessed_batch = resnet_preprocess(img_batch.copy())
    else:
        # Default auto behavior
        preprocessed_batch = img_batch

    return preprocessed_batch, pil_img


def predict_top_k(
    model: Any,
    image_input: Union[str, Image.Image],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Runs model inference on an image and returns the top-k highest confidence classes
    along with agronomic management details.

    Args:
        model: Loaded tf.keras.Model or TFLiteModelWrapper.
        image_input: Path or PIL image.
        top_k: Number of ranked predictions to return.

    Returns:
        List of dictionaries containing class_index, class_name, confidence, and disease details.
    """
    # Detect model type
    model_type = "mobilenet" if hasattr(model, "interpreter") or getattr(model, "model_type", None) == "tflite_mobilenet" else "resnet"
    
    img_batch, _ = load_and_preprocess_image(image_input, model_type=model_type)
    raw_preds = model.predict(img_batch, verbose=0)[0]

    # Sort descending by probability
    top_indices = np.argsort(raw_preds)[::-1][:top_k]

    results = []
    for idx in top_indices:
        confidence = float(raw_preds[idx])
        info = get_disease_info(int(idx))
        results.append({
            "class_index": int(idx),
            "class_name": CLASS_MAPPING.get(int(idx), f"Class {idx}"),
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 2),
            "details": info
        })

    return results


def format_cli_report(predictions: List[Dict[str, Any]], model_label: str = "Trained Model") -> str:
    """
    Formats the prediction results into a readable text report for terminal output.
    """
    if not predictions:
        return "No predictions available."

    top = predictions[0]
    top_info = top["details"]

    lines = [
        "=" * 65,
        f"🌱 CROP DISEASE PREDICTION REPORT  [{model_label}]",
        "=" * 65,
        f"🎯 Primary Diagnosis : {top['class_name']}",
        f"📊 Confidence Score  : {top['confidence_percentage']:.2f}%",
        f"🌾 Crop               : {top_info['crop']}",
        f"🔬 Pathogen Category : {top_info['category']}",
        "-" * 65,
        "🔍 Key Symptoms:",
        f"   {top_info['symptoms']}",
        "",
        "🌿 Organic / Cultural Treatment:",
        f"   {top_info['organic_treatment']}",
        "",
        "🧪 Chemical Control Options:",
        f"   {top_info['chemical_treatment']}",
        "",
        "🛡️ Prevention & Cultural Practices:",
        f"   {top_info['prevention']}",
        "-" * 65,
        "📊 Top Alternative Diagnoses:"
    ]

    for rank, item in enumerate(predictions[1:], start=2):
        lines.append(f"   [{rank}] {item['class_name']} ({item['confidence_percentage']:.2f}%)")

    lines.append("=" * 65)
    return "\n".join(lines)
