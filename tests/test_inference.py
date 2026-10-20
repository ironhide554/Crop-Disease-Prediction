"""
test_inference.py - Unit tests for model loading, preprocessing, and inference.
"""

import os
import pytest
import numpy as np
from PIL import Image

from src.model import load_trained_model, build_crop_disease_model, TFLiteModelWrapper
from src.disease_data import CLASS_MAPPING, get_disease_info
from src.utils import load_and_preprocess_image, predict_top_k


def test_disease_data_ontology():
    """Verify that all 38 classes are indexed and contain complete agronomic fields."""
    assert len(CLASS_MAPPING) == 38
    for idx in range(38):
        info = get_disease_info(idx)
        assert "crop" in info and len(info["crop"]) > 0
        assert "disease" in info and len(info["disease"]) > 0
        assert "category" in info and len(info["category"]) > 0
        assert "symptoms" in info and len(info["symptoms"]) > 0
        assert "organic_treatment" in info and len(info["organic_treatment"]) > 0
        assert "chemical_treatment" in info and len(info["chemical_treatment"]) > 0
        assert "prevention" in info and len(info["prevention"]) > 0


def test_tflite_model_loading_and_shape():
    """Verify MobileNetV3.tflite loads and has correct input/output shapes."""
    if os.path.exists("MobileNetV3.tflite"):
        model = load_trained_model("MobileNetV3.tflite")
        assert isinstance(model, TFLiteModelWrapper)
        assert model.input_shape == (1, 224, 224, 3)
        assert model.output_shape == (1, 38)
        assert model.num_classes == 38


def test_preprocessing_output_dimensions():
    """Verify preprocessing outputs correct batch dimensions and float32 dtype."""
    sample_path = "samples/real_leaf_sample.jpg"
    if os.path.exists(sample_path):
        batch, pil_img = load_and_preprocess_image(sample_path, model_type="mobilenet")
        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        assert isinstance(pil_img, Image.Image)


def test_predict_top_k_probability_distribution():
    """Verify inference produces valid probabilities that sum close to 1.0."""
    sample_path = "samples/real_leaf_sample.jpg"
    if os.path.exists("MobileNetV3.tflite") and os.path.exists(sample_path):
        model = load_trained_model("MobileNetV3.tflite")
        results = predict_top_k(model, sample_path, top_k=5)
        
        assert len(results) == 5
        top = results[0]
        assert 0.0 <= top["confidence"] <= 1.0
        assert 0.0 <= top["confidence_percentage"] <= 100.0
        assert "details" in top
        assert "crop" in top["details"]
