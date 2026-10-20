"""
explainability.py - Explainable AI (XAI) using Grad-CAM (Gradient-weighted Class Activation Mapping).
Generates visual attention heatmaps highlighting the specific regions, lesions,
and morphological features on the crop leaf that influenced the model's diagnosis.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm

from src.utils import load_and_preprocess_image


_GRADCAM_CACHE = {
    "keras_model": None,
    "resnet_conv": None,
    "classifier_model": None
}


def _get_gradcam_submodels(model_or_path: Optional[Union[str, tf.keras.Model]] = None):
    """
    Constructs and caches the intermediate feature extractor and classifier submodels
    for fast Grad-CAM execution on ResNet-50.
    """
    global _GRADCAM_CACHE

    if _GRADCAM_CACHE["resnet_conv"] is not None and _GRADCAM_CACHE["classifier_model"] is not None:
        return _GRADCAM_CACHE["resnet_conv"], _GRADCAM_CACHE["classifier_model"]

    # Load model
    if isinstance(model_or_path, tf.keras.Model):
        model = model_or_path
    else:
        model_path = model_or_path if model_or_path and os.path.exists(model_or_path) else os.path.join("models", "crop_disease_resnet50.h5")
        if not os.path.exists(model_path):
            from src.model import build_crop_disease_model
            model = build_crop_disease_model(num_classes=38)
        else:
            model = tf.keras.models.load_model(model_path)

    _GRADCAM_CACHE["keras_model"] = model

    # Inspect model architecture: find the backbone and classification layers
    try:
        # Check if model has nested resnet50
        resnet_layer = None
        for layer in model.layers:
            if "resnet" in layer.name.lower():
                resnet_layer = layer
                break

        if resnet_layer is not None and hasattr(resnet_layer, "get_layer"):
            # Nested functional architecture
            conv_layer = resnet_layer.get_layer("conv5_block3_out")
            resnet_conv = tf.keras.Model(inputs=resnet_layer.input, outputs=conv_layer.output)

            # Build classifier from the remaining layers
            c_in = tf.keras.Input(shape=conv_layer.output.shape[1:])
            x = c_in
            start_collecting = False
            for layer in model.layers:
                if layer == resnet_layer:
                    start_collecting = True
                    continue
                if start_collecting:
                    x = layer(x)
            classifier_model = tf.keras.Model(inputs=c_in, outputs=x)
        else:
            # Flat architecture
            conv_layer = model.get_layer("conv5_block3_out")
            resnet_conv = tf.keras.Model(inputs=model.input, outputs=conv_layer.output)
            
            c_in = tf.keras.Input(shape=conv_layer.output.shape[1:])
            x = c_in
            found = False
            for layer in model.layers:
                if layer == conv_layer:
                    found = True
                    continue
                if found:
                    x = layer(x)
            classifier_model = tf.keras.Model(inputs=c_in, outputs=x)

        _GRADCAM_CACHE["resnet_conv"] = resnet_conv
        _GRADCAM_CACHE["classifier_model"] = classifier_model
        return resnet_conv, classifier_model

    except Exception as e:
        print(f"Warning: Failed to construct submodels from custom model: {e}. Building standalone ResNet50 Grad-CAM extractor...")
        base = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        conv_out = base.get_layer("conv5_block3_out").output
        resnet_conv = tf.keras.Model(inputs=base.input, outputs=conv_out)
        
        c_in = tf.keras.Input(shape=conv_out.shape[1:])
        x = tf.keras.layers.GlobalAveragePooling2D()(c_in)
        x = tf.keras.layers.Dense(38, activation="softmax")(x)
        classifier_model = tf.keras.Model(inputs=c_in, outputs=x)

        _GRADCAM_CACHE["resnet_conv"] = resnet_conv
        _GRADCAM_CACHE["classifier_model"] = classifier_model
        return resnet_conv, classifier_model


def compute_gradcam_heatmap(
    image_input: Union[str, Image.Image],
    model_or_path: Optional[Union[str, tf.keras.Model]] = None,
    class_index: Optional[int] = None
) -> np.ndarray:
    """
    Computes normalized Grad-CAM activation heatmap for the given image.

    Args:
        image_input: File path (str) or PIL Image.
        model_or_path: Keras model instance or path to .h5 weights file.
        class_index: Target class index. If None, uses top predicted class.

    Returns:
        2D numpy array of shape (7, 7) with normalized values in [0.0, 1.0].
    """
    resnet_conv, classifier_model = _get_gradcam_submodels(model_or_path)

    # Preprocess image for ResNet-50
    x, _ = load_and_preprocess_image(image_input, target_size=(224, 224), model_type="resnet")

    with tf.GradientTape() as tape:
        conv_outputs = resnet_conv(x)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)

        if class_index is None:
            target_idx = tf.argmax(predictions[0])
        else:
            target_idx = class_index

        loss = predictions[:, target_idx]

    # Compute gradients of top class score with respect to feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature channels
    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to focus only on features that have positive influence
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_gradcam(
    heatmap_2d: np.ndarray,
    original_image: Union[str, Image.Image],
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Dict[str, Any]:
    """
    Applies colormap to the Grad-CAM heatmap and superimposes it onto the leaf image.

    Args:
        heatmap_2d: 2D numpy array (e.g. 7x7) with values in [0, 1].
        original_image: Path or PIL Image.
        alpha: Blending weight for the heatmap overlay (0.0 to 1.0).
        colormap_name: Name of matplotlib colormap ('jet', 'viridis', 'magma').

    Returns:
        Dictionary containing:
        - 'heatmap_colored': PIL Image of standalone heatmap
        - 'superimposed_image': PIL Image of blended overlay
        - 'peak_activation': Maximum activation intensity
        - 'focal_center': (x, y) coordinates of the highest attention zone
    """
    if isinstance(original_image, str):
        pil_img = Image.open(original_image).convert("RGB")
    elif isinstance(original_image, Image.Image):
        pil_img = original_image.convert("RGB")
    else:
        pil_img = Image.open(original_image).convert("RGB")

    orig_width, orig_height = pil_img.size

    # Resample 2D heatmap up to original image dimensions with smooth bilinear interpolation
    heatmap_img = Image.fromarray((heatmap_2d * 255.0).astype(np.uint8))
    heatmap_resampled = heatmap_img.resize((orig_width, orig_height), Image.Resampling.BILINEAR)
    heatmap_resampled_arr = np.array(heatmap_resampled, dtype=np.float32) / 255.0

    # Obtain colormap
    try:
        cmap = matplotlib.colormaps.get_cmap(colormap_name)
    except Exception:
        cmap = cm.get_cmap(colormap_name)

    # Colored heatmap (H, W, 3) in [0, 255] uint8
    colored_heatmap = (cmap(heatmap_resampled_arr)[:, :, :3] * 255.0).astype(np.uint8)
    colored_heatmap_pil = Image.fromarray(colored_heatmap)

    # Blend with original leaf
    orig_arr = np.array(pil_img, dtype=np.float32)
    blended_arr = (alpha * colored_heatmap + (1.0 - alpha) * orig_arr).astype(np.uint8)
    superimposed_pil = Image.fromarray(blended_arr)

    # Find peak attention coordinate
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap_resampled_arr), heatmap_resampled_arr.shape)

    return {
        "heatmap_colored": colored_heatmap_pil,
        "superimposed_image": superimposed_pil,
        "peak_activation": float(np.max(heatmap_2d)),
        "focal_center": (int(peak_x), int(peak_y)),
        "overlay_dimensions": (orig_width, orig_height)
    }


def explain_prediction(
    image_input: Union[str, Image.Image],
    model_or_path: Optional[Union[str, tf.keras.Model]] = None,
    class_index: Optional[int] = None,
    alpha: float = 0.45,
    colormap: str = "jet",
    model_path: Optional[Union[str, tf.keras.Model]] = None,
    target_class_idx: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience end-to-end function computing Grad-CAM and generating the superimposed image.

    Args:
        image_input: File path (str) or PIL Image.
        model_or_path: Keras model instance or path to weights.
        class_index: Target class index (or None for top predicted).
        alpha: Overlay blend weight.
        colormap: Matplotlib colormap ('jet', 'viridis', 'magma').
        model_path: Alias for model_or_path.
        target_class_idx: Alias for class_index.

    Returns:
        Dictionary containing heatmap, overlay, focal point, and metadata.
    """
    effective_model = model_path if model_path is not None else model_or_path
    effective_class_idx = target_class_idx if target_class_idx is not None else class_index

    if effective_class_idx is None:
        resnet_conv, classifier_model = _get_gradcam_submodels(effective_model)
        x, _ = load_and_preprocess_image(image_input, target_size=(224, 224), model_type="resnet")
        conv_out = resnet_conv(x)
        preds = classifier_model(conv_out)
        effective_class_idx = int(tf.argmax(preds[0]).numpy())

    heatmap_2d = compute_gradcam_heatmap(image_input, model_or_path=effective_model, class_index=effective_class_idx)
    result = overlay_gradcam(heatmap_2d, image_input, alpha=alpha, colormap_name=colormap)
    result["heatmap_raw"] = heatmap_2d
    result["target_class_idx"] = effective_class_idx
    result["focal_point"] = {"x": result["focal_center"][0], "y": result["focal_center"][1]}
    result["peak_intensity"] = result["peak_activation"]
    return result
