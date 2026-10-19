"""
predict.py - Command-line interface for Crop Disease Prediction.
Supports both MobileNetV3 (.tflite) and ResNet-50 (.h5 / .keras).

Usage:
    python predict.py --image_path path/to/leaf.jpg
    python predict.py --image_path path/to/leaf.jpg --model_path MobileNetV3.tflite
    python predict.py --image_path path/to/leaf.jpg --model_path models/crop_disease_resnet50.h5
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import argparse
import warnings

# Suppress verbose TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from src.model import load_trained_model, build_crop_disease_model, TFLiteModelWrapper
from src.utils import predict_top_k, format_cli_report


def find_default_model() -> str:
    """Finds the best available trained model in the workspace."""
    candidates = [
        "MobileNetV3.tflite",
        os.path.join("models", "crop_disease_resnet50.h5"),
        os.path.join("models", "renetmodel.h5"),
        "renetmodel.h5"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "MobileNetV3.tflite"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Crop Disease from leaf images using MobileNetV3 or ResNet-50."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to the crop leaf image file (.jpg, .jpeg, .png)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the trained model file (.tflite, .h5, or .keras). Defaults to MobileNetV3.tflite if present."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top class predictions to display (default: 5)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output prediction results in JSON format."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image_path):
        print(f"❌ Error: Image file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)

    model_path = args.model_path if args.model_path else find_default_model()

    if os.path.exists(model_path):
        if model_path.lower().endswith(".tflite"):
            model_label = f"MobileNetV3 TFLite (Fast Edge Model) [{os.path.basename(model_path)}]"
        else:
            model_label = f"ResNet-50 Keras Model [{os.path.basename(model_path)}]"
        
        print(f"⚡ Loading model: {model_label}...")
        model = load_trained_model(model_path)
    else:
        model_label = "Base ResNet-50 (ImageNet Initialized)"
        print(f"⚠️ Warning: Model file '{model_path}' not found.")
        print("💡 Initializing base ResNet-50 architecture with ImageNet weights for test inference...")
        model = build_crop_disease_model(num_classes=38)

    # Perform prediction
    predictions = predict_top_k(model, args.image_path, top_k=args.top_k)

    if args.json:
        print(json.dumps(predictions, indent=2))
    else:
        report = format_cli_report(predictions, model_label=model_label)
        print(report)


if __name__ == "__main__":
    main()
