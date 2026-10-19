"""
evaluate.py - CLI evaluation script for evaluating trained ResNet-50 models on test datasets.
Usage:
    python evaluate.py --model_path models/crop_disease_resnet50.h5 --data_dir data/test
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

from src.model import load_trained_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet-50 Crop Disease Model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("models", "crop_disease_resnet50.h5"),
        help="Path to trained model file (.h5 or .keras)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/test",
        help="Path to test or validation dataset directory."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size in pixels (default: 224)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Evaluation data directory not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"📦 Loading model from: {args.model_path}...")
    model = load_trained_model(args.model_path)

    print(f"📂 Preparing evaluation generator from: {args.data_dir}...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = test_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False
    )

    print(f"📊 Evaluating model on {test_gen.samples} images across {test_gen.num_classes} classes...")
    scores = model.evaluate(test_gen, verbose=1)
    print("\n" + "=" * 50)
    print(f"📉 Test Loss     : {scores[0]:.4f}")
    print(f"🎯 Test Accuracy : {scores[1] * 100:.2f}%")
    print("=" * 50)

    # Detailed metrics
    print("\n📈 Generating detailed classification report...")
    y_pred_probs = model.predict(test_gen, steps=len(test_gen), verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    target_names = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n" + report)


if __name__ == "__main__":
    main()
