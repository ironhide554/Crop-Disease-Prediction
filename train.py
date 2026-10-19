"""
train.py - CLI Training pipeline for Crop Disease Prediction with ResNet-50.
Usage:
    python train.py --data_dir data/ --epochs 30 --batch_size 32 --lr 0.001
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.model import build_crop_disease_model, compile_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 Crop Disease Prediction Model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to dataset directory containing class subfolders, or containing 'train' and 'valid' folders."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate for Adam optimizer (default: 0.001)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image resolution in pixels (default: 224)."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=os.path.join("models", "crop_disease_resnet50.h5"),
        help="Destination path to save the best trained model (default: models/crop_disease_resnet50.h5)."
    )
    return parser.parse_args()


def prepare_data_generators(data_dir: str, image_size: int, batch_size: int):
    """
    Prepares training and validation generators.
    Supports either separate train/valid subdirectories or automatic 80/20 split.
    """
    train_subdir = os.path.join(data_dir, "train")
    valid_subdir = os.path.join(data_dir, "valid")

    target_size = (image_size, image_size)

    if os.path.exists(train_subdir) and os.path.exists(valid_subdir):
        print(f"📁 Found dedicated train and valid subfolders in: {data_dir}")

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            zoom_range=0.3,
            shear_range=0.2,
            horizontal_flip=True,
            rotation_range=20
        )
        valid_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_gen = train_datagen.flow_from_directory(
            train_subdir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        valid_gen = valid_datagen.flow_from_directory(
            valid_subdir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False
        )
    else:
        print(f"📁 Splitting dataset from '{data_dir}' (80% train / 20% validation)...")
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            zoom_range=0.3,
            shear_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_gen = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=42
        )
        valid_gen = datagen.flow_from_directory(
            data_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )

    return train_gen, valid_gen


def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: '{args.data_dir}'", file=sys.stderr)
        print("💡 Download the PlantVillage dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print(f"   and extract it into the '{args.data_dir}/' folder.")
        sys.exit(1)

    # Ensure output model directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    print("🚀 Initializing Data Generators...")
    train_gen, valid_gen = prepare_data_generators(args.data_dir, args.image_size, args.batch_size)

    num_classes = train_gen.num_classes
    print(f"✅ Found {train_gen.samples} training images and {valid_gen.samples} validation images across {num_classes} classes.")

    print(f"🧠 Building ResNet-50 Transfer Learning Model ({num_classes} classes)...")
    model = build_crop_disease_model(num_classes=num_classes, input_shape=(args.image_size, args.image_size, 3))
    model = compile_model(model, learning_rate=args.lr)

    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=args.save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"🔥 Starting Training for {args.epochs} Epochs...")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=valid_gen,
        callbacks=callbacks
    )

    print(f"🎉 Training complete! Best model successfully saved to: {args.save_path}")


if __name__ == "__main__":
    main()
