"""
model.py - Neural network model architectures and loaders for Crop Disease and Pest Classification.
Supports ResNet-50 (Keras .h5 / .keras) and MobileNetV3 (TensorFlow Lite .tflite).
"""

import os
from typing import Union, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.resnet50 import ResNet50


class TFLiteModelWrapper:
    """
    High-performance wrapper around tf.lite.Interpreter mimicking the Keras model predict() API.
    Enables low-latency edge inference on CPU and Raspberry Pi.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model not found at: {model_path}")
        
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = tuple(self.input_details[0]['shape'])
        self.output_shape = tuple(self.output_details[0]['shape'])
        self.num_classes = self.output_shape[-1]
        self.model_type = "tflite_mobilenet"

    def predict(self, input_tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        """
        Runs TFLite model inference on input batch tensor.
        
        Args:
            input_tensor: Numpy float32 array of shape (batch, 224, 224, 3)
        Returns:
            Numpy array of shape (batch, num_classes) containing predicted class probabilities.
        """
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)

        # Handle single-sample vs batch
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        batch_size = input_tensor.shape[0]
        # In TFLite, if interpreter batch size matches
        if batch_size == 1:
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            outputs = []
            for i in range(batch_size):
                sample = np.expand_dims(input_tensor[i], axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], sample)
                self.interpreter.invoke()
                outputs.append(self.interpreter.get_tensor(self.output_details[0]['index'])[0])
            return np.array(outputs)


def build_crop_disease_model(
    num_classes: int = 38,
    input_shape: tuple = (224, 224, 3),
    freeze_base: bool = True,
    fine_tune_layers: int = 0
) -> Model:
    """
    Constructs the ResNet-50 Crop Disease Classification Model.
    Architecture matches the transfer learning pipeline with GlobalAveragePooling2D,
    Dense(128, relu), Dense(64, relu), and Dense(num_classes, softmax).
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False

        if fine_tune_layers > 0:
            for layer in base_model.layers[-fine_tune_layers:]:
                layer.trainable = True

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_64")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="CropDisease_ResNet50")
    return model


def build_pest_model(
    num_classes: int,
    input_shape: tuple = (224, 224, 3)
) -> Model:
    """
    Constructs the Pest Classification Model using ResNet-50 architecture
    as defined in Pest_classification.ipynb.
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    for layer in base_model.layers:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="pest_input_image")
    x = base_model(inputs, training=False)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(1024, activation="relu", name="dense_1024")(x)
    x = layers.Dropout(0.5, name="dropout_1024")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(0.3, name="dropout_512")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pest_predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="PestClassification_ResNet50")
    return model


def compile_model(
    model: Model,
    learning_rate: float = 0.001
) -> Model:
    """Compiles the model with Adam optimizer and categorical crossentropy loss."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def load_trained_model(model_path: str) -> Union[Model, TFLiteModelWrapper]:
    """
    Unified model loader supporting:
    - TensorFlow Lite (.tflite): MobileNetV3 edge model
    - Keras SavedModel / H5 (.h5, .keras): ResNet50 model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if model_path.lower().endswith(".tflite"):
        return TFLiteModelWrapper(model_path)
    else:
        return tf.keras.models.load_model(model_path)
