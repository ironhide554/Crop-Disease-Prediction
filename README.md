# 🌱 Crop Disease & Pest Prediction (AgriVision AI)

[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.14%2B-orange.svg)](https://www.tensorflow.org/)
[![ResNet-50](https://img.shields.io/badge/Architecture-ResNet--50-success.svg)](https://arxiv.org/abs/1512.03385)
[![Streamlit UI](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent deep learning computer vision system for automated crop disease diagnosis and agricultural pest classification using **MobileNetV3 (TensorFlow Lite)** and **ResNet-50**. 

The system analyzes crop leaf images, identifies **38 distinct crop disease & healthy conditions across 14 plant species**, and delivers actionable agronomic treatment prescriptions (symptoms, organic treatments, chemical controls, and preventative cultural practices).

- ⚡ **MobileNetV3 (TFLite)**: Ultra-fast, edge-optimized (~12 MB) model trained on PlantVillage achieving **~95% accuracy** for real-time edge, webcam, and Raspberry Pi diagnostics.
- 🧠 **ResNet-50 (Keras)**: Full 50-layer deep residual network (`models/crop_disease_resnet50.h5`, ~98 MB) for deep feature extraction.

---

## 📌 Table of Contents
- [Overview & Features](#-overview--features)
- [Repository Structure](#-repository-structure)
- [Supported Crops & Diseases (38 Classes)](#-supported-crops--diseases-38-classes)
- [Requirements & Installation](#-requirements--installation)
- [Quick Start](#-quick-start)
  - [1. Interactive Web Application](#1-launch-interactive-web-app)
  - [2. Command-Line Inference](#2-command-line-inference)
  - [3. Model Training](#3-train-the-model)
  - [4. Model Evaluation](#4-evaluate-the-model)
- [Dataset](#-dataset)
- [Agronomic Prescriptions](#-agronomic-prescriptions)
- [License](#-license)

---

## 🚀 Overview & Features

- **Deep Residual Learning (ResNet-50)**: Leverages 50 deep convolutional layers with identity shortcut connections to solve vanishing gradients and maximize classification accuracy.
- **38 Plant Health Classes**: Covers major cash & food crops (Apple, Blueberry, Cherry, Corn/Maize, Grape, Orange, Peach, Pepper Bell, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato).
- **Secondary Pest Classification Module**: Includes architectures and workflows for identifying harmful agricultural insect pests.
- **Actionable Treatment Prescriptions**: Translates raw machine learning predictions into direct advisory reports containing:
  - Pathogen classification (Fungal, Bacterial, Viral, Oomycete, or Pest).
  - Diagnostic field symptoms.
  - Organic & biological controls (Neem oil, Bacillus biocontrols, copper fungicides).
  - Chemical controls & active ingredients.
  - Long-term prevention strategies (crop rotation, irrigation management).
- **Interactive Streamlit Web Dashboard**: Drag-and-drop leaf photo diagnostics, top-5 probability breakdown charts, and built-in benchmark samples.
- **Production CLI Pipeline**: Modular Python scripts for training (`train.py`), evaluation (`evaluate.py`), and inference (`predict.py`).

---

## 📂 Repository Structure

```text
Crop-Disease-Prediction/
├── app.py                      # Interactive Streamlit Web Application
├── predict.py                  # Standalone CLI prediction script
├── train.py                    # Standalone CLI model training pipeline
├── evaluate.py                 # Standalone CLI evaluation & metrics script
├── generate_samples.py         # Benchmark test image generator
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
│
├── src/                        # Core Python package
│   ├── __init__.py             # Package init & runtime compatibility polyfills
│   ├── model.py                # ResNet-50 architectures for crop disease & pests
│   ├── disease_data.py         # 38-class metadata, symptoms, and remedies
│   └── utils.py                # Preprocessing, top-k scoring & report formatting
│
├── samples/                    # Sample test images for instant verification
│   ├── sample_apple_scab.jpg
│   ├── sample_tomato_early_blight.jpg
│   ├── sample_healthy_corn.jpg
│   └── sample_potato_late_blight.jpg
│
├── models/                     # Saved model weights directory (.h5 / .keras)
│   └── (crop_disease_resnet50.h5)
│
└── notebooks/                  # Original Google Colab research notebooks
    ├── ResNet50_completed.ipynb
    └── Pest_classification.ipynb
```

---

## 🌾 Supported Crops & Diseases (38 Classes)

The model detects diseases and healthy conditions across 14 plant species:

| Crop Species | Disease & Healthy Conditions Covered |
| :--- | :--- |
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Spot), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper Bell** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mite, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy |

---

## 🛠️ Requirements & Installation

> [!TIP]
> **Recommended Python Version**: Python 3.10 or 3.11 is strongly recommended for TensorFlow compatibility on Windows and Linux.

### 1. Clone the Repository
```bash
git clone https://github.com/ironhide554/Crop-Disease-Prediction.git
cd Crop-Disease-Prediction
```

### 2. Set Up a Virtual Environment
```bash
# Windows
py -3.11 -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### 1. Launch Interactive Web App
Launch the Streamlit web dashboard in your browser:
```bash
streamlit run app.py
```
- Upload your own leaf photos or click **Try Sample Images** to test out-of-the-box.
- View real-time disease diagnosis, confidence meters, and treatment prescriptions.

### 2. Command-Line Inference
Run diagnosis directly on an image:
```bash
# Basic prediction with CLI summary report
python predict.py --image_path samples/sample_tomato_early_blight.jpg

# Output formatted JSON (for integration with APIs or webhooks)
python predict.py --image_path samples/sample_apple_scab.jpg --json

# Specify custom trained weights and top-k predictions
python predict.py --image_path path/to/leaf.jpg --model_path models/crop_disease_resnet50.h5 --top_k 3
```

### 3. Train the Model
Train the ResNet-50 model on your dataset:
```bash
python train.py --data_dir data/ --epochs 30 --batch_size 32 --lr 0.001 --save_path models/crop_disease_resnet50.h5
```
*Supports either dedicated `data/train/` and `data/valid/` folders or automatic 80/20 train/validation splitting.*

### 4. Evaluate the Model
Compute test loss, categorical accuracy, classification report, and confusion matrices:
```bash
python evaluate.py --model_path models/crop_disease_resnet50.h5 --data_dir data/test
```

---

## 📊 Dataset

The model is trained on the benchmark **New Plant Diseases Dataset (PlantVillage)**:
- **Source**: [Kaggle - New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Total Images**: ~87,000 augmented leaf images.
- **Classes**: 38 distinct crop health categories.
- **To train locally**: Download the dataset from Kaggle, extract the zip file, and place the directories into the `data/` folder.

---

## 💊 Agronomic Prescriptions

Unlike generic image classifiers, AgriVision AI couples computer vision predictions with actionable agricultural advisory data:
- **Scientific Etiology**: Explains the causal organism (e.g., *Alternaria solani*, *Phytophthora infestans*, *Venturia inaequalis*).
- **Organic & Biological Controls**: Recommends natural remedies suitable for organic growers (e.g., Neem oil, potassium bicarbonate, *Bacillus subtilis* biocontrols).
- **Targeted Chemical Controls**: Identifies standard agronomic active ingredients (e.g., Chlorothalonil, Mancozeb, Myclobutanil).
- **Preventative Culture**: Guides farmers on spacing, drip irrigation, crop rotation, and sanitization to stop future outbreaks.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author & Contributions
Developed by **Sparsh Tiwari** ([sparshtiwari544@gmail.com](mailto:sparshtiwari544@gmail.com)).  
Contributions, issues, and feature requests are welcome!
