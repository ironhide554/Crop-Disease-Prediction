# 🌱 Crop Disease & Pest Prediction (AgriVision AI)

[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.14%2B-orange.svg)](https://www.tensorflow.org/)
[![ResNet-50](https://img.shields.io/badge/Architecture-ResNet--50-success.svg)](https://arxiv.org/abs/1512.03385)
[![Streamlit UI](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-blue.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent deep learning computer vision system for automated crop disease diagnosis and agricultural pest classification using **MobileNetV3 (TensorFlow Lite)** and **ResNet-50**. 

The system analyzes crop leaf images, identifies **38 distinct crop disease & healthy conditions across 14 plant species**, and delivers actionable agronomic treatment prescriptions (symptoms, organic treatments, chemical controls, and preventative cultural practices).

- ⚡ **MobileNetV3 (TFLite)**: Ultra-fast, edge-optimized (~12 MB) model trained on PlantVillage achieving **~95% accuracy** with **~10–25 ms latency** for real-time edge, webcam, mobile, and Raspberry Pi diagnostics.
- 🧠 **ResNet-50 (Keras)**: Full 50-layer deep residual network (`models/crop_disease_resnet50.h5`, ~98 MB) for deep feature extraction.
- 🌐 **Production REST API (FastAPI)**: Real-time endpoints for mobile applications, IoT devices, and agricultural drones.
- 🐳 **Docker & Docker Compose**: 1-command microservices containerization ready for cloud deployment.

---

## 📌 Table of Contents
- [Overview & Features](#-overview--features)
- [Repository Structure](#-repository-structure)
- [Supported Crops & Diseases (38 Classes)](#-supported-crops--diseases-38-classes)
- [Requirements & Installation](#-requirements--installation)
- [Quick Start](#-quick-start)
  - [1. Production REST API (FastAPI)](#1-production-rest-api-fastapi)
  - [2. Interactive Web Application (Streamlit)](#2-interactive-web-application-streamlit)
  - [3. Docker & Docker Compose Deployment](#3-docker--docker-compose-deployment)
  - [4. Command-Line Inference](#4-command-line-inference)
  - [5. Model Training](#5-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
  - [7. Automated Testing Suite](#7-automated-testing-suite)
- [REST API Reference & Examples](#-rest-api-reference--examples)
- [Dataset](#-dataset)
- [Agronomic Prescriptions](#-agronomic-prescriptions)
- [License](#-license)

---

## 🚀 Overview & Features

- **Deep Residual Learning & Mobile Edge AI**: Employs MobileNetV3-Large (TFLite, 12MB) for low-latency edge computing and ResNet-50 for high-capacity transfer learning.
- **38 Plant Health Classes**: Covers major cash & food crops (Apple, Blueberry, Cherry, Corn/Maize, Grape, Orange, Peach, Pepper Bell, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato).
- **Production REST API (FastAPI)**: Fully async, CORS-enabled endpoints (`/predict`, `/predict/report`, `/classes`, `/crops`, `/health`) with interactive Swagger docs (`/docs`).
- **Interactive Streamlit Web Dashboard**: Live mobile/webcam camera capture, drag-and-drop leaf photo diagnostics, top-5 probability breakdown charts, and built-in benchmark samples.
- **Exportable PDF Field Reports**: One-click generation of branded, printable Agronomic Field Diagnostic Reports containing leaf photos, primary diagnoses, confidence metrics, and complete treatment prescriptions.
- **Containerized Microservices**: `Dockerfile` and `docker-compose.yml` orchestrating both API and Web UI with automated health checks.
- **Production CLI Pipeline**: Modular Python scripts for training (`train.py`), evaluation (`evaluate.py`), and inference (`predict.py`).
- **Comprehensive Test Suite**: Automated unit and integration tests covering API endpoints, model shapes, and probability sanity.

---

## 📂 Repository Structure

```text
Crop-Disease-Prediction/
├── api.py                      # Production FastAPI REST API
├── app.py                      # Interactive Streamlit Web Application (Camera + PDF Export)
├── predict.py                  # Standalone CLI prediction script
├── train.py                    # Standalone CLI model training pipeline
├── evaluate.py                 # Standalone CLI evaluation & metrics script
├── generate_samples.py         # Benchmark test image generator
├── Dockerfile                  # Multi-service production Docker container
├── docker-compose.yml          # Container orchestration (FastAPI + Streamlit)
├── docker-entrypoint.sh        # Smart entrypoint switcher
├── .dockerignore               # Docker build exclusions
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
│
├── src/                        # Core Python package
│   ├── __init__.py             # Package init & runtime compatibility polyfills
│   ├── model.py                # ResNet-50 & MobileNetV3 TFLite architectures
│   ├── disease_data.py         # 38-class metadata, symptoms, and remedies
│   ├── report_generator.py     # PDF Agronomic Field Report generator (ReportLab)
│   └── utils.py                # Preprocessing, top-k scoring & report formatting
│
├── samples/                    # Sample test images for instant verification
│   ├── real_leaf_sample.jpg
│   ├── sample_apple_scab.jpg
│   ├── sample_tomato_early_blight.jpg
│   ├── sample_healthy_corn.jpg
│   └── sample_potato_late_blight.jpg
│
├── tests/                      # Automated test suite
│   ├── __init__.py
│   ├── test_api.py             # FastAPI endpoint verification
│   └── test_inference.py       # Model loading, shapes & probability distribution tests
│
└── models/                     # Saved model weights directory (.h5 / .keras)
    └── (crop_disease_resnet50.h5)
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

### 1. Production REST API (FastAPI)
Start the REST API server on `http://localhost:8000`:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
- **Interactive Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Technical Reference**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. Interactive Web Application (Streamlit)
Launch the Streamlit web dashboard in your browser:
```bash
streamlit run app.py
```
- Open `http://localhost:8501`.
- Snap a live photo using **Live Camera Capture**, upload an image, or test with benchmark samples.
- Download the generated **PDF Diagnostic Field Report** with one click.

### 3. Docker & Docker Compose Deployment
Launch both the **FastAPI REST API** and **Streamlit Dashboard** as containerized services:
```bash
docker-compose up --build
```
- **FastAPI Endpoint**: `http://localhost:8000`
- **Streamlit Web UI**: `http://localhost:8501`

To run as background daemon:
```bash
docker-compose up -d
```

### 4. Command-Line Inference
Run diagnosis directly from terminal:
```bash
# Basic prediction with CLI summary report
python predict.py --image_path samples/real_leaf_sample.jpg

# Output formatted JSON (for integration with scripts or pipelines)
python predict.py --image_path samples/sample_apple_scab.jpg --json

# Specify custom trained weights and top-k predictions
python predict.py --image_path path/to/leaf.jpg --model_path models/crop_disease_resnet50.h5 --top_k 3
```

### 5. Model Training
Train the ResNet-50 model on your dataset:
```bash
python train.py --data_dir data/ --epochs 30 --batch_size 32 --lr 0.001 --save_path models/crop_disease_resnet50.h5
```
*Supports either dedicated `data/train/` and `data/valid/` folders or automatic 80/20 train/validation splitting.*

### 6. Model Evaluation
Compute test loss, categorical accuracy, classification report, and confusion matrices:
```bash
python evaluate.py --model_path models/crop_disease_resnet50.h5 --data_dir data/test
```

### 7. Automated Testing Suite
Run the automated test suite covering API endpoints and model inference sanity:
```bash
pytest tests/ -v
```

---

## 🌐 REST API Reference & Examples

### Predict Disease from Image (`POST /predict`)
```bash
curl -X POST "http://localhost:8000/predict?top_k=3" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@samples/real_leaf_sample.jpg"
```

**JSON Response:**
```json
{
  "success": true,
  "specimen_name": "real_leaf_sample.jpg",
  "engine": "MobileNetV3 TFLite Edge Model [MobileNetV3.tflite]",
  "latency_ms": 18.24,
  "timestamp": "2026-10-20T08:45:00.000",
  "primary_diagnosis": {
    "class_index": 34,
    "class_name": "Tomato - Target Spot",
    "confidence": 0.5203,
    "confidence_percentage": 52.03,
    "is_healthy": false,
    "details": {
      "crop": "Tomato",
      "disease": "Target Spot",
      "category": "Fungal (Corynespora cassiicola)",
      "symptoms": "Brown circular lesions with faint concentric rings...",
      "organic_treatment": "Copper fungicides and bio-fungicides; remove severely diseased leaves.",
      "chemical_treatment": "Boscalid, Pyraclostrobin, or Chlorothalonil.",
      "prevention": "Wide plant spacing; prune suckers to increase airflow..."
    }
  },
  "top_candidates": [...]
}
```

### Download Diagnostic Field Report as PDF (`POST /predict/report`)
```bash
curl -X POST "http://localhost:8000/predict/report" \
  -H "accept: application/pdf" \
  -F "file=@samples/real_leaf_sample.jpg" \
  --output diagnostic_report.pdf
```

### Check System Health (`GET /health`)
```bash
curl -X GET "http://localhost:8000/health"
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
