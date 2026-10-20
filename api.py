"""
api.py - Enterprise-grade FastAPI REST API for Crop Disease Prediction.
Enables real-time inference for web apps, mobile clients (iOS/Android),
and edge IoT devices (Raspberry Pi, drones) via HTTP POST.

Features:
- Preloaded MobileNetV3 TFLite engine (~10-25 ms latency)
- Full CORS support for cross-origin frontend requests
- OpenAPI / Swagger documentation at /docs and ReDoc at /redoc
- PDF report generation endpoint (/predict/report)
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import io
import time
import datetime
from contextlib import asynccontextmanager
from typing import List, Optional
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import load_trained_model, build_crop_disease_model
from src.disease_data import CLASS_MAPPING, DISEASE_DETAILS, get_disease_info
from src.utils import predict_top_k
from src.report_generator import generate_pdf_report


# ==========================================
# Pydantic Response Schemas
# ==========================================

class DiseaseDetailsSchema(BaseModel):
    crop: str
    disease: str
    category: str
    symptoms: str
    organic_treatment: str
    chemical_treatment: str
    prevention: str


class PredictionItemSchema(BaseModel):
    class_index: int
    class_name: str
    confidence: float
    confidence_percentage: float
    is_healthy: bool
    details: DiseaseDetailsSchema


class PredictionResponseSchema(BaseModel):
    success: bool = True
    specimen_name: str
    engine: str
    latency_ms: float
    timestamp: str
    primary_diagnosis: PredictionItemSchema
    top_candidates: List[PredictionItemSchema]


class HealthResponseSchema(BaseModel):
    status: str = "healthy"
    engine: str
    num_classes: int = 38
    timestamp: str


class ClassInfoSchema(BaseModel):
    index: int
    class_name: str
    crop: str
    condition: str
    category: str


# ==========================================
# Application & State Initialization
# ==========================================

MODEL_STATE = {
    "model": None,
    "engine_name": "Uninitialized",
    "model_path": None
}


def find_best_model_path() -> str:
    """Finds best available trained model in workspace."""
    candidates = [
        "MobileNetV3.tflite",
        os.path.join("models", "crop_disease_resnet50.h5"),
        "crop_disease_resnet50.h5"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "MobileNetV3.tflite"


def preload_model():
    """Preloads the model into memory for zero-latency inferences."""
    path = find_best_model_path()
    if os.path.exists(path):
        try:
            MODEL_STATE["model"] = load_trained_model(path)
            MODEL_STATE["model_path"] = path
            if path.lower().endswith(".tflite"):
                MODEL_STATE["engine_name"] = f"MobileNetV3 TFLite Edge Model [{os.path.basename(path)}]"
            else:
                MODEL_STATE["engine_name"] = f"ResNet-50 Keras Model [{os.path.basename(path)}]"
            print(f"Preloaded active model: {MODEL_STATE['engine_name']}")
            return
        except Exception as e:
            print(f"Warning: Error loading model from {path}: {e}")

    # Fallback to base model
    print("Initializing base ResNet-50 architecture...")
    MODEL_STATE["model"] = build_crop_disease_model(num_classes=38)
    MODEL_STATE["engine_name"] = "Base ResNet-50 (ImageNet Initialized)"
    MODEL_STATE["model_path"] = "in-memory"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI lifespan context manager for startup and shutdown events."""
    preload_model()
    yield
    MODEL_STATE["model"] = None


app = FastAPI(
    title="🌱 AgriVision AI — Crop Disease & Pest Prediction REST API",
    description="High-performance deep learning REST API for automated crop leaf disease diagnosis and agronomic management.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Endpoints
# ==========================================

@app.get("/", tags=["General"])
def root():
    """Welcome root endpoint directing to documentation."""
    return {
        "title": "🌱 AgriVision AI — Crop Disease Prediction API",
        "version": "1.0.0",
        "active_engine": MODEL_STATE["engine_name"],
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponseSchema, tags=["General"])
def health_check():
    """Health check endpoint returning model readiness status."""
    return {
        "status": "healthy" if MODEL_STATE["model"] is not None else "degraded",
        "engine": MODEL_STATE["engine_name"],
        "num_classes": len(CLASS_MAPPING),
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.get("/classes", response_model=List[ClassInfoSchema], tags=["Dataset & Metadata"])
def list_classes(crop: Optional[str] = Query(None, description="Optional filter by crop name (e.g. Tomato, Apple)")):
    """Returns all 38 supported crop disease and healthy classes."""
    results = []
    for idx, details in DISEASE_DETAILS.items():
        if crop is None or crop.lower() in details["crop"].lower():
            results.append({
                "index": idx,
                "class_name": CLASS_MAPPING.get(idx, f"Class {idx}"),
                "crop": details["crop"],
                "condition": details["disease"],
                "category": details["category"]
            })
    return results


@app.get("/classes/{class_index}", response_model=DiseaseDetailsSchema, tags=["Dataset & Metadata"])
def get_class_details(class_index: int):
    """Retrieves detailed agronomic profile and treatment options for a specific class index (0-37)."""
    if class_index not in DISEASE_DETAILS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class index {class_index} not found. Must be between 0 and 37."
        )
    return get_disease_info(class_index)


@app.get("/crops", tags=["Dataset & Metadata"])
def list_crops():
    """Lists all 14 unique crop species supported by the system."""
    crops = sorted(list(set(d["crop"] for d in DISEASE_DETAILS.values())))
    return {"total_crops": len(crops), "crops": crops}


@app.post("/predict", response_model=PredictionResponseSchema, tags=["Inference"])
async def predict_crop_disease(
    file: UploadFile = File(..., description="Crop leaf image file (JPG, PNG, WebP)"),
    top_k: int = Query(5, ge=1, le=38, description="Number of top candidates to return")
):
    """
    Submits a crop leaf image for real-time disease diagnosis.
    Returns primary diagnosis, statistical confidence, top alternative candidates,
    symptoms, organic biocontrols, chemical fungicides, and prevention advice.
    """
    if MODEL_STATE["model"] is None:
        preload_model()

    # Validate content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{file.filename}' is not an image (MIME: {file.content_type})."
        )

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}"
        )

    # Perform inference and record latency
    start_t = time.perf_counter()
    raw_predictions = predict_top_k(MODEL_STATE["model"], pil_image, top_k=top_k)
    latency_ms = (time.perf_counter() - start_t) * 1000

    # Format response
    formatted_candidates = []
    for item in raw_predictions:
        is_healthy = "healthy" in item["class_name"].lower()
        formatted_candidates.append({
            "class_index": item["class_index"],
            "class_name": item["class_name"],
            "confidence": item["confidence"],
            "confidence_percentage": item["confidence_percentage"],
            "is_healthy": is_healthy,
            "details": item["details"]
        })

    primary = formatted_candidates[0]

    return {
        "success": True,
        "specimen_name": file.filename,
        "engine": MODEL_STATE["engine_name"],
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.datetime.now().isoformat(),
        "primary_diagnosis": primary,
        "top_candidates": formatted_candidates
    }


@app.post("/predict/report", tags=["Inference"])
async def predict_and_generate_report(
    file: UploadFile = File(..., description="Crop leaf image file (JPG, PNG, WebP)"),
    top_k: int = Query(5, ge=1, le=38, description="Number of top candidates in report")
):
    """
    Submits a crop leaf image, runs disease diagnosis, and returns
    a ready-to-print branded Agronomic Field Diagnostic Report in PDF format.
    """
    if MODEL_STATE["model"] is None:
        preload_model()

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{file.filename}' is not an image."
        )

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {str(e)}"
        )

    start_t = time.perf_counter()
    raw_predictions = predict_top_k(MODEL_STATE["model"], pil_image, top_k=top_k)
    latency_ms = (time.perf_counter() - start_t) * 1000

    pdf_bytes = generate_pdf_report(
        predictions=raw_predictions,
        leaf_image=pil_image,
        model_name=MODEL_STATE["engine_name"],
        latency_ms=latency_ms,
        specimen_name=file.filename
    )

    crop_name = raw_predictions[0]["details"]["crop"]
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"AgriVision_Report_{crop_name}_{timestamp_str}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
