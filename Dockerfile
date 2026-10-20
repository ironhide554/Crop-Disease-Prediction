# Multi-purpose Production Container for AgriVision AI
# Supports both FastAPI REST API (port 8000) and Streamlit Web UI (port 8501)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_MODE=api
ENV PORT=8000

WORKDIR /app

# Install system dependencies for Pillow, OpenCV and ReportLab graphics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY src/ src/
COPY samples/ samples/
COPY api.py app.py predict.py train.py evaluate.py generate_samples.py MobileNetV3.tflite ./
COPY docker-entrypoint.sh .

# Ensure entrypoint is executable
RUN chmod +x docker-entrypoint.sh

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
