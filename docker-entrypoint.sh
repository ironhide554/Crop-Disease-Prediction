#!/bin/sh
set -e

if [ "$APP_MODE" = "streamlit" ] || [ "$1" = "streamlit" ]; then
    echo "🌱 Starting AgriVision AI Streamlit Dashboard on port ${PORT:-8501}..."
    exec streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0 --server.headless=true
else
    echo "⚡ Starting AgriVision AI FastAPI REST API on port ${PORT:-8000}..."
    exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
fi
