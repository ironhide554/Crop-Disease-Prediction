"""
test_api.py - Pytest test suite for FastAPI REST API endpoints.
"""

import sys
import asyncio
import httpx
import pytest

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from api import app, preload_model


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Ensure model is preloaded for the test session."""
    preload_model()


def test_root_endpoint():
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert "AgriVision AI" in data["title"]
            assert data["docs_url"] == "/docs"
    asyncio.run(_run())


def test_health_endpoint():
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["num_classes"] == 38
            assert "MobileNet" in data["engine"] or "ResNet" in data["engine"]
    asyncio.run(_run())


def test_crops_endpoint():
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/crops")
            assert response.status_code == 200
            data = response.json()
            assert data["total_crops"] >= 14
            assert "Tomato" in data["crops"]
            assert "Apple" in data["crops"]
    asyncio.run(_run())


def test_classes_list_and_filter():
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # All classes
            r_all = await client.get("/classes")
            assert r_all.status_code == 200
            assert len(r_all.json()) == 38

            # Filter by Tomato
            r_tomato = await client.get("/classes?crop=Tomato")
            assert r_tomato.status_code == 200
            tomato_classes = r_tomato.json()
            assert len(tomato_classes) == 10
            assert all(c["crop"] == "Tomato" for c in tomato_classes)
    asyncio.run(_run())


def test_single_class_details():
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/classes/34")
            assert response.status_code == 200
            data = response.json()
            assert data["crop"] == "Tomato"
            assert data["disease"] == "Target Spot"
            assert "organic_treatment" in data
            assert "chemical_treatment" in data
            assert "symptoms" in data
    asyncio.run(_run())


def test_predict_endpoint_with_sample():
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with open("samples/real_leaf_sample.jpg", "rb") as f:
                files = {"file": ("real_leaf_sample.jpg", f.read(), "image/jpeg")}
                response = await client.post("/predict?top_k=5", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["top_candidates"]) == 5
            assert data["latency_ms"] > 0
            assert "primary_diagnosis" in data
            assert data["primary_diagnosis"]["confidence_percentage"] > 0
    asyncio.run(_run())


def test_predict_report_endpoint():
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with open("samples/real_leaf_sample.jpg", "rb") as f:
                files = {"file": ("real_leaf_sample.jpg", f.read(), "image/jpeg")}
                response = await client.post("/predict/report", files=files)

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.content.startswith(b"%PDF")
            assert len(response.content) > 10000
    asyncio.run(_run())
