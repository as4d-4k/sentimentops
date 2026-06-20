import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np


# ── Setup ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """Create a mock sklearn pipeline for testing."""
    mock = MagicMock()
    mock.predict.return_value = [1]                          # positive
    mock.predict_proba.return_value = np.array([[0.1, 0.9]]) # 90% confidence
    return mock


@pytest.fixture
def client(mock_pipeline):
    """Create test client with mocked model."""
    with patch("api.main.pipeline", mock_pipeline):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """Health endpoint should return ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_positive(client):
    """Positive review should return positive sentiment."""
    response = client.post(
        "/predict",
        json={"text": "This movie was absolutely fantastic"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model_version" in data


def test_predict_empty_text(client):
    """Empty text should return 422 validation error."""
    response = client.post(
        "/predict",
        json={"text": "   "}
    )
    assert response.status_code == 422


def test_predict_missing_text(client):
    """Missing text field should return 422."""
    response = client.post(
        "/predict",
        json={}
    )
    assert response.status_code == 422


def test_root_endpoint(client):
    """Root endpoint should return API info."""
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()