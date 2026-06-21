import pytest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_sentiment, predict_batch


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.predict.return_value       = [1]
    mock.predict_proba.return_value = np.array([[0.1, 0.9]])
    return mock


def test_predict_sentiment_positive(mock_pipeline):
    result = predict_sentiment("great movie", mock_pipeline)
    assert result["sentiment"]  == "positive"
    assert result["confidence"] == 0.9
    assert result["label"]      == 1
    assert "cleaned" in result


def test_predict_sentiment_negative(mock_pipeline):
    mock_pipeline.predict.return_value       = [0]
    mock_pipeline.predict_proba.return_value = np.array([[0.85, 0.15]])
    result = predict_sentiment("terrible movie", mock_pipeline)
    assert result["sentiment"]  == "negative"
    assert result["confidence"] == 0.85
    assert result["label"]      == 0


def test_predict_sentiment_empty_text(mock_pipeline):
    with pytest.raises(ValueError):
        predict_sentiment("   ", mock_pipeline)


def test_predict_batch(mock_pipeline):
    mock_pipeline.predict.return_value       = [1, 0]
    mock_pipeline.predict_proba.return_value = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    results = predict_batch(["great movie", "terrible film"], mock_pipeline)
    assert len(results)           == 2
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"


def test_predict_batch_empty(mock_pipeline):
    with pytest.raises(ValueError):
        predict_batch([], mock_pipeline)