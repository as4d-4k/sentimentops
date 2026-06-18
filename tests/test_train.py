import pytest
from src.train import build_pipeline, evaluate
import pandas as pd


def test_pipeline_builds():
    """Pipeline should build without errors."""
    pipeline = build_pipeline(max_features=1000, ngram_range=(1,1), C=1.0)
    assert pipeline is not None


def test_pipeline_predicts():
    """Pipeline should train and predict on small data."""
    pipeline = build_pipeline(max_features=1000, ngram_range=(1,1), C=1.0)

    X_train = pd.Series(["great movie", "terrible film", "loved it", "hated it"])
    y_train = pd.Series([1, 0, 1, 0])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_train)

    assert len(predictions) == 4
    assert set(predictions).issubset({0, 1})


def test_evaluate_returns_accuracy():
    """Evaluate should return accuracy between 0 and 1."""
    pipeline = build_pipeline(max_features=1000, ngram_range=(1,1), C=1.0)

    X = pd.Series(["great movie", "terrible film", "loved it", "hated it"])
    y = pd.Series([1, 0, 1, 0])

    pipeline.fit(X, y)
    metrics = evaluate(pipeline, X, y)

    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0