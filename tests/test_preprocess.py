import pytest
from src.preprocess import clean_text, preprocess
import pandas as pd


def test_clean_text_lowercase():
    """Text should be lowercased."""
    assert clean_text("GREAT Movie") == "great movie"


def test_clean_text_removes_html():
    """HTML tags should be removed."""
    assert clean_text("<br />great film") == "great film"


def test_clean_text_removes_special_chars():
    """Special characters should be removed."""
    assert clean_text("great!!!") == "great"


def test_clean_text_strips_whitespace():
    """Extra whitespace should be collapsed."""
    assert clean_text("great   movie") == "great movie"


def test_preprocess_adds_columns():
    """Preprocess should add text_raw and text_length columns."""
    df = pd.DataFrame({
        "text" : ["Great movie!", "Terrible film."],
        "label": [1, 0]
    })
    result = preprocess(df)
    assert "text_raw"    in result.columns
    assert "text_length" in result.columns


def test_preprocess_preserves_labels():
    """Labels should not be modified by preprocessing."""
    df = pd.DataFrame({
        "text" : ["Great movie!", "Terrible film."],
        "label": [1, 0]
    })
    result = preprocess(df)
    assert list(result["label"]) == [1, 0]