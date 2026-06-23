import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str = "data/model.joblib"):
    """
    Load and return the trained pipeline from disk.
    Raises FileNotFoundError if model doesn't exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            f"Run 'sentimentops train-model' first."
        )
    return joblib.load(model_path)


# ── Core Prediction Logic ─────────────────────────────────────────────────────

def predict_sentiment(text: str, pipeline) -> dict:
    """
    Core prediction logic — single source of truth.

    Called by the API, CLI, and any future interface.
    Returns a dict so each interface can format output its own way.

    Args:
        text     : raw review text (will be cleaned internally)
        pipeline : fitted sklearn Pipeline

    Returns:
        {
            "sentiment"  : "positive" or "negative",
            "confidence" : float between 0 and 1,
            "label"      : 0 or 1,
            "cleaned"    : the cleaned version of the input text
        }
    """
    if not text or not text.strip():
        raise ValueError("Input text must not be empty")

    cleaned       = clean_text(text)
    prediction    = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]
    confidence    = float(probabilities[prediction])

    return {
        "sentiment"  : "positive" if prediction == 1 else "negative",
        "confidence" : round(confidence, 4),
        "label"      : int(prediction),
        "cleaned"    : cleaned,
    }


# ── Batch Prediction ──────────────────────────────────────────────────────────

def predict_batch(texts: list[str], pipeline) -> list[dict]:
    """
    Predict sentiment for a list of texts.
    More efficient than calling predict_sentiment() in a loop.

    Args:
        texts    : list of raw review strings
        pipeline : fitted sklearn Pipeline

    Returns:
        list of prediction dicts, one per input text
    """
    if not texts:
        raise ValueError("texts list must not be empty")

    cleaned_texts = [clean_text(t) for t in texts]
    predictions   = pipeline.predict(cleaned_texts)
    probabilities = pipeline.predict_proba(cleaned_texts)

    results = []
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        confidence = float(probs[pred])
        results.append({
            "sentiment"  : "positive" if pred == 1 else "negative",
            "confidence" : round(confidence, 4),
            "label"      : int(pred),
            "cleaned"    : cleaned_texts[i],
        })

    return results



# ____DistilBERT_____________________________________________
def load_distilbert(model_path: str = "asadullahrehmann/imdb-distilbert-sentimentops"):
    """
    Load DistilBERT from either:
    - local path:      "data/distilbert_model"
    - HuggingFace Hub: "asadullahrehmann/imdb-distilbert-sentimentops"
    """
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )
    

    # works for both local path and HuggingFace Hub ID
    print(f"Loading DistilBERT from: {model_path}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model     = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    return model, tokenizer


def predict_distilbert(text: str, model, tokenizer):
    """
    Predict sentiment using fine-tuned DistilBERT.
    Single source of truth for transformer inference.
    """
    import torch

    if not text or not text.strip():
        raise ValueError("Input text must not be empty")

    # tokenize
    inputs = tokenizer(
        text,
        truncation  = True,
        padding     = True,
        max_length  = 512,
        return_tensors = "pt",   # return PyTorch tensors
    )

    # predict
    with torch.no_grad():        # no gradient computation needed for inference
        outputs = model(**inputs)

    # convert logits to probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction    = probabilities.argmax().item()
    confidence    = probabilities[0][prediction].item()

    return {
        "sentiment"  : "positive" if prediction == 1 else "negative",
        "confidence" : round(confidence, 4),
        "label"      : int(prediction),
        "model_type" : "distilbert",
    }
