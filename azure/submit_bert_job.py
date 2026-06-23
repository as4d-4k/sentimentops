def load_distilbert(model_path: str = "data/distilbert_model"):
    """Load fine-tuned DistilBERT model and tokenizer."""
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )
    import torch

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DistilBERT model not found at '{model_path}'. "
            f"Run 'python src/train_bert.py' first."
        )

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model     = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    return model, tokenizer


def predict_distilbert(text: str, model, tokenizer) -> dict:
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