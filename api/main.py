import os
import sys
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_model, predict_sentiment, predict_batch


# ── Request / Response Models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class BatchPredictRequest(BaseModel):
    texts: list[str]

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("texts list must not be empty")
        if len(v) > 100:
            raise ValueError("maximum 100 texts per batch request")
        return v


class PredictResponse(BaseModel):
    sentiment     : str
    confidence    : float
    model_version : str


class BatchPredictResponse(BaseModel):
    results       : list[PredictResponse]
    total         : int
    model_version : str


class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool


# ── App State ─────────────────────────────────────────────────────────────────

MODEL_PATH       = os.getenv("MODEL_PATH",       "data/model.joblib")
DISTILBERT_PATH  = os.getenv("DISTILBERT_PATH",  "data/distilbert_model")
MODEL_VERSION    = os.getenv("MODEL_VERSION",    "tfidf-logreg-v1")
BERT_VERSION     = os.getenv("BERT_VERSION",     "distilbert-v1")

pipeline              = None   # sklearn model
distilbert_model      = None   # distilbert model
distilbert_tokenizer  = None   # distilbert tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, distilbert_model, distilbert_tokenizer

    # ── Load sklearn model ────────────────────────────────────────────
    print(f"Loading sklearn model from {MODEL_PATH}...")
    pipeline = load_model(MODEL_PATH)
    print("Sklearn model loaded.")

    # ── Load DistilBERT (optional — skip if not found) ────────────────
    try:
        from predict import load_distilbert
        print(f"Loading DistilBERT from {DISTILBERT_PATH}...")
        distilbert_model, distilbert_tokenizer = load_distilbert(DISTILBERT_PATH)
        print("DistilBERT loaded.")
    except FileNotFoundError:
        print(f"DistilBERT not found at {DISTILBERT_PATH} — skipping.")
        print("Set DISTILBERT_PATH env var to load it.")

    yield
    pipeline             = None
    distilbert_model     = None
    distilbert_tokenizer = None

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "SentimentOps API",
    description = "Sentiment analysis API for movie reviews",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """Check if the API and model are running."""
    return HealthResponse(
        status       = "ok",
        model_loaded = pipeline is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict sentiment of a single movie review."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predict_sentiment(request.text, pipeline)  # ← one line now
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        sentiment     = result["sentiment"],
        confidence    = result["confidence"],
        model_version = MODEL_VERSION,
    )


@app.post("/predict/distilbert", response_model=PredictResponse)
def predict_distilbert_endpoint(request: PredictRequest):
    """
    Predict sentiment using fine-tuned DistilBERT.
    More accurate than /predict but slower (~50ms vs ~1ms).
    Requires DistilBERT model to be loaded at startup.
    """
    if distilbert_model is None or distilbert_tokenizer is None:
        raise HTTPException(
            status_code = 503,
            detail      = (
                "DistilBERT model not loaded. "
                "Set DISTILBERT_PATH env var and restart the server."
            )
        )

    try:
        from predict import predict_distilbert
        result = predict_distilbert(request.text, distilbert_model, distilbert_tokenizer)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictResponse(
        sentiment     = result["sentiment"],
        confidence    = result["confidence"],
        model_version = BERT_VERSION,
    )


@app.post("/predict/compare", response_model=dict)
def predict_compare(request: PredictRequest):
    """
    Run the same review through both models and compare results.
    Useful for seeing the difference between TF-IDF and DistilBERT.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Sklearn model not loaded")

    # sklearn prediction
    sklearn_result = predict_sentiment(request.text, pipeline)

    # distilbert prediction (if available)
    bert_result = None
    if distilbert_model is not None:
        from predict import predict_distilbert
        bert_result = predict_distilbert(request.text, distilbert_model, distilbert_tokenizer)

    return {
        "text"       : request.text[:100],
        "sklearn"    : {
            "sentiment"  : sklearn_result["sentiment"],
            "confidence" : sklearn_result["confidence"],
            "model"      : "tfidf-logreg-v1",
        },
        "distilbert" : {
            "sentiment"  : bert_result["sentiment"]   if bert_result else None,
            "confidence" : bert_result["confidence"]  if bert_result else None,
            "model"      : BERT_VERSION               if bert_result else None,
        } if bert_result else "not loaded",
    }


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch_endpoint(request: BatchPredictRequest):
    """
    Predict sentiment for multiple reviews in one request.
    Maximum 100 texts per request.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = predict_batch(request.texts, pipeline)  # ← uses predict.py
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return BatchPredictResponse(
        results = [
            PredictResponse(
                sentiment     = r["sentiment"],
                confidence    = r["confidence"],
                model_version = MODEL_VERSION,
            )
            for r in results
        ],
        total         = len(results),
        model_version = MODEL_VERSION,
    )


@app.get("/")
def root():
    return {
        "name"             : "SentimentOps API",
        "version"          : "1.0.0",
        "docs"             : "/docs",
        "health"           : "/health",
        "endpoints"        : {
            "sklearn"      : "POST /predict",
            "distilbert"   : "POST /predict/distilbert",
            "batch"        : "POST /predict/batch",
            "compare"      : "POST /predict/compare",
        }
    }