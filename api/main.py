import os
import sys
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

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

MODEL_PATH    = os.getenv("MODEL_PATH",    "data/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "tfidf-logreg-v1")

pipeline = None


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print(f"Loading model from {MODEL_PATH}...")
    pipeline = load_model(MODEL_PATH)      # ← now uses predict.py
    print("Model loaded successfully.")
    yield
    pipeline = None


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
        "name"    : "SentimentOps API",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "health"  : "/health",
        "predict" : "/predict",
        "batch"   : "/predict/batch",
    }