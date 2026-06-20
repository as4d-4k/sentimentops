import os, sys, joblib, mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","src"))
from preprocess import clean_text



# Request /Response model
class PredictRequest(BaseModel):
    text: str

    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v:str):
        if not v.strip():
            raise ValueError('text must not be empty')
        return v
    
class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

MODEL_PATH = os.getenv("MODEL_PATH", "data/model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", 'tfidf-logreg-v1')

pipeline = None

#LifeSpan
@asynccontextmanager
async def lifespan(app: FastAPI):
    #Load Model on startup, clean up on shutdown
    global pipeline
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
    
    pipeline = joblib.load(MODEL_PATH)
    print("Model Loaded Successfully")
    yield
    print("Shutting Down...")
    pipeline = None

# APP
app = FastAPI(
    title= "Sentiment-Ops",
    description= "Sentiment Analysis for Movie Reviews",
    version= "1.0.0",
    lifespan= lifespan,
)
#End Points

@app.get("/health", response_model= HealthResponse)
def health():
    #Check if the API and model are running.
    return HealthResponse(
        status= "ok",
        model_loaded= pipeline is not None,
    )

@app.post("/predict", response_model= PredictResponse)
def predict(request: PredictRequest):
    # Predicts Sentiment of a Movie review
    # Returns Sentiment (positive/negative) via confidence score.
    if pipeline is None:
        raise HTTPException(status_code= 503, detail="Model not Loaded")
    cleaned = clean_text(request.text)

    if not cleaned.strip():
        raise HTTPException(
            status_code= 422,
            detail= "Text is empty after cleaning"
        )
    
    #Predict
    prediction = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]
    confidence = float(probabilities[prediction])

    return PredictResponse(
        sentiment= "positive" if prediction == 1 else 'negative',
        confidence= round(confidence, 4),
        model_version= MODEL_VERSION,
    )

@app.get("/")
def root():
    return {
        "name": "SentimentOps API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": '/health',
        "predict": "/predict",
    }