# SentimentOps — End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://docker.com)
[![Azure](https://img.shields.io/badge/Azure-Container%20Apps-blue)](https://azure.microsoft.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow)](https://huggingface.co/asadullahrehmann/imdb-distilbert-sentimentops)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-grade MLOps pipeline for sentiment analysis — covering
everything from raw data to a live cloud API. Built to demonstrate
end-to-end ML engineering skills beyond just model training.

## Live Demo

| Resource | Link |
|---|---|
| Live API | https://sentimentops-api.thankfulmeadow-07d26880.uaenorth.azurecontainerapps.io |
| API Docs | https://sentimentops-api.thankfulmeadow-07d26880.uaenorth.azurecontainerapps.io/docs |
| HuggingFace Model | https://huggingface.co/asadullahrehmann/imdb-distilbert-sentimentops |
| GitHub | https://github.com/as4d-4k/sentimentops |

---

## What This Project Does

Classifies movie reviews as positive or negative using two models:

```bash
# try it right now
curl -X POST https://sentimentops-api.thankfulmeadow-07d26880.uaenorth.azurecontainerapps.io/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic"}'

# response
{"sentiment": "positive", "confidence": 0.9976, "model_version": "tfidf-logreg-v1"}
```

---

## Model Performance

| Model | Accuracy | Inference | Training |
|---|---|---|---|
| TF-IDF + Logistic Regression | 90.0% | ~1ms | 2 min (CPU) |
| DistilBERT (fine-tuned) | 97.2% | ~50ms | 50+ min (T4 GPU) |

The difference becomes clear on ambiguous reviews:

```json
POST /predict/compare
{"text": "Not bad but not great either, just average"}

{
  "sklearn":    {"sentiment": "negative", "confidence": 0.7562},
  "distilbert": {"sentiment": "negative", "confidence": 0.5243}
}
```

DistilBERT correctly expresses uncertainty (52%) on genuinely
ambiguous text while TF-IDF is overconfident (75%).

---

## Architecture

```
Data (HuggingFace IMDB)
        ↓
Preprocessing (src/preprocess.py)
        ↓
    ┌───────────────────────────────┐
    │                               │
Stage 1: TF-IDF + LogReg     Stage 2: DistilBERT
sklearn Pipeline             PyTorch fine-tuning
90% accuracy                 97.2% accuracy
    │                               │
    └───────────────┬───────────────┘
                    ↓
            MLflow Tracking
                    ↓
            FastAPI REST API
            ├── POST /predict
            ├── POST /predict/distilbert
            ├── POST /predict/batch
            └── POST /predict/compare
                    ↓
            Docker Container
                    ↓
        Azure Container Apps
        (Live public HTTPS endpoint)
```

---

## Project Structure

```
sentimentops/
├── src/
│   ├── preprocess.py      # data cleaning pipeline
│   ├── train.py           # TF-IDF + LogReg training
│   ├── train_bert.py      # DistilBERT fine-tuning
│   ├── predict.py         # inference — single source of truth
│   └── cli.py             # Click CLI tool
├── api/
│   └── main.py            # FastAPI application
├── azure/
│   ├── create_workspace.py
│   ├── submit_job.py      # TF-IDF Azure ML job
│   ├── submit_bert_job.py # DistilBERT Azure ML job
│   └── deploy.py          # Container Apps deployment
├── tests/
│   ├── test_preprocess.py
│   ├── test_train.py
│   ├── test_predict.py
│   ├── test_api.py
│   └── test_cli.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| ML — Classical | scikit-learn (TF-IDF + LogReg) |
| ML — Transformer | PyTorch + HuggingFace Transformers |
| Experiment Tracking | MLflow |
| API | FastAPI + uvicorn |
| CLI | Click |
| Containerization | Docker |
| Cloud Training | Azure ML (CPU + GPU clusters) |
| Cloud Deployment | Azure Container Apps |
| Model Registry | HuggingFace Hub |
| Testing | Pytest + pytest-cov |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker Desktop
- Azure CLI (for cloud features)

### Run Locally

```bash
# clone
git clone https://github.com/as4d-4k/sentimentops.git
cd sentimentops

# setup
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
pip install -e .

# preprocess data
sentimentops preprocess-data

# train sklearn model
sentimentops train-model

# start API
uvicorn api.main:app --reload

# visit http://localhost:8000/docs
```

### Run with Docker

```bash
docker build -t sentimentops:latest .
docker run -p 8000:8000 sentimentops:latest
# visit http://localhost:8000/docs
```

### CLI Usage

```bash
# predict
sentimentops predict "This movie was absolutely fantastic"
# → POSITIVE (96.21%)

# predict with DistilBERT
sentimentops predict "Great film" --model-type distilbert

# evaluate
sentimentops evaluate-model

# submit cloud training job
sentimentops cloud-train --max-features 50000 --c 1.0
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Sklearn prediction |
| POST | `/predict/distilbert` | DistilBERT prediction |
| POST | `/predict/batch` | Batch predictions (max 100) |
| POST | `/predict/compare` | Side-by-side model comparison |

---

## MLOps Pipeline

```
Local Development
├── src/train.py           → TF-IDF training with MLflow tracking
├── mlflow ui              → compare experiments locally
└── pytest tests/          → 19 tests, full coverage

Cloud Training
├── sentimentops cloud-train     → submit TF-IDF job to Azure ML
├── python azure/submit_bert_job.py → submit DistilBERT to GPU cluster
└── Azure ML Studio        → monitor jobs, view metrics

Deployment
├── docker build           → containerize the API
├── docker push            → push to Azure Container Registry
└── az containerapp create → deploy to Azure Container Apps
```

---

## Key Engineering Decisions

**Why two models?**
TF-IDF provides a fast, interpretable baseline (1ms inference).
DistilBERT provides higher accuracy for cases where speed
is less critical (50ms inference). The `/predict/compare`
endpoint lets users see both simultaneously.

**Why centralize inference in predict.py?**
Both the API and CLI call the same `predict_sentiment()`
and `predict_distilbert()` functions. Fixing a bug in one
place fixes it everywhere.

**Why lazy imports in preprocess.py?**
The `datasets` library is only needed for training, not
inference. Lazy-importing it inside `load_imdb()` keeps
the Docker image 500MB smaller.

**Why `AZUREML_RUN_ID` detection?**
MLflow's model registry API differs between local and Azure.
Detecting the Azure environment automatically uses the right
logging approach without manual configuration.

---

## Future Improvements

- Azure ML Data Assets for versioned dataset storage
- CI/CD pipeline with GitHub Actions
- Model drift detection and automatic retraining
- Load model from MLflow registry instead of local file
- A/B testing endpoint routing between model versions
- Request rate limiting and API authentication

---

## Author

**Asad Ullah**
AI/ML Engineer | CS Student at Forman Christian College

- GitHub: https://github.com/as4d-4k
- Portfolio: https://portfolio-steel-nine-aur.vercel.app
- HuggingFace: https://huggingface.co/asadullahrehmann
