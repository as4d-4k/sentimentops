FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# ── Install torch CPU separately first ────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# ── Install everything else ───────────────────────────────────────────
RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.4.2 \
        fastapi \
        uvicorn \
        joblib \
        python-dotenv \
        transformers \
        huggingface_hub

# ── Copy Application Code ─────────────────────────────────────────────
COPY src/ ./src/
COPY api/ ./api/

# ── Copy Models ───────────────────────────────────────────────────────
COPY data/model.joblib         ./data/model.joblib
COPY data/distilbert_model/    ./data/distilbert_model/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]