# ── Stage 1: Base Image ───────────────────────────────────────────────
FROM python:3.11-slim

# ── System Setup ──────────────────────────────────────────────────────
# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python from buffering stdout/stderr
# This ensures logs appear immediately in docker logs
ENV PYTHONUNBUFFERED=1

# ── Install Dependencies ───────────────────────────────────────────────
# Copy requirements first — Docker layer caching
# If requirements.txt doesn't change, this layer is cached
# and pip install doesn't rerun on every build
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.4.2 \
        fastapi \
        uvicorn \
        joblib \
        python-dotenv 

# ── Copy Application Code ─────────────────────────────────────────────
COPY src/ ./src/
COPY api/ ./api/

# ── Copy Model ────────────────────────────────────────────────────────
COPY data/model.joblib ./data/model.joblib

# ── Expose Port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Start Command ─────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]