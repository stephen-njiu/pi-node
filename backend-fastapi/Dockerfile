# syntax=docker/dockerfile:1
# FastAPI backend for Gate Access

FROM python:3.11-slim AS base

# -------------------------
# Environment
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -------------------------
# System deps (for opencv, onnxruntime, insightface)
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Set working directory
# -------------------------
WORKDIR /app/backend-fastapi

# -------------------------
# Install Python dependencies first for caching
# -------------------------
COPY backend-fastapi/requirements.txt backend-fastapi/extras-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r extras-ml.txt

# -------------------------
# Copy backend contents (not the folder itself)
# -------------------------
COPY backend-fastapi/ . 

# -------------------------
# Expose FastAPI port
# -------------------------
EXPOSE 8000

# -------------------------
# Default environment (can override in Railway)
# -------------------------
ENV FG_EMBEDDING_BACKEND=onnx_arcface \
    FG_ARC_FACE_ONNX_PATH=models/arcface.onnx \
    FG_INSIGHTFACE_ROOT=models \
    FG_PINECONE_NAMESPACE=__default__

# -------------------------
# Start FastAPI server
# -------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
