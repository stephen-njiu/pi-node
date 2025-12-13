# Backend FastAPI – Docker + Railway Deployment

This guide packages the backend (`backend-fastapi`) as a Docker image, pushes it to Docker Hub, and deploys it on Railway using that image.

## Prerequisites

- Docker installed locally
- A Docker Hub account (username and PAT)
- A Railway account

## 1) Build and tag the Docker image

From the `backend-fastapi` folder:

```powershell
# Windows PowerShell
# Navigate to backend-fastapi
cd C:\Python_Programming\gate_access\backend-fastapi

# Build and tag image (replace <your-dockerhub-username>)
docker build -t <your-dockerhub-username>/gate-backend:latest .
```

## 2) Login and push to Docker Hub

```powershell
# Login (will prompt for username and password token)
docker login

# Push the image
docker push <your-dockerhub-username>/gate-backend:latest
```

## 3) Deploy on Railway (using Docker Hub image)

In Railway:

1. Create a new project (Service) → "Deploy from Docker Hub"
2. Provide the image: `<your-dockerhub-username>/gate-backend:latest`
3. Set the service "PORT" to `8000`
4. Add Environment Variables (under Service Settings → Variables):

Required (match your setup):

- `FG_EMBEDDING_BACKEND=onnx_arcface`
- `FG_ARC_FACE_ONNX_PATH=models/arcface.onnx`
- `FG_ARC_FACE_ONNX_URL=https://github.com/onnx/models/raw/refs/heads/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx?download=`
- `FG_INSIGHTFACE_ROOT=models`
- `FG_INSIGHTFACE_MODEL_NAME=buffalo_l`
- `FG_ALLOW_ORIGINS=http://localhost:3000,https://yourfrontend.example`

Pinecone:

- `FG_PINECONE_API_KEY=<your-pinecone-api-key>`
- `FG_PINECONE_INDEX_HOST=<your-index-host>`
- `FG_PINECONE_NAMESPACE=__default__` # or your chosen namespace

Notes:

- Ensure your Pinecone index is a dense index with dimension `512` and cosine metric.
- The backend will auto-download the ArcFace ONNX to `models/arcface.onnx` if the file is missing and the URL is set.
- InsightFace (SCRFD) will auto-download detection models into `models/` at first run.

## 4) Health check and routes

- Health: `GET /health`
- Embeddings: `POST /api/v1/embeddings`

Expose `8000` on Railway and set a health check to `/health` (optional).

## 5) Troubleshooting

- Container fails to start:

  - Verify ONNX URL and path (`FG_ARC_FACE_ONNX_*`).
  - Check Pinecone host and API key.

- No vectors visible in Pinecone:

  - Confirm you’re viewing the correct index host and namespace (`__default__` vs custom).
  - Ensure index dimension is `512`.

- CORS errors:
  - Update `FG_ALLOW_ORIGINS` to include your frontend URL(s).

## 6) Optional: Different tags and rollbacks

- Tag image with version to allow rollbacks:

```powershell
docker build -t <your-dockerhub-username>/gate-backend:v1 .
docker push <your-dockerhub-username>/gate-backend:v1
```

Then point Railway to `:v1` instead of `:latest`.
