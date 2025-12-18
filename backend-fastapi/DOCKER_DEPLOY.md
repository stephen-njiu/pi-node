# Step-by-Step Guide: Build, Push, and Deploy FastAPI Backend

## Prerequisites

- Docker Desktop installed and running
- Logged into Docker Hub (`docker login` completed)
- Railway account ready

---

## Step 1: Navigate to the backend directory

```powershell
cd C:\Python_Programming\gate_access\backend-fastapi
```

---

## Step 2: Build the Docker image

Replace `<your-dockerhub-username>` with your actual Docker Hub username:

```powershell
docker build -t <your-dockerhub-username>/gate-backend:latest .
```

Example:

```powershell
docker build -t johndoe/gate-backend:latest .
```

This will:

- Use the Dockerfile in the current directory
- Install all Python dependencies
- Copy your application code
- Tag the image as `<your-dockerhub-username>/gate-backend:latest`

---

## Step 3: Test the image locally (optional but recommended)

```powershell
docker run -p 8000:8000 --env-file .env <your-dockerhub-username>/gate-backend:latest
```

Then visit: `http://localhost:8000/health`

You should see:

```json
{ "status": "ok", "backend": "onnx_arcface", "store": "pinecone" }
```

Press `Ctrl+C` to stop the container.

---

## Step 4: Push the image to Docker Hub

```powershell
docker push <your-dockerhub-username>/gate-backend:latest
```

Example:

```powershell
docker push johndoe/gate-backend:latest
```

This uploads your image to Docker Hub. It may take a few minutes depending on your internet speed.

---

## Step 5: Deploy to Railway

### 5.1 Create a new service on Railway

1. Go to https://railway.app
2. Click **"New Project"** → **"Deploy from Docker Image"**
3. Enter your Docker Hub image: `<your-dockerhub-username>/gate-backend:latest`

### 5.2 Configure environment variables

In Railway, go to your service → **Variables** tab and add:

**Required variables:**

```
FG_EMBEDDING_BACKEND=onnx_arcface
FG_ARC_FACE_ONNX_PATH=models/arcface.onnx
FG_ARC_FACE_ONNX_URL=https://github.com/onnx/models/raw/refs/heads/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx?download=
FG_INSIGHTFACE_ROOT=models
FG_INSIGHTFACE_MODEL_NAME=buffalo_l
FG_ALLOW_ORIGINS=http://localhost:3000,https://your-frontend-domain.com
FG_PINECONE_API_KEY=your_pinecone_api_key_here
FG_PINECONE_INDEX_HOST=https://your-index-host.pinecone.io
FG_PINECONE_NAMESPACE=__default__
PORT=8000
```

**Important:**

- Replace `your_pinecone_api_key_here` with your actual Pinecone API key
- Replace `https://your-index-host.pinecone.io` with your actual Pinecone index host
- Update `FG_ALLOW_ORIGINS` with your actual frontend URL(s)

### 5.3 Deploy

Railway will automatically deploy once you add the Docker image. Wait for the deployment to complete.

---

## Step 6: Verify deployment

Once Railway shows "Deployed" (green status):

1. Copy your Railway service URL (e.g., `https://your-service.railway.app`)
2. Visit: `https://your-service.railway.app/health`

You should see:

```json
{ "status": "ok", "backend": "onnx_arcface", "store": "pinecone" }
```

---

## Quick Reference Commands

```powershell
# Navigate to backend
cd C:\Python_Programming\gate_access\backend-fastapi

# Build image
docker build -t <your-dockerhub-username>/gate-backend:latest .

# Test locally (optional)
docker run -p 8000:8000 --env-file .env <your-dockerhub-username>/gate-backend:latest

# Push to Docker Hub
docker push <your-dockerhub-username>/gate-backend:latest
```

---

## Troubleshooting

**Build fails:**

- Ensure you're in the `backend-fastapi` directory
- Check that `requirements.txt` and `extras-ml.txt` exist
- Verify Docker Desktop is running

**Push fails:**

- Run `docker login` and enter your credentials
- Ensure the image name includes your Docker Hub username

**Railway deployment fails:**

- Verify all environment variables are set correctly
- Check Railway logs for specific error messages
- Ensure Pinecone API key and index host are correct

**Health check fails:**

- Check Railway logs for startup errors
- Verify the ArcFace model downloads successfully
- Confirm Pinecone connection is working

---

## Updating your deployment

When you make code changes:

1. Rebuild the image:

   ```powershell
   docker build -t <your-dockerhub-username>/gate-backend:latest .
   ```

2. Push the new version:

   ```powershell
   docker push <your-dockerhub-username>/gate-backend:latest
   ```

3. In Railway, trigger a redeploy (it may auto-deploy if watching for image changes)

---

## Notes

- The `.env` file is excluded from the Docker image for security
- All configuration happens via Railway environment variables
- The image auto-downloads ArcFace and InsightFace models on first run
- Initial startup may take 30-60 seconds while models download
