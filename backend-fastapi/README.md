# Backend (FastAPI)

Minimal API to accept up to 5 images and store their face embeddings.

## Endpoints

- `GET /health` – Simple health check.
- `POST /api/v1/embeddings` – Multipart form upload of up to 5 images. Returns vectors and stores them (local JSONL by default).
  - Query params:
    - `augment` (bool): generate variants per image (default: true).
    - `aug_per_image` (int ≤ 5): number of variants per image (default: 3).

### Face detection vs whole-image

The backend performs face detection with SCRFD (via `insightface`) and 5-point alignment to 112×112 before augmentation and embedding. If detection fails, it falls back to the original image.
Default behavior is to treat uploads as raw camera images and run detection+alignment (i.e., `FG_ASSUME_ALIGNED=false`).
Set `FG_ASSUME_ALIGNED=true` only if you already provide cropped/aligned faces.

### Augmentation policy (gate access)

We apply mild, identity-preserving transforms per image (up to `aug_per_image`):

- Geometric: small rotations (±5–10°), tiny translations (±4 px), mild scale (0.95–1.05), optional horizontal flip (off by default)
- Photometric: brightness/contrast jitter (±10–20%), mild color shift
- Artifacts: low Gaussian noise, slight blur
  These aim to mimic real camera variations without drifting from identity features.

  ### Optional ML dependencies

  Install extras to enable detection and ONNX-based embeddings:

  ```powershell
  pip install -r extras-ml.txt
  ```

  This includes `insightface` (SCRFD) and `onnxruntime`. OpenCV headless is used for image warping.

  ### Offline model usage (no downloads)

  Place the InsightFace model zoo locally and point the API to it so it won't download at runtime:

  1. Copy the model folder (e.g., `buffalo_l/` containing SCRFD detector files) into `backend-fastapi/models/`.
  2. Ensure structure like: `backend-fastapi/models/buffalo_l/<model files .onnx>`.
  3. Set the environment variable (optional if you kept this path):

  ```powershell
  $env:FG_INSIGHTFACE_ROOT = "models"
  ```

  On startup, the API will load SCRFD from this folder and skip downloads. If you store models elsewhere, set `FG_INSIGHTFACE_ROOT` to that absolute or relative path.

## Quick start (development)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run the API (mock embeddings by default):

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open docs: http://localhost:8000/docs

## Configuration

Environment variables (prefixed with `FG_`, read from `.env` if present):

- `FG_EMBEDDING_BACKEND` – `mock` (default) or `onnx_arcface`
- `FG_ARC_FACE_ONNX_PATH` – Path to ArcFace ONNX model (when using `onnx_arcface`)
- `FG_ASSUME_ALIGNED` – `true` if uploaded images are already face-cropped/aligned (default)
- `FG_VECTOR_STORE` – `local` (default) or `pinecone`
- `FG_LOCAL_STORE_PATH` – Path for local JSONL store (default `data/embeddings.jsonl`)
- `FG_PINECONE_API_KEY`, `FG_PINECONE_INDEX`, `FG_PINECONE_NAMESPACE` – Pinecone (optional)

## Notes

- The `mock` backend returns deterministic 512-d vectors derived from the image bytes; it requires only NumPy.
- The ONNX ArcFace backend is scaffolded and assumes face-cropped images for now. Enable by installing optional deps:

```powershell
pip install -r extras-ml.txt
$env:FG_EMBEDDING_BACKEND = "onnx_arcface"
$env:FG_ARC_FACE_ONNX_PATH = "C:\\models\\arcface.onnx"  # adjust path
```

- Local JSONL store is used by default. Each line contains: `{id, person_id, vector, metadata}`.
