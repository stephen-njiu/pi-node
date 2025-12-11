# Backend (FastAPI)

Minimal and reliable embeddings API used by Savannah Gates’ frontend. Accepts up to 5 images, performs detection/alignment, and returns vectors (mock or ONNX ArcFace).

## Endpoints

- `GET /health` – Health check
- `POST /api/v1/embeddings` – Multipart upload of ≤ 5 images.
  - Form fields used by the frontend: `files` (images), plus metadata like `fullName`, `email`, `organization`, `role`, `notes`, and `wanted`.
  - Optional query params:
    - `augment` (bool): identity‑preserving variants per image (default: true)
    - `aug_per_image` (int ≤ 5): number of variants (default: 3)

## Detection and alignment

- SCRFD via `insightface` to detect faces and 5‑point alignment to 112×112
- Falls back to original image if detection fails
- Set `FG_ASSUME_ALIGNED=true` only if you send pre‑cropped/aligned faces

## Optional ML dependencies

Install extras for detection and ONNX embeddings:

```powershell
pip install -r extras-ml.txt
```

Includes `insightface` (SCRFD) and `onnxruntime`. OpenCV headless handles image warping.

### Offline model usage

Place InsightFace model files locally to avoid runtime downloads:

1. Copy model folder (e.g., `buffalo_l/`) into `backend-fastapi/models/`
2. Ensure: `backend-fastapi/models/buffalo_l/*.onnx`
3. Optionally set:

```powershell
$env:FG_INSIGHTFACE_ROOT = "models"
```

## Quick start (development)

1. Create venv and install deps

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run API (mock embeddings default)

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Docs: http://localhost:8000/docs

## Configuration (env)

- `FG_EMBEDDING_BACKEND`: `mock` (default) | `onnx_arcface`
- `FG_ARC_FACE_ONNX_PATH`: path to ArcFace ONNX when using `onnx_arcface`
- `FG_ASSUME_ALIGNED`: `true` if inputs are pre‑aligned
- `FG_VECTOR_STORE`: `local` (default) | `pinecone`
- `FG_LOCAL_STORE_PATH`: default `data/embeddings.jsonl`
- Pinecone: `FG_PINECONE_API_KEY`, `FG_PINECONE_INDEX`, `FG_PINECONE_NAMESPACE`

## Notes

- The `mock` backend returns deterministic 512‑d vectors from image bytes; good for local dev.
- ONNX ArcFace requires aligned faces and the model path.
- Local JSONL store is default: `{id, person_id, vector, metadata}` rows.

— Made with 💙 from Silicon Savannah 💙
