from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4
import os

from .config.settings import settings
from .services.embedding import EmbeddingService
from .utils.image import load_image_to_rgb
from .utils.augment import generate_variants
from .utils.face import detect_and_align

# =========================
# Pinecone
# =========================
try:
    from pinecone.grpc import PineconeGRPC as Pinecone
except Exception:
    from pinecone import Pinecone  # fallback

# =========================
# App
# =========================
# Disable docs in production (set ENVIRONMENT=production in Railway)
is_production = os.getenv("ENVIRONMENT", "development") == "production"

app = FastAPI(
    title="Gate Backend API", 
    version="0.1.0",
    docs_url=None,  # Disable default docs, we'll create custom one
    redoc_url="/redoc" if not is_production else None,
)

# =========================
# Startup
# =========================
@app.on_event("startup")
async def startup_event():
    # Embedding service
    app.state.embedding_service = EmbeddingService()

    # Validate settings
    if not settings.PINECONE_API_KEY or not settings.PINECONE_INDEX_HOST:
        raise RuntimeError("❌ Pinecone settings missing")

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host=settings.PINECONE_INDEX_HOST)

    app.state.pinecone_index = index
    app.state.pinecone_namespace = settings.PINECONE_NAMESPACE

# =========================
# Models
# =========================
class EmbeddingItem(BaseModel):
    id: Optional[str]
    vector: List[float]

class EmbeddingResponse(BaseModel):
    person_id: Optional[str]
    count: int
    items: List[EmbeddingItem]
    stored: bool
    store: str

def get_embedding_service() -> EmbeddingService:
    return app.state.embedding_service  # type: ignore

# =========================
# CORS
# =========================
origins_cfg = settings.ALLOW_ORIGINS or "http://localhost:3000,http://127.0.0.1:3000"
allow_origins = [o.strip() for o in origins_cfg.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Custom Docs (Read-Only)
# =========================
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI with "Try it out" disabled.
    Only shows API documentation without interactive testing.
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Documentation",
        swagger_ui_parameters={
            "supportedSubmitMethods": [],  # Disables "Try it out" buttons
        }
    )

# =========================
# Routes
# =========================
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": settings.EMBEDDING_BACKEND,
        "store": "pinecone",
    }

@app.post(f"{settings.API_PREFIX}/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    files: List[UploadFile] = File(...),
    person_id: Optional[str] = None,
    augment: bool = Query(default=True),
    aug_per_image: int = Query(default=3, ge=0, le=5),
    fullName: str = Form(...),
    email: str = Form(...),
    organization: str = Form(...),
    role: str = Form(...),
    wanted: bool = Form(...),
    notes: Optional[str] = Form(None),
    svc: EmbeddingService = Depends(get_embedding_service),
):
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > 5:
        raise HTTPException(400, "Maximum of 5 images allowed")

    images_bytes: List[bytes] = []

    for f in files:
        raw = await f.read()
        img = load_image_to_rgb(raw)
        aligned = detect_and_align(img) if not settings.ASSUME_ALIGNED else img
        aligned = aligned or img

        import io
        buf = io.BytesIO()
        aligned.save(buf, format="JPEG", quality=95)
        images_bytes.append(buf.getvalue())

        if augment:
            for _, aug in generate_variants(aligned, aug_per_image, allow_flip=False):
                buf = io.BytesIO()
                aug.save(buf, format="JPEG", quality=90)
                images_bytes.append(buf.getvalue())

    embeddings = svc.embed_many(images_bytes)
    vectors = [e.vector.tolist() for e in embeddings]

    index = app.state.pinecone_index
    namespace = app.state.pinecone_namespace

    meta = {
        "person_id": person_id,
        "fullName": fullName,
        "email": email,
        "organization": organization,
        "role": role,
        "notes": notes,
        "wanted": wanted,
    }
    meta = {k: v for k, v in meta.items() if v is not None}

    records, ids = [], []
    for v in vectors:
        _id = str(uuid4())
        ids.append(_id)
        records.append({"id": _id, "values": v, "metadata": meta})

    try:
        index.upsert(vectors=records, namespace=namespace)
    except Exception as e:
        raise HTTPException(502, f"Pinecone upsert failed: {e}")

    return EmbeddingResponse(
        person_id=person_id,
        count=len(ids),
        items=[EmbeddingItem(id=ids[i], vector=vectors[i]) for i in range(len(ids))],
        stored=True,
        store="pinecone",
    )
