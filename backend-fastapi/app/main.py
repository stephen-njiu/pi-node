from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import numpy as np

from .config.settings import settings
from .services.embedding import EmbeddingService
from .storage import get_store
from .utils.image import load_image_to_rgb
from .utils.augment import generate_variants
from .utils.face import detect_and_align

app = FastAPI(title="Gate Backend API", version="0.1.0")


class EmbeddingItem(BaseModel):
    id: Optional[str] = None
    vector: List[float] = Field(..., description="L2-normalized embedding vector")


class EmbeddingResponse(BaseModel):
    person_id: Optional[str] = None
    count: int
    items: List[EmbeddingItem]
    stored: bool = False
    store: str


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "backend": settings.EMBEDDING_BACKEND, "store": settings.VECTOR_STORE}


@app.post(f"{settings.API_PREFIX}/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    files: List[UploadFile] = File(..., description="Up to 5 images"),
    person_id: Optional[str] = None,
    augment: bool = Query(default=True, description="Generate augmented variants per image (default: true)"),
    aug_per_image: int = Query(default=3, ge=0, le=5, description="Max augmented variants per image (<=5, default: 3)"),
    svc: EmbeddingService = Depends(get_embedding_service),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum of 5 images allowed")

    # Read all bytes, detect+align face, then optionally create augmented variants
    images_bytes: List[bytes] = []
    for f in files:
        b = await f.read()
        try:
            img = load_image_to_rgb(b)
            aligned = detect_and_align(img) if not settings.ASSUME_ALIGNED else img
            if aligned is None:
                # if detection failed, fallback to original
                aligned = img
            # encode aligned crop to bytes
            import io
            buf = io.BytesIO()
            aligned.save(buf, format="JPEG", quality=95)
            images_bytes.append(buf.getvalue())
        except Exception:
            images_bytes.append(b)

        if augment and aug_per_image > 0:
            try:
                # Generate variants from the aligned crop
                img_for_aug = aligned if 'aligned' in locals() and aligned is not None else load_image_to_rgb(b)
                variants = generate_variants(img_for_aug, max_variants=aug_per_image, allow_flip=False)
                for name, aug_img in variants:
                    # encode to jpeg bytes
                    import io
                    buf = io.BytesIO()
                    aug_img.save(buf, format="JPEG", quality=90)
                    images_bytes.append(buf.getvalue())
            except Exception:
                # if augmentation fails, continue with original
                pass

    # Compute embeddings
    embeddings = svc.embed_many(images_bytes)
    vectors = [emb.vector.astype(float).tolist() for emb in embeddings]

    # Store vectors
    stored_ids: List[str] = []
    stored = False
    store_name = settings.VECTOR_STORE
    try:
        store = get_store()
        stored_items = store.upsert_many(vectors=vectors, person_id=person_id, metadata={"source": "upload"})
        stored_ids = [it.id for it in stored_items]
        stored = True
    except Exception as e:
        # Fallback: don't crash; return embeddings without storing
        stored = False

    items = [EmbeddingItem(id=(stored_ids[i] if i < len(stored_ids) else None), vector=vectors[i]) for i in range(len(vectors))]
    return EmbeddingResponse(person_id=person_id, count=len(items), items=items, stored=stored, store=store_name)
