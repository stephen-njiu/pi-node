from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
import os
import traceback
import threading

from .config.settings import settings
from .services.embedding import EmbeddingService
from .storage.postgres import PostgresStore

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
# Disable docs in production (set FG_ENVIRONMENT=production in Railway)
is_production = settings.ENVIRONMENT == "production"

app = FastAPI(
    title="Gate Backend API", 
    version="0.1.0",
    docs_url=None,  # Disable default docs, we'll create custom one
    redoc_url="/redoc" if not is_production else None,
)

# =========================
# CORS (MUST be added BEFORE any routes)
# =========================
# Load allowed origins from settings (env var FG_ALLOW_ORIGINS expected)
# Default to the public frontend origin if not provided
origins_cfg = settings.ALLOW_ORIGINS or "https://savannah-gates.vercel.app"

# Parse comma-separated list and explicitly disallow wildcard entries
if isinstance(origins_cfg, str):
    parsed = [o.strip() for o in origins_cfg.split(",") if o.strip()]
else:
    parsed = list(origins_cfg)

# Remove any wildcard entries - we do not allow '*'
parsed = [o for o in parsed if o != "*"]

# If nothing left after parsing, fall back to the public frontend
if not parsed:
    parsed = ["https://savannah-gates.vercel.app"]

allow_origins = parsed

print(f"🌐 CORS Origins (from FG_ALLOW_ORIGINS): {allow_origins}")

# Credentials allowed only when explicit origins are set (not wildcard)
allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print("⏳ Loading embedding models (buffalo_l)... this may take a few seconds")
    app.state.embedding_service = EmbeddingService()
    print("✅ Model buffalo_l loaded successfully")
    # Keep a lock available (used by get_embedding_service as a safeguard)
    if not hasattr(app.state, "_embedding_lock"):
        app.state._embedding_lock = threading.Lock()

    # Validate settings
    if not settings.PINECONE_API_KEY or not settings.PINECONE_INDEX_HOST:
        raise RuntimeError("❌ Pinecone settings missing")

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(host=settings.PINECONE_INDEX_HOST)

    app.state.pinecone_index = index
    app.state.pinecone_namespace = settings.PINECONE_NAMESPACE

    # Postgres for Pi sync
    if settings.DATABASE_URL:
        pg_store = PostgresStore(settings.DATABASE_URL)
        await pg_store.connect()
        app.state.postgres_store = pg_store
        print("✅ Postgres connected for Pi sync")
    else:
        app.state.postgres_store = None
        print("⚠️ DATABASE_URL not set - Pi sync disabled")


@app.on_event("shutdown")
async def shutdown_event():
    # Close Postgres pool
    if hasattr(app.state, 'postgres_store') and app.state.postgres_store:
        await app.state.postgres_store.disconnect()
        print("✅ Postgres disconnected")

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
    # Lazy, thread-safe singleton creation for EmbeddingService.
    # This avoids loading large ONNX models during process startup which
    # can cause out-of-memory crashes on constrained hosts.
    svc = getattr(app.state, "embedding_service", None)
    if svc is None:
        # Acquire lock stored on app.state during startup
        lock = getattr(app.state, "_embedding_lock", None)
        if lock is None:
            # Fallback: create a lock if missing
            lock = threading.Lock()
            app.state._embedding_lock = lock
        with lock:
            svc = getattr(app.state, "embedding_service", None)
            if svc is None:
                svc = EmbeddingService()
                app.state.embedding_service = svc
    return svc

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
    pg_connected = hasattr(app.state, 'postgres_store') and app.state.postgres_store is not None
    return {
        "status": "ok",
        # "backend": settings.EMBEDDING_BACKEND,
        "store": "pinecone",
        "pi_sync": "postgres" if pg_connected else "disabled",
    }


@app.post(f"{settings.API_PREFIX}/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    files: List[UploadFile] = File(...),
    person_id: Optional[str] = None,
    fullName: str = Form(...),
    email: str = Form(...),
    organization: str = Form(...),
    role: str = Form(...),
    wanted: bool = Form(...),
    notes: Optional[str] = Form(None),
    svc: EmbeddingService = Depends(get_embedding_service),
):
    """
    Create face embeddings from uploaded images.
    
    Pipeline (buffalo_l):
        1. det_10g.onnx → detect faces + 5-point landmarks
        2. Similarity transform alignment → 112×112 RGB
        3. w600k_r50.onnx → 512-D embedding (L2 normalized)
    
    Each image produces ONE embedding (from the largest detected face).
    Embeddings are stored in both Pinecone and PostgreSQL.
    """
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > 5:
        raise HTTPException(400, "Maximum of 5 images allowed")

    # Read all image bytes
    images_bytes: List[bytes] = []
    for f in files:
        raw = await f.read()
        images_bytes.append(raw)

    # Extract embeddings using the buffalo_l pipeline
    # (detection → alignment → recognition → L2 normalize)
    embeddings = svc.embed_many(images_bytes)
    
    if not embeddings:
        raise HTTPException(400, "No faces detected in any of the uploaded images")
    
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

    # =========================
    # Dual-write to Postgres (for Pi sync)
    # =========================
    pg_stored = False
    pg_store: Optional[PostgresStore] = app.state.postgres_store
    
    if pg_store:
        try:
            # Determine face status
            face_status = "WANTED" if wanted else "AUTHORIZED"
            
            # Store all embeddings to Postgres
            faces_to_store = [
                {
                    "id": ids[i],
                    "person_id": person_id,
                    "org_id": organization,
                    "full_name": fullName,
                    "email": email,
                    "role": role,
                    "status": face_status,
                    "embedding": vectors[i],
                    "notes": notes,
                }
                for i in range(len(ids))
            ]
            
            await pg_store.upsert_faces_batch(faces_to_store)
            pg_stored = True
        except Exception as e:
            # Log but don't fail - Pinecone is primary
            print(f"⚠️ Postgres upsert failed (non-fatal): {e}")

    return EmbeddingResponse(
        person_id=person_id,
        count=len(ids),
        items=[EmbeddingItem(id=ids[i], vector=vectors[i]) for i in range(len(ids))],
        stored=True,
        store="pinecone+postgres" if pg_stored else "pinecone",
    )


# =========================
# Pi Sync Endpoint
# =========================
class FaceSyncItem(BaseModel):
    id: str
    person_id: Optional[str]
    full_name: str
    email: Optional[str]
    role: Optional[str]
    status: str
    embedding: List[float]
    image_url: Optional[str]
    notes: Optional[str]


class FaceSyncResponse(BaseModel):
    version: str
    upserts: List[FaceSyncItem]
    deletes: List[str]
    count: int


@app.get(f"{settings.API_PREFIX}/faces/sync", response_model=FaceSyncResponse)
async def sync_faces(
    org_id: str = Query(..., description="Organization ID to sync faces for"),
    since: Optional[str] = Query(None, description="ISO timestamp for delta sync (e.g., 2026-01-08T00:00:00Z)"),
):
    """
    Sync endpoint for Pi gate nodes.
    
    - First sync: omit `since` to get all faces for the organization
    - Delta sync: pass `since` (from previous response's `version`) to get only changes
    
    Returns upserts (new/updated faces) and deletes (removed face IDs).
    Pi should save `version` and use it as `since` in next poll.
    """
    pg_store: Optional[PostgresStore] = app.state.postgres_store
    
    if not pg_store:
        raise HTTPException(503, "Pi sync not configured - DATABASE_URL not set")
    
    # Parse since timestamp
    since_dt: Optional[datetime] = None
    if since:
        try:
            # Handle ISO format with or without timezone
            parsed = datetime.fromisoformat(since.replace('Z', '+00:00'))
            # Strip timezone for naive comparison (Postgres timestamp without tz)
            since_dt = parsed.replace(tzinfo=None)
        except ValueError:
            raise HTTPException(400, f"Invalid timestamp format: {since}")
    
    try:
        result = await pg_store.get_faces_for_sync(org_id, since_dt)
    except Exception as e:
        import traceback
        print(f"❌ Database query failed: {e}")
        print(f"   org_id: {org_id}, since: {since}, since_dt: {since_dt}")
        traceback.print_exc()
        raise HTTPException(502, f"Database query failed: {e}")
    
    return FaceSyncResponse(
        version=result["version"],
        upserts=[FaceSyncItem(**face) for face in result["upserts"]],
        deletes=result["deletes"],
        count=len(result["upserts"]),
    )


@app.get(f"{settings.API_PREFIX}/faces/count")
async def face_count(
    org_id: str = Query(..., description="Organization ID"),
):
    """Get count of active faces for an organization."""
    pg_store: Optional[PostgresStore] = app.state.postgres_store
    
    if not pg_store:
        raise HTTPException(503, "Pi sync not configured - DATABASE_URL not set")
    
    try:
        count = await pg_store.get_face_count(org_id)
        return {"org_id": org_id, "count": count}
    except Exception as e:
        raise HTTPException(502, f"Database query failed: {e}")


@app.get(f"{settings.API_PREFIX}/faces/debug")
async def face_debug(
    org_id: Optional[str] = Query(None, description="Filter by organization ID"),
):
    """Debug endpoint to see face records and their timestamps."""
    pg_store: Optional[PostgresStore] = app.state.postgres_store
    
    if not pg_store:
        raise HTTPException(503, "Pi sync not configured - DATABASE_URL not set")
    
    try:
        async with pg_store.pool.acquire() as conn:
            if org_id:
                query = """
                    SELECT id, "orgId", "fullName", status, "createdAt", "updatedAt", "deletedAt"
                    FROM face
                    WHERE "orgId" = $1
                    ORDER BY "updatedAt" DESC
                    LIMIT 50
                """
                rows = await conn.fetch(query, org_id)
            else:
                query = """
                    SELECT id, "orgId", "fullName", status, "createdAt", "updatedAt", "deletedAt"
                    FROM face
                    ORDER BY "updatedAt" DESC
                    LIMIT 50
                """
                rows = await conn.fetch(query)
            
            # Also get distinct org_ids
            org_query = """SELECT DISTINCT "orgId", COUNT(*) as count FROM face GROUP BY "orgId" """
            org_rows = await conn.fetch(org_query)
        
        return {
            "total_shown": len(rows),
            "orgs": [{"org_id": r["orgId"], "count": r["count"]} for r in org_rows],
            "faces": [
                {
                    "id": r["id"][:8] + "...",
                    "org_id": r["orgId"],
                    "full_name": r["fullName"],
                    "status": r["status"],
                    "created_at": r["createdAt"].isoformat() if r["createdAt"] else None,
                    "updated_at": r["updatedAt"].isoformat() if r["updatedAt"] else None,
                    "deleted_at": r["deletedAt"].isoformat() if r["deletedAt"] else None,
                }
                for r in rows
            ],
        }
    except Exception as e:
        raise HTTPException(502, f"Database query failed: {e}")
