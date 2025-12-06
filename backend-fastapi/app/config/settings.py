from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Embedding backend: 'mock' (default) or 'onnx_arcface'
    EMBEDDING_BACKEND: str = "mock"

    # ONNX model path for ArcFace (when EMBEDDING_BACKEND=onnx_arcface)
    ARC_FACE_ONNX_PATH: Optional[str] = None

    # Assume uploaded images are already face-cropped/aligned
    # Default false for raw camera images: enables detection+alignment in pipeline
    ASSUME_ALIGNED: bool = False

    # Storage: 'local' (jsonl) or 'pinecone'
    VECTOR_STORE: str = "local"

    # Local store path
    LOCAL_STORE_PATH: str = "data/embeddings.jsonl"

    # Pinecone config (optional for later)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX: Optional[str] = None
    PINECONE_ENV: Optional[str] = None
    PINECONE_NAMESPACE: str = "default"

    # API
    API_PREFIX: str = "/api/v1"

    # InsightFace (SCRFD) model root for offline use. If provided, models are loaded from here.
    # Place the model zoo folder (e.g., `buffalo_l`) inside this directory.
    INSIGHTFACE_ROOT: Optional[str] = "models"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FG_", case_sensitive=False)


settings = Settings()