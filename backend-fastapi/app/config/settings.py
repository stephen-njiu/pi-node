from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # ===============================
    # Environment
    # ===============================
    ENVIRONMENT: str = "development"  # Set to "production" in Railway

    # ===============================
    # Embeddings
    # ===============================
    EMBEDDING_BACKEND: str = "onnx_arcface"

    ARC_FACE_ONNX_PATH: Optional[str] = "models/arcface.onnx"
    ARC_FACE_ONNX_URL: Optional[str] = None

    ASSUME_ALIGNED: bool = False

    # ===============================
    # Vector Store
    # ===============================
    VECTOR_STORE: str = "pinecone"

    LOCAL_STORE_PATH: str = "data/embeddings.jsonl"

    # 🔑 Pinecone (ALL via Pydantic)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX: Optional[str] = None
    PINECONE_INDEX_HOST: Optional[str] = None
    PINECONE_NAMESPACE: str = "__default__"

    # ===============================
    # API
    # ===============================
    API_PREFIX: str = "/api/v1"
    # Comma-separated CORS origins (with FG_ prefix), e.g.,
    # FG_ALLOW_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,https://myapp.com"
    ALLOW_ORIGINS: Optional[str] = None

    # ===============================
    # InsightFace
    # ===============================
    INSIGHTFACE_ROOT: Optional[str] = "models"
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    INSIGHTFACE_ALLOW_AUTO_DOWNLOAD: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FG_",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
