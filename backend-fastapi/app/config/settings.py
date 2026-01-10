from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # ===============================
    # Environment
    # ===============================
    ENVIRONMENT: str = "development"  # Set to "production" in Railway

    # ===============================
    # InsightFace buffalo_l Models
    # ===============================
    # Root directory containing buffalo_l folder with:
    #   - det_10g.onnx (face detection)
    #   - w600k_r50.onnx (face recognition)
    INSIGHTFACE_ROOT: Optional[str] = "models"
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"

    # ===============================
    # Vector Store (Pinecone)
    # ===============================
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
    # Postgres (for Pi sync)
    # ===============================
    # Database URL for face embeddings storage (same DB as Next.js/Prisma)
    # FG_DATABASE_URL="postgresql://user:pass@host:5432/db"
    DATABASE_URL: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FG_",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
