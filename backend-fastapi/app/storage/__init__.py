from __future__ import annotations
from typing import Optional

from ..config.settings import settings
from .local_store import LocalJsonlStore

try:
    from .pinecone_store import PineconeStore  # type: ignore
except Exception:
    PineconeStore = None  # type: ignore


def get_store():
    store_name = settings.VECTOR_STORE.lower()
    if store_name == "pinecone" and PineconeStore is not None and settings.PINECONE_API_KEY and settings.PINECONE_INDEX:
        return PineconeStore(
            api_key=settings.PINECONE_API_KEY,
            index_name=settings.PINECONE_INDEX,
            namespace=settings.PINECONE_NAMESPACE,
        )
    # default fallback to local jsonl store
    return LocalJsonlStore(settings.LOCAL_STORE_PATH)
