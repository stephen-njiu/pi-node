from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class StoredItem:
    id: str
    person_id: Optional[str]
    vector: List[float]
    metadata: Dict[str, Any]


class PineconeStore:
    def __init__(self, api_key: str, index_name: str, namespace: str = "default"):
        self.api_key = api_key
        self.index_name = index_name
        self.namespace = namespace
        # Lazy import to avoid hard dep
        try:
            from pinecone import Pinecone  # type: ignore
        except Exception as e:  # pragma: no cover - optional
            raise RuntimeError(
                "pinecone-client not installed. Install optional deps to enable Pinecone."
            ) from e
        self.client = Pinecone(api_key=self.api_key)
        self.index = self.client.Index(index_name)

    def upsert_many(
        self,
        vectors: List[List[float]],
        person_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[StoredItem]:
        md = metadata or {}
        # Generate IDs; in production you might want to use deterministic IDs per person/image
        from uuid import uuid4

        items = []
        to_upsert = []
        for v in vectors:
            _id = str(uuid4())
            to_upsert.append({"id": _id, "values": v, "metadata": {**md, "person_id": person_id}})
            items.append(StoredItem(id=_id, person_id=person_id, vector=v, metadata=md))

        self.index.upsert(vectors=to_upsert, namespace=self.namespace)
        return items
