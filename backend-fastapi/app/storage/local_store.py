from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from uuid import uuid4


@dataclass
class StoredItem:
    id: str
    person_id: Optional[str]
    vector: List[float]
    metadata: Dict[str, Any]


class LocalJsonlStore:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            # touch file
            with open(self.path, "a", encoding="utf-8"):
                pass

    def upsert_many(
        self,
        vectors: List[List[float]],
        person_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[StoredItem]:
        items: List[StoredItem] = []
        md = metadata or {}
        with open(self.path, "a", encoding="utf-8") as f:
            for v in vectors:
                _id = str(uuid4())
                item = {
                    "id": _id,
                    "person_id": person_id,
                    "vector": v,
                    "metadata": md,
                }
                f.write(json.dumps(item) + "\n")
                items.append(StoredItem(id=_id, person_id=person_id, vector=v, metadata=md))
        return items
