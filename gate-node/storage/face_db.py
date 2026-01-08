"""
Face Database using hnswlib for fast vector similarity search.
Stores face embeddings locally and syncs with backend.
"""

import json
import os
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

import numpy as np

try:
    import hnswlib
except ImportError:
    hnswlib = None
    print("Warning: hnswlib not installed. Using brute-force search.")


logger = logging.getLogger(__name__)


@dataclass
class FaceRecord:
    """Represents a stored face record."""
    face_id: str
    user_id: str
    name: str
    status: str  # AUTHORIZED, WANTED, UNKNOWN
    embedding: np.ndarray


@dataclass
class MatchResult:
    """Result from a face match query."""
    face_id: str
    user_id: str
    name: str
    status: str
    distance: float
    confidence: float


class FaceDatabase:
    """
    Local face database using hnswlib for fast ANN search.
    Falls back to brute-force numpy if hnswlib unavailable.
    """
    
    def __init__(
        self,
        index_path: str = "data/faces.index",
        metadata_path: str = "data/faces_metadata.json",
        version_path: str = "data/sync_version.txt",
        dimension: int = 512,
        max_elements: int = 10000,
        ef_construction: int = 200,
        m: int = 16
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.version_path = version_path
        self.dim = dimension
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.m = m
        
        self._lock = threading.Lock()
        self._metadata: dict[int, dict] = {}  # idx -> {face_id, user_id, name, status}
        self._face_id_to_idx: dict[str, int] = {}  # face_id -> idx
        self._next_idx = 0
        self._current_version = 0
        
        # hnswlib index or fallback embeddings array
        self._index = None
        self._embeddings: Optional[np.ndarray] = None  # Fallback storage
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        
        self._init_index()
        self._load()
    
    def _init_index(self):
        """Initialize hnswlib index or fallback."""
        if hnswlib:
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            self._index.init_index(
                max_elements=self.max_elements,
                ef_construction=self.ef_construction,
                M=self.m
            )
            self._index.set_ef(50)  # Query-time parameter
        else:
            # Fallback: store embeddings in numpy array
            self._embeddings = np.zeros((0, self.dim), dtype=np.float32)
    
    def _load(self):
        """Load existing index and metadata from disk."""
        with self._lock:
            # Load metadata
            if os.path.exists(self.metadata_path):
                try:
                    with open(self.metadata_path, "r") as f:
                        data = json.load(f)
                    self._metadata = {int(k): v for k, v in data.get("metadata", {}).items()}
                    self._face_id_to_idx = data.get("face_id_to_idx", {})
                    self._next_idx = data.get("next_idx", 0)
                    logger.info(f"Loaded {len(self._metadata)} face records from metadata")
                except Exception as e:
                    logger.error(f"Failed to load metadata: {e}")
            
            # Load index
            if hnswlib and os.path.exists(self.index_path):
                try:
                    self._index.load_index(self.index_path, max_elements=self.max_elements)
                    logger.info(f"Loaded hnswlib index from {self.index_path}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}")
                    self._init_index()
            
            # Load version
            if os.path.exists(self.version_path):
                try:
                    with open(self.version_path, "r") as f:
                        self._current_version = int(f.read().strip())
                    logger.info(f"Current sync version: {self._current_version}")
                except Exception as e:
                    logger.warning(f"Failed to load version: {e}")
    
    def _save(self):
        """Save index and metadata to disk."""
        with self._lock:
            # Save metadata
            try:
                data = {
                    "metadata": self._metadata,
                    "face_id_to_idx": self._face_id_to_idx,
                    "next_idx": self._next_idx
                }
                with open(self.metadata_path, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")
            
            # Save index
            if hnswlib and self._index.get_current_count() > 0:
                try:
                    self._index.save_index(self.index_path)
                except Exception as e:
                    logger.error(f"Failed to save index: {e}")
            
            # Save version
            try:
                with open(self.version_path, "w") as f:
                    f.write(str(self._current_version))
            except Exception as e:
                logger.error(f"Failed to save version: {e}")
    
    def get_version(self) -> int:
        """Get current sync version."""
        return self._current_version
    
    def set_version(self, version: int):
        """Set sync version."""
        self._current_version = version
        self._save()
    
    def add_face(
        self,
        face_id: str,
        user_id: str,
        name: str,
        status: str,
        embedding: np.ndarray
    ) -> bool:
        """
        Add or update a face in the database.
        Returns True if successful.
        """
        with self._lock:
            # Normalize embedding
            embedding = embedding.astype(np.float32).flatten()
            if embedding.shape[0] != self.dim:
                logger.error(f"Invalid embedding dimension: {embedding.shape[0]} != {self.dim}")
                return False
            
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Check if face already exists
            if face_id in self._face_id_to_idx:
                # Update existing
                idx = self._face_id_to_idx[face_id]
                self._metadata[idx] = {
                    "face_id": face_id,
                    "user_id": user_id,
                    "name": name,
                    "status": status
                }
                # Note: hnswlib doesn't support update, would need to rebuild
                # For simplicity, we skip embedding update (status updates are main concern)
                logger.info(f"Updated face {face_id} metadata")
            else:
                # Add new
                idx = self._next_idx
                self._next_idx += 1
                
                self._metadata[idx] = {
                    "face_id": face_id,
                    "user_id": user_id,
                    "name": name,
                    "status": status
                }
                self._face_id_to_idx[face_id] = idx
                
                if hnswlib:
                    self._index.add_items(embedding.reshape(1, -1), np.array([idx]))
                else:
                    # Fallback
                    self._embeddings = np.vstack([self._embeddings, embedding.reshape(1, -1)])
                
                logger.info(f"Added face {face_id} ({name}) with status {status}")
            
            return True
    
    def remove_face(self, face_id: str) -> bool:
        """Remove a face from the database."""
        with self._lock:
            if face_id not in self._face_id_to_idx:
                return False
            
            idx = self._face_id_to_idx[face_id]
            del self._metadata[idx]
            del self._face_id_to_idx[face_id]
            
            # Note: hnswlib mark_deleted would work but we keep it simple
            # Full rebuild on next sync if needed
            
            logger.info(f"Removed face {face_id}")
            return True
    
    def search(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
        k: int = 1
    ) -> List[Tuple[str, float, dict]]:
        """
        Search for matching faces.
        Returns list of matches within threshold.
        
        Args:
            embedding: Query face embedding (512-dim)
            threshold: Max cosine distance (0 = identical, 2 = opposite)
            k: Number of nearest neighbors to return
        
        Returns:
            List of (person_id, distance, metadata) tuples sorted by distance
        """
        with self._lock:
            if not self._metadata:
                return []
            
            # Normalize query
            embedding = embedding.astype(np.float32).flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            results = []
            
            if hnswlib and self._index.get_current_count() > 0:
                try:
                    labels, distances = self._index.knn_query(
                        embedding.reshape(1, -1),
                        k=min(k, self._index.get_current_count())
                    )
                    
                    for idx, dist in zip(labels[0], distances[0]):
                        if idx in self._metadata and dist <= threshold:
                            meta = self._metadata[idx]
                            # Return as (person_id, distance, metadata)
                            results.append((
                                meta["face_id"],
                                float(dist),
                                {
                                    "face_id": meta["face_id"],
                                    "user_id": meta["user_id"],
                                    "full_name": meta["name"],
                                    "status": meta["status"]
                                }
                            ))
                except Exception as e:
                    logger.error(f"Search error: {e}")
            
            elif self._embeddings is not None and len(self._embeddings) > 0:
                # Brute-force fallback
                distances = 1 - np.dot(self._embeddings, embedding)  # Cosine distance
                sorted_idxs = np.argsort(distances)[:k]
                
                for idx in sorted_idxs:
                    dist = distances[idx]
                    if int(idx) in self._metadata and dist <= threshold:
                        meta = self._metadata[int(idx)]
                        results.append((
                            meta["face_id"],
                            float(dist),
                            {
                                "face_id": meta["face_id"],
                                "user_id": meta["user_id"],
                                "full_name": meta["name"],
                                "status": meta["status"]
                            }
                        ))
            
            return results
    
    def count(self) -> int:
        """Return the number of faces in the database."""
        with self._lock:
            return len(self._metadata)
            
            return results
    
    def sync_from_backend(self, faces: list[dict], version: int):
        """
        Full sync from backend.
        Replaces all faces with the provided list.
        
        Args:
            faces: List of face dicts with face_id, user_id, name, status, embedding
            version: New sync version
        """
        logger.info(f"Syncing {len(faces)} faces from backend (version {version})")
        
        # Clear and rebuild
        with self._lock:
            self._metadata.clear()
            self._face_id_to_idx.clear()
            self._next_idx = 0
            self._init_index()
        
        # Add all faces
        for face in faces:
            try:
                embedding = np.array(face["embedding"], dtype=np.float32)
                self.add_face(
                    face_id=face["face_id"],
                    user_id=face["user_id"],
                    name=face["name"],
                    status=face["status"],
                    embedding=embedding
                )
            except Exception as e:
                logger.error(f"Failed to add face {face.get('face_id')}: {e}")
        
        self._current_version = version
        self._save()
        logger.info(f"Sync complete. {len(self._metadata)} faces loaded.")
    
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._lock:
            status_counts = {}
            for meta in self._metadata.values():
                status = meta.get("status", "UNKNOWN")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_faces": len(self._metadata),
                "version": self._current_version,
                "status_counts": status_counts
            }
