from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import hashlib
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional
    ort = None  # type: ignore

from ..config.settings import settings
from ..utils.image import load_image_to_rgb, resize_for_arcface


@dataclass
class Embedding:
    vector: np.ndarray  # shape (D,)
    model: str


class BaseEmbeddingBackend:
    def embed_many(self, images_bytes: List[bytes]) -> List[Embedding]:
        raise NotImplementedError


class MockEmbeddingBackend(BaseEmbeddingBackend):
    """Deterministic mock embeddings derived from SHA256 of bytes.
    Useful for development without ML dependencies.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bytes_to_vec(self, b: bytes) -> np.ndarray:
        # Expand SHA256 digest deterministically to dim elements
        h = hashlib.sha256(b).digest()
        # Repeat digest to exceed desired dim
        repeats = (self.dim * 4 + len(h) - 1) // len(h)
        buf = (h * repeats)[: self.dim]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # Normalize to unit vector
        norm = np.linalg.norm(arr) or 1.0
        return arr / norm

    def embed_many(self, images_bytes: List[bytes]) -> List[Embedding]:
        return [Embedding(self._bytes_to_vec(b), model="mock-sha256") for b in images_bytes]


class OnnxArcFaceBackend(BaseEmbeddingBackend):
    def __init__(self, model_path: str, assume_aligned: bool = True):
        if ort is None:
            raise RuntimeError("onnxruntime not available. Install optional ML deps.")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # noqa: S603
        self.input_name = self.session.get_inputs()[0].name
        self.assume_aligned = assume_aligned

    def _preprocess(self, b: bytes) -> np.ndarray:
        from PIL import Image  # local import to avoid hard dep if unused
        img = load_image_to_rgb(b)
        # For MVP we assume the image is already a face crop/aligned
        arr = resize_for_arcface(img)
        # Add batch dim
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _postprocess(self, out: np.ndarray) -> np.ndarray:
        vec = out.flatten().astype(np.float32)
        norm = np.linalg.norm(vec) or 1.0
        return vec / norm

    def embed_many(self, images_bytes: List[bytes]) -> List[Embedding]:
        embs: List[Embedding] = []
        for b in images_bytes:
            inp = self._preprocess(b)
            outputs = self.session.run(None, {self.input_name: inp})  # type: ignore[arg-type]
            vec = self._postprocess(outputs[0])
            embs.append(Embedding(vec, model="arcface-onnx"))
        return embs


class EmbeddingService:
    def __init__(self):
        backend_name = settings.EMBEDDING_BACKEND.lower()
        if backend_name == "onnx_arcface" and settings.ARC_FACE_ONNX_PATH:
            self.backend: BaseEmbeddingBackend = OnnxArcFaceBackend(
                model_path=settings.ARC_FACE_ONNX_PATH,
                assume_aligned=settings.ASSUME_ALIGNED,
            )
        else:
            self.backend = MockEmbeddingBackend()

    def embed_many(self, images_bytes: List[bytes]) -> List[Embedding]:
        return self.backend.embed_many(images_bytes)
