from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import hashlib
import numpy as np
import os

try:
    import onnxruntime as ort  # type: ignore
except Exception:
    ort = None  # type: ignore

from ..config.settings import settings
from ..utils.image import load_image_to_rgb, resize_for_arcface


@dataclass
class Embedding:
    vector: np.ndarray
    model: str


class BaseEmbeddingBackend:
    def embed_many(self, images_bytes: List[bytes]) -> List[Embedding]:
        raise NotImplementedError


class MockEmbeddingBackend(BaseEmbeddingBackend):
    """Deterministic mock embeddings derived from SHA256 of bytes."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bytes_to_vec(self, b: bytes) -> np.ndarray:
        h = hashlib.sha256(b).digest()
        repeats = (self.dim * 4 + len(h) - 1) // len(h)
        buf = (h * repeats)[: self.dim]
        arr = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        norm = np.linalg.norm(arr) or 1.0
        return arr / norm

    def embed_many(self, images_bytes: List[bytes]) -> List[Embedding]:
        return [Embedding(self._bytes_to_vec(b), model="mock-sha256") for b in images_bytes]


class OnnxArcFaceBackend(BaseEmbeddingBackend):
    def __init__(self, model_path: str, assume_aligned: bool = True):
        if ort is None:
            raise RuntimeError("onnxruntime not available. Install optional ML dependencies.")

        # Ensure model exists; download if URL is provided
        if not os.path.exists(model_path):
            url = getattr(settings, "ARC_FACE_ONNX_URL", None)
            if url:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                try:
                    import urllib.request
                    urllib.request.urlretrieve(url, model_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to download ArcFace ONNX from {url}: {e}")
            else:
                raise RuntimeError(
                    f"ArcFace ONNX model not found at '{model_path}'. "
                    "Set ARC_FACE_ONNX_PATH to a valid file or provide FG_ARC_FACE_ONNX_URL to auto-download."
                )

        # Prefer GPU if available
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available_providers = ort.get_available_providers() if ort else []
        used_providers = [p for p in providers if p in available_providers]
        if not used_providers:
            raise RuntimeError(f"No available ONNX Runtime providers found. Tried: {providers}")

        self.session = ort.InferenceSession(model_path, providers=used_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.assume_aligned = assume_aligned

    def _preprocess(self, b: bytes) -> np.ndarray:
        img = load_image_to_rgb(b)
        arr = resize_for_arcface(img)
        arr = np.expand_dims(arr, axis=0)  # Add batch dim
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
