"""
ArcFace Face Recognition using ONNX Runtime.
Extracts 512-dimensional face embeddings for recognition.
"""

import numpy as np
import cv2
import logging
from typing import Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None


logger = logging.getLogger(__name__)


class ArcFaceRecognizer:
    """
    ArcFace face recognition model.
    Extracts 512-dimensional embedding from aligned 112x112 face images.
    """
    
    def __init__(
        self,
        model_path: str = "models/w600k_r50.onnx",
        input_size: tuple = (112, 112)
    ):
        self.model_path = model_path
        self.input_size = input_size
        
        self._session = None
        self._input_name = None
        self._output_name = None
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model."""
        if ort is None:
            logger.error("ONNX Runtime not available")
            return
        
        try:
            providers = ['CPUExecutionProvider']
            
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self._session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            
            # Get embedding dimension
            output_shape = self._session.get_outputs()[0].shape
            self.embedding_dim = output_shape[-1] if len(output_shape) > 1 else 512
            
            logger.info(f"Loaded ArcFace model from {self.model_path}")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {e}")
            self._session = None
    
    def _preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for model input.
        
        MUST MATCH backend-fastapi preprocessing exactly:
        - Normalize: (arr - 127.5) / 128.0
        - Format: RGB, CHW
        
        Args:
            face: Aligned face image (112x112 BGR)
        
        Returns:
            Preprocessed blob (1, 3, 112, 112)
        """
        # Resize if needed
        if face.shape[:2] != self.input_size:
            face = cv2.resize(face, self.input_size)
        
        # Convert BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize to [-1, 1] - MUST match backend: (arr - 127.5) / 128.0
        face = (face.astype(np.float32) - 127.5) / 128.0
        
        # HWC to CHW
        face = face.transpose(2, 0, 1)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def get_embedding(self, face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from aligned face image.
        
        Args:
            face: Aligned face image (112x112 BGR)
        
        Returns:
            512-dimensional normalized embedding, or None on error
        """
        if self._session is None:
            logger.warning("Model not loaded")
            return None
        
        try:
            # Preprocess
            blob = self._preprocess(face)
            
            # Run inference
            embedding = self._session.run(
                [self._output_name],
                {self._input_name: blob}
            )[0]
            
            # Flatten and normalize
            embedding = embedding.flatten()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None
    
    def get_embeddings_batch(self, faces: list[np.ndarray]) -> list[Optional[np.ndarray]]:
        """
        Extract embeddings from multiple faces (batch processing).
        
        Args:
            faces: List of aligned face images
        
        Returns:
            List of embeddings (None for failed extractions)
        """
        if self._session is None or not faces:
            return [None] * len(faces)
        
        try:
            # Preprocess all faces
            blobs = []
            for face in faces:
                blob = self._preprocess(face)
                blobs.append(blob[0])  # Remove batch dim for stacking
            
            batch = np.stack(blobs, axis=0)
            
            # Run batch inference
            embeddings = self._session.run(
                [self._output_name],
                {self._input_name: batch}
            )[0]
            
            # Normalize each embedding
            results = []
            for emb in embeddings:
                emb = emb.flatten()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                results.append(emb)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return [None] * len(faces)
    
    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding (normalized)
            emb2: Second embedding (normalized)
        
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        return float(np.dot(emb1, emb2))
    
    @staticmethod
    def compute_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine distance between two embeddings.
        
        Args:
            emb1: First embedding (normalized)
            emb2: Second embedding (normalized)
        
        Returns:
            Distance score (0-2, lower = more similar)
        """
        return float(1 - np.dot(emb1, emb2))
