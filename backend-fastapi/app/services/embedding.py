"""
Face Embedding Service using InsightFace buffalo_l stack.

Pipeline (EXACT):
    det_10g.onnx (detection)
           ↓
    5 landmarks
           ↓  (similarity transform to ArcFace template)
    aligned 112×112 RGB face
           ↓
    w600k_r50.onnx (recognition)
           ↓
    512-D embedding (L2 normalized)

Preprocessing for w600k_r50.onnx:
    - RGB format
    - 112 × 112 pixels
    - (pixel - 127.5) / 128.0  → range [-1, 1]
    - CHW format (channels first)
    - Batch dimension added

Output:
    - 512-dimensional float32 vector
    - L2-normalized (unit length)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import os

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from ..config.settings import settings


# ==============================================================================
# ArcFace Reference Template (5 landmarks for 112x112 aligned face)
# ==============================================================================
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041],  # right mouth corner
], dtype=np.float32)

ALIGNED_SIZE = (112, 112)


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class Embedding:
    """A face embedding vector."""
    vector: np.ndarray  # 512-D, L2-normalized
    model: str


@dataclass
class DetectedFace:
    """A detected face with bounding box and landmarks."""
    bbox: np.ndarray      # [x1, y1, x2, y2]
    score: float          # detection confidence
    landmarks: np.ndarray  # 5x2 array of landmark coordinates


# ==============================================================================
# Umeyama Similarity Transform
# ==============================================================================
def umeyama_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transform (rotation, scale, translation) from src to dst.
    
    Args:
        src: Source points, shape (N, 2)
        dst: Destination points, shape (N, 2)
    
    Returns:
        2x3 affine transformation matrix
    """
    num = src.shape[0]
    
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    # Covariance matrix
    A = (dst_demean.T @ src_demean) / num
    
    # SVD
    U, S, Vt = np.linalg.svd(A)
    
    # Handle reflection
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0 if d >= 0 else -1.0])
    
    # Rotation
    R = U @ D @ Vt
    
    # Scale
    src_var = (src_demean ** 2).sum() / num
    scale = np.sum(S * np.diag(D)) / src_var
    
    # Translation
    t = dst_mean - scale * (R @ src_mean)
    
    # Build 2x3 matrix
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = t
    
    return M


# ==============================================================================
# Face Alignment
# ==============================================================================
def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: Tuple[int, int] = ALIGNED_SIZE,
    template: np.ndarray = ARCFACE_TEMPLATE,
) -> np.ndarray:
    """
    Align a face using 5-point landmarks to the ArcFace template.
    
    Args:
        image: Input image (H, W, 3), RGB or BGR
        landmarks: 5x2 array of landmark coordinates
        target_size: Output size (width, height)
        template: Target landmark positions
    
    Returns:
        Aligned face image (112, 112, 3)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for face alignment")
    
    M = umeyama_transform(landmarks, template)
    aligned = cv2.warpAffine(image, M, target_size, flags=cv2.INTER_LINEAR)
    return aligned


# ==============================================================================
# Buffalo_L Face Pipeline
# ==============================================================================
class BuffaloLPipeline:
    """
    InsightFace buffalo_l pipeline using explicit ONNX models.
    
    Models used:
        - det_10g.onnx: Face detection (outputs bboxes + 5 landmarks)
        - w600k_r50.onnx: Face recognition (outputs 512-D embedding)
    """
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        det_size: Tuple[int, int] = (640, 640),
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is required")
        if cv2 is None:
            raise RuntimeError("OpenCV is required")
        
        # Verify model files exist
        if not os.path.exists(det_model_path):
            raise RuntimeError(f"Detection model not found: {det_model_path}")
        if not os.path.exists(rec_model_path):
            raise RuntimeError(f"Recognition model not found: {rec_model_path}")
        
        # ONNX Runtime providers (prefer GPU)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        self.providers = [p for p in providers if p in available]
        
        self.det_size = det_size
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        
        # Load recognition model
        # Configure SessionOptions to reduce peak memory usage on constrained hosts
        try:
            sess_opts = ort.SessionOptions()
            # Lower thread usage and disable memory arena / pattern to reduce peak RSS
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            # Turn off memory pattern and CPU memory arena (trades some perf for lower peak memory)
            try:
                sess_opts.enable_mem_pattern = False
            except Exception:
                pass
            try:
                sess_opts.enable_cpu_mem_arena = False
            except Exception:
                pass

            self.rec_session = ort.InferenceSession(rec_model_path, sess_options=sess_opts, providers=self.providers)
        except Exception:
            # Fall back to default session creation if SessionOptions not supported
            self.rec_session = ort.InferenceSession(rec_model_path, providers=self.providers)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        
        # Detection via InsightFace FaceAnalysis (handles det_10g parsing)
        self._face_app = None
        
        print(f"BuffaloL Pipeline initialized")
        print(f"   Detection: {det_model_path}")
        print(f"   Recognition: {rec_model_path}")
        print(f"   Providers: {self.providers}")
    
    def _get_detector(self) -> Any:
        """Lazy-load the face detector."""
        if self._face_app is not None:
            return self._face_app
        
        if FaceAnalysis is None:
            raise RuntimeError("insightface is required")
        
        # InsightFace FaceAnalysis expects models at: {root}/models/{name}/
        root = settings.INSIGHTFACE_ROOT or "models"
        
        # Don't restrict allowed_modules - let InsightFace load what it needs
        # The detection module alone doesn't work properly
        self._face_app = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL_NAME,
            root=root,
            providers=['CPUExecutionProvider'],
        )
        self._face_app.prepare(ctx_id=-1, det_size=self.det_size)
        return self._face_app
    
    def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces in a BGR image.
        
        Args:
            image: BGR image (as loaded by cv2.imread or converted from PIL)
        """
        app = self._get_detector()
        
        # InsightFace expects BGR directly
        results = app.get(image)
        
        faces = []
        for face in results:
            if face.kps is None or len(face.kps) < 5:
                continue
            faces.append(DetectedFace(
                bbox=np.array(face.bbox, dtype=np.float32),
                score=float(face.det_score),
                landmarks=np.array(face.kps[:5], dtype=np.float32),  # kps is the 5-point landmarks
            ))
        
        return faces
    
    def _preprocess_for_recognition(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for w600k_r50.onnx.
        
        EXACT preprocessing:
            - RGB, 112x112
            - (pixel - 127.5) / 128.0 -> [-1, 1]
            - HWC -> CHW
            - Add batch -> (1, 3, 112, 112)
        """
        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))
        
        arr = aligned_face.astype(np.float32)
        arr = (arr - 127.5) / 128.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        
        return arr
    
    def _postprocess_embedding(self, output: np.ndarray) -> np.ndarray:
        """L2-normalize the embedding vector."""
        vec = output.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    
    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extract 512-D embedding from aligned 112x112 RGB face."""
        inp = self._preprocess_for_recognition(aligned_face)
        outputs = self.rec_session.run(None, {self.rec_input_name: inp})
        embedding = self._postprocess_embedding(outputs[0])
        return embedding
    
    def process_image(self, image: np.ndarray) -> List[Tuple[DetectedFace, np.ndarray]]:
        """
        Full pipeline: detect -> align -> embed.
        
        Args:
            image: BGR image (H, W, 3) as loaded by cv2
        
        Returns:
            List of (DetectedFace, embedding) tuples
        """
        results = []
        faces = self.detect_faces(image)
        
        for face in faces:
            # Align face (works on BGR, outputs BGR aligned face)
            aligned_bgr = align_face(image, face.landmarks)
            # Convert to RGB for recognition model
            aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
            embedding = self.get_embedding(aligned_rgb)
            results.append((face, embedding))
        
        return results


# ==============================================================================
# Embedding Service (High-Level API)
# ==============================================================================

def _ensure_buffalo_l_models(models_root: str) -> Tuple[str, str]:
    """
    Ensure buffalo_l models are downloaded and return paths.
    
    Downloads from InsightFace model zoo if not present.
    
    Returns:
        Tuple of (det_model_path, rec_model_path)
    """
    # Expected paths
    model_dir = os.path.join(models_root, "models", "buffalo_l")
    det_path = os.path.join(model_dir, "det_10g.onnx")
    rec_path = os.path.join(model_dir, "w600k_r50.onnx")
    
    # Check if models already exist
    if os.path.exists(det_path) and os.path.exists(rec_path):
        print(f"Models found at {model_dir}")
        return det_path, rec_path
    
    print(f"Models not found at {model_dir}, downloading buffalo_l...")
    
    # Create directory structure
    os.makedirs(model_dir, exist_ok=True)
    
    # Method 1: Use insightface's built-in model download
    try:
        from insightface.utils import storage
        
        print("Downloading buffalo_l using InsightFace...")
        
        # Get the default insightface home directory
        home = os.path.expanduser("~")
        insightface_home = os.path.join(home, ".insightface")
        buffalo_dir = os.path.join(insightface_home, "models", "buffalo_l")
        
        # Download to InsightFace home if not exists there
        if not os.path.exists(os.path.join(buffalo_dir, "det_10g.onnx")):
            # This downloads the full buffalo_l pack
            storage.download_onnx("buffalo_l")
        
        # Copy from InsightFace home to our models directory
        if os.path.exists(buffalo_dir):
            import shutil
            
            for filename in ["det_10g.onnx", "w600k_r50.onnx"]:
                src = os.path.join(buffalo_dir, filename)
                dst = os.path.join(model_dir, filename)
                if os.path.exists(src) and not os.path.exists(dst):
                    print(f"Copying {filename} to {model_dir}")
                    shutil.copy2(src, dst)
            
            if os.path.exists(det_path) and os.path.exists(rec_path):
                print("Models downloaded successfully!")
                return det_path, rec_path
                
    except Exception as e:
        print(f"InsightFace download method 1 failed: {e}")
    
    # Method 2: Direct download from correct URLs
    try:
        import urllib.request
        
        print("Attempting direct download from InsightFace releases...")
        print("The latest cors updated")
        
        # Correct URLs for individual model files from buffalo_l release
        # The buffalo_l pack is at: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
        pack_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        
        zip_path = os.path.join(model_dir, "buffalo_l.zip")
        
        print(f"Downloading buffalo_l.zip from {pack_url}...")
        urllib.request.urlretrieve(pack_url, zip_path)
        
        # Extract the zip - extract files one by one to handle partial failures
        import zipfile
        print("Extracting buffalo_l.zip...")
        
        needed_files = ["det_10g.onnx", "w600k_r50.onnx"]
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for filename in needed_files:
                try:
                    if filename in zip_ref.namelist():
                        print(f"Extracting {filename}...")
                        zip_ref.extract(filename, model_dir)
                except Exception as e:
                    print(f"Warning: Failed to extract {filename}: {e}")
        
        # Clean up zip file
        try:
            os.remove(zip_path)
        except:
            pass
        
        if os.path.exists(det_path) and os.path.exists(rec_path):
            print("Models downloaded and extracted successfully!")
            return det_path, rec_path
            
    except Exception as e:
        print(f"Direct download failed: {e}")
    
    # Method 3: Check InsightFace default location and copy
    try:
        home = os.path.expanduser("~")
        default_locations = [
            os.path.join(home, ".insightface", "models", "buffalo_l"),
            os.path.join(home, ".insightface", "buffalo_l"),
            "/root/.insightface/models/buffalo_l",  # Docker
        ]
        
        import shutil
        
        for loc in default_locations:
            src_det = os.path.join(loc, "det_10g.onnx")
            src_rec = os.path.join(loc, "w600k_r50.onnx")
            
            if os.path.exists(src_det) and os.path.exists(src_rec):
                print(f"Found models at {loc}, copying...")
                shutil.copy2(src_det, det_path)
                shutil.copy2(src_rec, rec_path)
                return det_path, rec_path
                
    except Exception as e:
        print(f"Copy from default location failed: {e}")
    
    # If we get here, list what files we have
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"Files in {model_dir}: {files}")
    
    raise RuntimeError(
        f"Could not download buffalo_l models. "
        f"Please manually download buffalo_l.zip from "
        f"https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip "
        f"and extract det_10g.onnx and w600k_r50.onnx to {model_dir}"
    )


class EmbeddingService:
    """
    High-level embedding service for the FastAPI backend.
    
    Pipeline:
        det_10g -> landmarks -> alignment -> w600k_r50 -> L2-normalized 512-D
    
    Models are automatically downloaded if not present.
    """
    
    def __init__(self):
        models_root = settings.INSIGHTFACE_ROOT or "models"
        
        # Ensure models are downloaded (auto-download if missing)
        det_path, rec_path = _ensure_buffalo_l_models(models_root)
        
        print(f"Initializing EmbeddingService with:")
        print(f"  Detection model: {det_path}")
        print(f"  Recognition model: {rec_path}")
        
        self.pipeline = BuffaloLPipeline(
            det_model_path=det_path,
            rec_model_path=rec_path,
        )
    
    def embed_many(self, images_bytes: List[bytes]) -> List[Embedding]:
        """
        Process multiple images and return one embedding per image.
        Takes the largest face from each image.
        """
        embeddings = []
        
        for img_bytes in images_bytes:
            # Load image as BGR (what InsightFace expects)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if bgr_image is None:
                print(f"Failed to decode image, skipping")
                continue
            
            results = self.pipeline.process_image(bgr_image)
            
            if not results:
                print(f"No face detected in image, skipping")
                continue
            
            # Pick largest face
            best_face, best_emb = max(
                results,
                key=lambda x: (x[0].bbox[2] - x[0].bbox[0]) * (x[0].bbox[3] - x[0].bbox[1])
            )
            
            embeddings.append(Embedding(vector=best_emb, model="buffalo_l/w600k_r50"))
        
        return embeddings