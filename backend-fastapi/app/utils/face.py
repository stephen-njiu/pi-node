from __future__ import annotations
from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image
import os
from pathlib import Path

try:
    from insightface.app import FaceAnalysis  # type: ignore
except Exception:
    FaceAnalysis = None  # type: ignore

# Standard ArcFace template landmarks (112x112) - reference points
ARC_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth
        [70.7299, 92.2041],  # right mouth
    ],
    dtype=np.float32,
)

TARGET_SIZE = (112, 112)

# Global singleton to avoid re-initializing per request
DETECTOR: Any = None


def _resolve_root(root: Optional[str]) -> Optional[str]:
    if not root:
        return None
    p = Path(root)
    if not p.is_absolute():
        # Resolve relative to backend-fastapi root
        here = Path(__file__).resolve().parents[2]  # .../backend-fastapi
        p = (here / root).resolve()
    return str(p)


def _get_detector() -> Any:
    from ..config.settings import settings
    global DETECTOR
    if DETECTOR is not None:
        return DETECTOR
    if FaceAnalysis is None:
        raise RuntimeError("insightface not installed. Install extras-ml.txt to enable SCRFD.")
    root = _resolve_root(settings.INSIGHTFACE_ROOT)
    # Initialize FaceAnalysis with SCRFD only to avoid downloading extra modules
    app = FaceAnalysis(name='buffalo_l', root=root, allowed_modules=['detection'])  # includes SCRFD
    app.prepare(ctx_id=0, det_size=(640, 640))
    DETECTOR = app
    return DETECTOR


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    # estimate transform from src (5x2) to dst (5x2)
    # Using Umeyama method
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = src_demean.T @ dst_demean / src.shape[0]
    U, S, Vt = np.linalg.svd(A)
    R = (U @ Vt)
    d = np.sign(np.linalg.det(R))
    S_mat = np.diag([1, d])
    R = U @ S_mat @ Vt
    var_src = (src_demean ** 2).sum() / src.shape[0]
    scale = np.trace(np.diag(S) @ S_mat) / var_src
    t = dst_mean - scale * (R @ src_mean)
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = t
    return M


def detect_and_align(image: Image.Image) -> Optional[Image.Image]:
    """
    Detect the most prominent face and align to 112x112 using 5 landmarks.
    Returns aligned PIL image or None if no face.
    """
    app = _get_detector()
    # insightface expects numpy array (BGR)
    rgb = np.asarray(image)
    import cv2  # type: ignore
    bgr_for_det = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = app.get(bgr_for_det)
    if not faces:
        return None
    # pick face with largest bbox
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    if best.landmark is None or len(best.landmark) < 5:
        return None
    src = np.asarray(best.landmark, dtype=np.float32)
    dst = ARC_TEMPLATE
    M = _similarity_transform(src, dst)
    # warp to target size using OpenCV (headless)
    warped = cv2.warpAffine(bgr_for_det, M, TARGET_SIZE, flags=cv2.INTER_LINEAR)
    aligned_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(aligned_rgb)
