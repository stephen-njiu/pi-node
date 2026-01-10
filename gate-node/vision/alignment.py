"""
Face Alignment utility for ArcFace recognition.
Aligns detected faces to a standard 112x112 template using 5-point landmarks.
"""

import numpy as np
import cv2
from typing import Optional


# Standard ArcFace template landmarks (112x112) - reference points
# These are the canonical positions for a properly aligned face
ARC_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

TARGET_SIZE = (112, 112)


def estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Estimate 2D similarity transform (rotation, scale, translation) from src to dst.
    Uses Umeyama algorithm - MATCHES backend-fastapi/app/services/embedding.py exactly.
    
    Args:
        src: Source landmarks (5x2) - detected face landmarks
        dst: Destination landmarks (5x2) - ArcFace template
    
    Returns:
        2x3 transformation matrix
    """
    num = src.shape[0]
    
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    # Covariance matrix: dst^T @ src (NOT src^T @ dst!)
    A = (dst_demean.T @ src_demean) / num
    
    # SVD
    U, S, Vt = np.linalg.svd(A)
    
    # Handle reflection
    d = np.linalg.det(U @ Vt)
    D = np.diag([1.0, 1.0 if d >= 0 else -1.0])
    
    # Rotation
    R = U @ D @ Vt
    
    # Scale - use element-wise multiply with diagonal, not trace
    src_var = (src_demean ** 2).sum() / num
    scale = np.sum(S * np.diag(D)) / src_var
    
    # Translation
    t = dst_mean - scale * (R @ src_mean)
    
    # Build 2x3 matrix
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:, 2] = t
    return M


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    target_size: tuple = TARGET_SIZE,
) -> Optional[np.ndarray]:
    """
    Align a face image using 5-point landmarks to match ArcFace template.
    
    This is CRITICAL for face recognition - embeddings from unaligned faces
    will not match embeddings from aligned faces stored in the database.
    
    Args:
        image: Input image (BGR format from OpenCV)
        landmarks: 5x2 array of facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        target_size: Output size (default 112x112 for ArcFace)
    
    Returns:
        Aligned face image (112x112 BGR) or None if alignment fails
    """
    if landmarks is None or len(landmarks) < 5:
        return None
    
    # Ensure landmarks are float32 and correct shape
    src = np.array(landmarks[:5], dtype=np.float32).reshape(5, 2)
    dst = ARC_TEMPLATE
    
    # Estimate similarity transform
    M = estimate_similarity_transform(src, dst)
    
    # Warp image to align face
    aligned = cv2.warpAffine(
        image, M, target_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return aligned


def align_face_from_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    target_size: tuple = TARGET_SIZE,
    margin: float = 0.0,
) -> np.ndarray:
    """
    Align face using landmarks if available, otherwise fall back to bbox crop.
    
    Args:
        image: Input image (BGR)
        bbox: Face bounding box [x1, y1, x2, y2]
        landmarks: Optional 5x2 landmarks
        target_size: Output size
        margin: Extra margin around bbox (as fraction of bbox size)
    
    Returns:
        Aligned/cropped face image
    """
    # If we have landmarks, use proper alignment
    if landmarks is not None and len(landmarks) >= 5:
        aligned = align_face(image, landmarks, target_size)
        if aligned is not None:
            return aligned
    
    # Fallback: simple bbox crop with resize
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # Add margin
    if margin > 0:
        w, h = x2 - x1, y2 - y1
        mx, my = int(w * margin), int(h * margin)
        x1, y1 = max(0, x1 - mx), max(0, y1 - my)
        x2, y2 = min(image.shape[1], x2 + mx), min(image.shape[0], y2 + my)
    
    # Clamp to image bounds
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    face_crop = image[y1:y2, x1:x2]
    return cv2.resize(face_crop, target_size)
