"""
Face Quality Assessment for Gate Access Control.

Rejects faces that are:
- Too small (< MIN_FACE_WIDTH pixels)
- Too blurry (Laplacian variance < BLUR_THRESHOLD)
- Extreme head pose (estimated from landmarks)

This prevents low-quality faces from being sent to recognition,
which would waste compute and produce unreliable results.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of face quality assessment."""
    passed: bool
    face_width: int
    blur_score: float
    pose_score: float  # 0 = frontal, 1 = extreme profile
    rejection_reason: Optional[str] = None


# Quality thresholds
MIN_FACE_WIDTH = 80       # Minimum face width in pixels
BLUR_THRESHOLD = 100.0    # Laplacian variance below this = blurry
MAX_YAW_RATIO = 0.5       # Max asymmetry ratio for yaw detection
MAX_PITCH_RATIO = 0.4     # Max ratio for pitch detection


def compute_blur_score(face_image: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    Higher = sharper, lower = blurrier.
    
    Args:
        face_image: BGR face crop
        
    Returns:
        Laplacian variance (higher is sharper)
    """
    if face_image is None or face_image.size == 0:
        return 0.0
    
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return float(variance)


def estimate_pose_from_landmarks(landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Estimate head pose (yaw, pitch) from 5-point landmarks.
    
    Landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    Args:
        landmarks: 5x2 array of facial landmarks
        
    Returns:
        (yaw_score, pitch_score) where 0 = frontal, higher = more extreme
    """
    if landmarks is None or len(landmarks) < 5:
        return 0.0, 0.0
    
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # Eye distance (reference)
    eye_dist = np.linalg.norm(right_eye - left_eye)
    if eye_dist < 1:
        return 0.0, 0.0
    
    # YAW estimation: Compare nose position relative to eye midpoint
    # In frontal face, nose is centered between eyes
    eye_center = (left_eye + right_eye) / 2
    nose_offset = nose[0] - eye_center[0]
    yaw_ratio = abs(nose_offset) / (eye_dist / 2)
    yaw_score = min(yaw_ratio, 1.0)
    
    # PITCH estimation: Compare vertical distances
    # In frontal face, nose is roughly centered vertically
    mouth_center = (left_mouth + right_mouth) / 2
    face_height = mouth_center[1] - eye_center[1]
    
    if face_height > 0:
        nose_to_eyes = nose[1] - eye_center[1]
        nose_to_mouth = mouth_center[1] - nose[1]
        
        # Ideal ratio is roughly 1:1 for eyes-nose and nose-mouth
        if nose_to_mouth > 0:
            pitch_ratio = abs(nose_to_eyes / nose_to_mouth - 1.0)
        else:
            pitch_ratio = 1.0
        pitch_score = min(pitch_ratio, 1.0)
    else:
        pitch_score = 1.0  # Invalid geometry
    
    return yaw_score, pitch_score


def assess_face_quality(
    bbox: np.ndarray,
    landmarks: Optional[np.ndarray],
    frame: Optional[np.ndarray] = None,
    min_width: int = MIN_FACE_WIDTH,
    blur_threshold: float = BLUR_THRESHOLD,
    max_yaw: float = MAX_YAW_RATIO,
    max_pitch: float = MAX_PITCH_RATIO,
) -> QualityResult:
    """
    Assess face quality for recognition suitability.
    
    Args:
        bbox: Face bounding box [x1, y1, x2, y2]
        landmarks: 5-point facial landmarks (optional)
        frame: Full frame image for blur detection (optional)
        min_width: Minimum acceptable face width
        blur_threshold: Minimum Laplacian variance (higher = sharper required)
        max_yaw: Maximum yaw score (0-1) before rejection
        max_pitch: Maximum pitch score (0-1) before rejection
        
    Returns:
        QualityResult with pass/fail and scores
    """
    # Calculate face dimensions
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Default scores
    blur_score = BLUR_THRESHOLD + 1  # Default to passing
    pose_score = 0.0
    
    # =========================
    # CHECK 1: Face Size
    # =========================
    if face_width < min_width:
        return QualityResult(
            passed=False,
            face_width=face_width,
            blur_score=blur_score,
            pose_score=pose_score,
            rejection_reason=f"Face too small: {face_width}px < {min_width}px minimum"
        )
    
    # =========================
    # CHECK 2: Blur (if frame provided)
    # =========================
    if frame is not None:
        # Extract face region
        h, w = frame.shape[:2]
        x1_clamp = max(0, x1)
        y1_clamp = max(0, y1)
        x2_clamp = min(w, x2)
        y2_clamp = min(h, y2)
        
        if x2_clamp > x1_clamp and y2_clamp > y1_clamp:
            face_crop = frame[y1_clamp:y2_clamp, x1_clamp:x2_clamp]
            blur_score = compute_blur_score(face_crop)
            
            if blur_score < blur_threshold:
                return QualityResult(
                    passed=False,
                    face_width=face_width,
                    blur_score=blur_score,
                    pose_score=pose_score,
                    rejection_reason=f"Face too blurry: {blur_score:.1f} < {blur_threshold:.1f}"
                )
    
    # =========================
    # CHECK 3: Head Pose (if landmarks provided)
    # =========================
    if landmarks is not None and len(landmarks) >= 5:
        yaw_score, pitch_score = estimate_pose_from_landmarks(landmarks)
        pose_score = max(yaw_score, pitch_score)
        
        if yaw_score > max_yaw:
            return QualityResult(
                passed=False,
                face_width=face_width,
                blur_score=blur_score,
                pose_score=pose_score,
                rejection_reason=f"Extreme yaw: {yaw_score:.2f} > {max_yaw:.2f}"
            )
        
        if pitch_score > max_pitch:
            return QualityResult(
                passed=False,
                face_width=face_width,
                blur_score=blur_score,
                pose_score=pose_score,
                rejection_reason=f"Extreme pitch: {pitch_score:.2f} > {max_pitch:.2f}"
            )
    
    # All checks passed
    return QualityResult(
        passed=True,
        face_width=face_width,
        blur_score=blur_score,
        pose_score=pose_score,
        rejection_reason=None
    )


def filter_quality_detections(
    detections: list,
    frame: Optional[np.ndarray] = None,
    min_width: int = MIN_FACE_WIDTH,
    blur_threshold: float = BLUR_THRESHOLD,
    check_blur: bool = True,
    check_pose: bool = True,
) -> list:
    """
    Filter detections to keep only high-quality faces.
    
    Args:
        detections: List of Detection objects (with bbox, score, landmarks)
        frame: Full frame for blur detection
        min_width: Minimum face width
        blur_threshold: Minimum sharpness
        check_blur: Whether to check blur
        check_pose: Whether to check head pose
        
    Returns:
        Filtered list of detections that pass quality checks
    """
    filtered = []
    
    for det in detections:
        quality = assess_face_quality(
            bbox=det.bbox,
            landmarks=det.landmarks,
            frame=frame if check_blur else None,
            min_width=min_width,
            blur_threshold=blur_threshold if check_blur else 0,
        )
        
        if quality.passed:
            filtered.append(det)
        else:
            logger.debug(f"Rejected detection: {quality.rejection_reason}")
    
    rejected_count = len(detections) - len(filtered)
    if rejected_count > 0:
        logger.debug(f"Quality filter: {rejected_count}/{len(detections)} faces rejected")
    
    return filtered
