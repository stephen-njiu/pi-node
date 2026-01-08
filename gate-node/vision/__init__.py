"""Vision module for face detection, recognition, and tracking."""

from .detector import SCRFDDetector
from .recognizer import ArcFaceRecognizer
from .tracker import SimpleTracker, DeepSORTLiteTracker, Track, TrackPhase, TrackerStatistics

# Re-export align_face from detector
def align_face(image, landmarks):
    """Align face using 5-point landmarks (wrapper for detector method)."""
    import cv2
    import numpy as np
    
    # Standard template for 112x112 face
    template = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    
    src = landmarks.astype(np.float32)
    M = cv2.estimateAffinePartial2D(src, template)[0]
    
    if M is None:
        return cv2.resize(image, (112, 112))
    
    return cv2.warpAffine(image, M, (112, 112), borderValue=0)

__all__ = [
    "SCRFDDetector", 
    "ArcFaceRecognizer", 
    "SimpleTracker",
    "DeepSORTLiteTracker",
    "Track",
    "TrackPhase",
    "TrackerStatistics",
    "align_face",
]
