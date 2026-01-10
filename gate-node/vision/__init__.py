"""Vision module for face detection, recognition, and tracking."""

from .detector import SCRFDDetector
from .recognizer import ArcFaceRecognizer
from .tracker import SimpleTracker, DeepSORTLiteTracker, Track, TrackPhase, TrackerStatistics
from .alignment import align_face, align_face_from_bbox
from .quality import (
    assess_face_quality,
    filter_quality_detections,
    QualityResult,
    MIN_FACE_WIDTH,
    BLUR_THRESHOLD,
)

__all__ = [
    "SCRFDDetector", 
    "ArcFaceRecognizer", 
    "SimpleTracker",
    "DeepSORTLiteTracker",
    "Track",
    "TrackPhase",
    "TrackerStatistics",
    "align_face",
    "align_face_from_bbox",
    "assess_face_quality",
    "filter_quality_detections",
    "QualityResult",
    "MIN_FACE_WIDTH",
    "BLUR_THRESHOLD",
]
