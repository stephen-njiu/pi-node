"""
SCRFD Face Detector using ONNX Runtime.
Detects faces and facial landmarks in images.
"""

import numpy as np
import cv2
import logging
from dataclasses import dataclass
from typing import Optional

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print("Warning: onnxruntime not installed")


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Face detection result."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    score: float
    landmarks: Optional[np.ndarray]  # 5 landmarks: 2 eyes, nose, 2 mouth corners


class SCRFDDetector:
    """
    SCRFD (Sample and Computation Redistribution for Efficient Face Detection).
    Uses ONNX model for face detection with landmarks.
    """
    
    def __init__(
        self,
        model_path: str = "models/scrfd_10g_bnkps.onnx",
        input_size: tuple = (640, 640),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        self._session = None
        self._input_name = None
        self._output_names = None
        
        # Feature map strides for SCRFD
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model."""
        if ort is None:
            logger.error("ONNX Runtime not available")
            return
        
        try:
            # Use CPU execution provider (GPU optional)
            providers = ['CPUExecutionProvider']
            
            # Try GPU if available
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self._session = ort.InferenceSession(
                self.model_path,
                providers=providers
            )
            
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            
            logger.info(f"Loaded SCRFD model from {self.model_path}")
            logger.info(f"Using providers: {self._session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load SCRFD model: {e}")
            self._session = None
    
    def _preprocess(self, image: np.ndarray) -> tuple:
        """
        Preprocess image for model input.
        Returns (blob, scale, pad).
        """
        img = image.copy()
        
        # Calculate scale to fit input size while maintaining aspect ratio
        h, w = img.shape[:2]
        target_h, target_w = self.input_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        img = cv2.resize(img, (new_w, new_h))
        
        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        
        img = cv2.copyMakeBorder(
            img, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        
        # Convert to blob
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 128.0, self.input_size,
            (127.5, 127.5, 127.5), swapRB=True
        )
        
        return blob, scale, (0, 0)  # pad offset not used
    
    def _distance2bbox(self, points, distance):
        """Convert distance to bounding box."""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def _distance2kps(self, points, distance):
        """Convert distance to keypoints."""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)
    
    def _postprocess(self, outputs: list, scale: float, orig_size: tuple) -> list[Detection]:
        """
        Postprocess model outputs to get detections.
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []
        
        input_h, input_w = self.input_size
        
        for idx, stride in enumerate(self._feat_stride_fpn):
            # Get outputs for this stride
            scores = outputs[idx]
            bbox_preds = outputs[idx + len(self._feat_stride_fpn)]
            
            # Keypoints (if available)
            kps_preds = None
            if len(outputs) > len(self._feat_stride_fpn) * 2:
                kps_preds = outputs[idx + len(self._feat_stride_fpn) * 2]
            
            # Generate anchor centers
            height = input_h // stride
            width = input_w // stride
            
            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)
            
            if self._num_anchors > 1:
                anchor_centers = np.stack(
                    [anchor_centers] * self._num_anchors, axis=1
                ).reshape(-1, 2)
            
            # Get valid detections
            scores = scores.reshape(-1)
            pos_inds = np.where(scores >= self.conf_threshold)[0]
            
            if len(pos_inds) == 0:
                continue
            
            # Decode bboxes
            bbox_preds = bbox_preds.reshape(-1, 4)
            bboxes = self._distance2bbox(anchor_centers, bbox_preds * stride)
            
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            
            # Decode keypoints
            if kps_preds is not None:
                kps_preds = kps_preds.reshape(-1, 10)
                kpss = self._distance2kps(anchor_centers, kps_preds * stride)
                kpss_list.append(kpss[pos_inds])
        
        if len(scores_list) == 0:
            return []
        
        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list) if kpss_list else None
        
        # Scale back to original image
        bboxes = bboxes / scale
        if kpss is not None:
            kpss = kpss / scale
        
        # Clip to image bounds
        h, w = orig_size
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, w)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, h)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, h)
        
        # NMS
        keep = self._nms(bboxes, scores)
        
        detections = []
        for i in keep:
            det = Detection(
                bbox=bboxes[i],
                score=float(scores[i]),
                landmarks=kpss[i].reshape(5, 2) if kpss is not None else None
            )
            detections.append(det)
        
        return detections
    
    def _nms(self, bboxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """Non-Maximum Suppression."""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Detect faces in image.
        
        Args:
            image: BGR image (H, W, 3)
        
        Returns:
            List of Detection objects
        """
        if self._session is None:
            logger.warning("Model not loaded, returning empty detections")
            return []
        
        orig_size = image.shape[:2]
        
        # Preprocess
        blob, scale, _ = self._preprocess(image)
        
        # Run inference
        try:
            outputs = self._session.run(self._output_names, {self._input_name: blob})
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return []
        
        # Postprocess
        detections = self._postprocess(outputs, scale, orig_size)
        
        return detections
    
    def detect_align(self, image: np.ndarray) -> list[tuple]:
        """
        Detect faces and return aligned face crops for recognition.
        
        Args:
            image: BGR image
        
        Returns:
            List of (aligned_face, detection) tuples
        """
        detections = self.detect(image)
        
        results = []
        for det in detections:
            if det.landmarks is not None:
                aligned = self._align_face(image, det.landmarks)
                results.append((aligned, det))
            else:
                # Fallback: crop without alignment
                bbox = det.bbox.astype(int)
                crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if crop.size > 0:
                    crop = cv2.resize(crop, (112, 112))
                    results.append((crop, det))
        
        return results
    
    def _align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face using 5-point landmarks.
        Returns 112x112 aligned face crop.
        """
        # Standard template for 112x112 face
        template = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Estimate affine transform
        src = landmarks.astype(np.float32)
        M = cv2.estimateAffinePartial2D(src, template)[0]
        
        if M is None:
            # Fallback
            return cv2.resize(image, (112, 112))
        
        # Apply transform
        aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0)
        
        return aligned
