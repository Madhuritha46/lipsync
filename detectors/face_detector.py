"""
detectors/face_detector.py
Extracts face bounding box, 468-point face mesh landmarks, and an overall
confidence score using MediaPipe FaceMesh.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FaceDetectionResult:
    found: bool
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    landmarks: Optional[List[Tuple[int, int]]]  # pixel coords, 468 points
    confidence: float


class FaceDetector:
    """Wraps MediaPipe FaceMesh for landmark + bbox extraction."""

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info("FaceDetector initialised")

    def detect(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceDetectionResult(found=False, bbox=None, landmarks=None, confidence=0.0)

        face_landmarks = results.multi_face_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = max(min(xs), 0), min(max(xs), w)
        y_min, y_max = max(min(ys), 0), min(max(ys), h)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # MediaPipe doesn't expose a per-landmark confidence for FaceMesh;
        # we approximate a confidence score from bbox coverage as a proxy.
        area_ratio = (bbox[2] * bbox[3]) / float(w * h + 1e-6)
        confidence = float(min(1.0, 0.55 + area_ratio * 2.0))

        return FaceDetectionResult(found=True, bbox=bbox, landmarks=pts, confidence=confidence)

    def close(self) -> None:
        self._mesh.close()
