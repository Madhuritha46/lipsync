"""
detectors/person_detector.py

Determines how many people are visible in a frame.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from mediapipe.python.solutions import face_detection

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class PersonDetectionResult:
    count: int
    boxes: List[Tuple[int, int, int, int]]
    status: str


class PersonDetector:
    """Wraps MediaPipe Face Detection to count people in a frame."""

    def __init__(self, min_confidence: float = 0.5):
        self._detector = face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_confidence
        )

        logger.info(
            "PersonDetector initialised (min_confidence=%.2f)",
            min_confidence
        )

    def detect(
        self,
        frame_bgr: np.ndarray
    ) -> PersonDetectionResult:

        rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = self._detector.process(rgb)

        boxes: List[Tuple[int, int, int, int]] = []

        h, w, _ = frame_bgr.shape

        if results.detections:
            for det in results.detections:

                bbox = det.location_data.relative_bounding_box

                x = max(
                    0,
                    int(bbox.xmin * w)
                )

                y = max(
                    0,
                    int(bbox.ymin * h)
                )

                bw = int(
                    bbox.width * w
                )

                bh = int(
                    bbox.height * h
                )

                boxes.append(
                    (x, y, bw, bh)
                )

        count = len(boxes)

        if count == 0:
            status = "NONE"

        elif count == 1:
            status = "SINGLE"

        else:
            status = "MULTIPLE"

        return PersonDetectionResult(
            count=count,
            boxes=boxes,
            status=status
        )

    def close(self) -> None:
        self._detector.close()