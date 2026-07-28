"""
detectors/lip_detector.py

IMPORTANT / HONEST NOTE:
Real deep-learning lip reading (e.g. LipNet-style visual speech recognition)
requires a model trained on thousands of labelled lip-video clips (GRID
corpus etc.) plus a GPU training pipeline. That is out of scope for a local
mini project with no dataset.

What this module actually does instead:
  1. Extracts lip landmarks from the 468-point face mesh (MediaPipe indices).
  2. Tracks mouth "aperture" (how open the mouth is, normalised by face
     width) across a short rolling window of frames.
  3. Computes simple statistical features of that window (mean aperture,
     std deviation, number of open/close cycles).
  4. Matches those features against hand-authored templates for a small,
     fixed vocabulary (config.LIP_TEMPLATES) using a distance score.

This is a legitimate heuristic approach for a demo, not a trained
speech-recognition model. Document this distinction in your project report.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import LIP_TEMPLATES
from utils.logger import get_logger

logger = get_logger(__name__)

# MediaPipe FaceMesh landmark indices relevant to the lips.
UPPER_LIP_TOP = 13
LOWER_LIP_BOTTOM = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
OUTER_LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


@dataclass
class LipReadingResult:
    lip_points: List[Tuple[int, int]]
    aperture: float          # normalised mouth-open ratio for this frame
    word: Optional[str]      # best-matching vocabulary word, or None
    confidence: float        # 0-1 match confidence


class LipDetector:
    """Tracks lip movement and heuristically maps it to a vocabulary word."""

    def __init__(self, window_size: int = 24, min_confidence: float = 0.45):
        self._window: deque = deque(maxlen=window_size)
        self._min_confidence = min_confidence
        logger.info("LipDetector initialised (window=%d)", window_size)

    def extract_lip_points(self, face_landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Return pixel coordinates of the outer lip contour."""
        return [face_landmarks[i] for i in OUTER_LIP_IDX if i < len(face_landmarks)]

    def _compute_aperture(self, face_landmarks: List[Tuple[int, int]]) -> float:
        top = np.array(face_landmarks[UPPER_LIP_TOP])
        bottom = np.array(face_landmarks[LOWER_LIP_BOTTOM])
        left = np.array(face_landmarks[LEFT_MOUTH_CORNER])
        right = np.array(face_landmarks[RIGHT_MOUTH_CORNER])

        mouth_height = np.linalg.norm(top - bottom)
        mouth_width = np.linalg.norm(left - right) + 1e-6
        return float(mouth_height / mouth_width)

    def _match_template(self) -> Tuple[Optional[str], float]:
        if len(self._window) < self._window.maxlen:
            return None, 0.0

        values = np.array(self._window)
        mean_ap = float(np.mean(values))
        std_ap = float(np.std(values))

        # count peaks (simple sign-change-of-derivative peak counter)
        diffs = np.diff(values)
        peaks = 0
        for i in range(1, len(diffs)):
            if diffs[i - 1] > 0 and diffs[i] <= 0:
                peaks += 1

        best_word, best_score = None, -1.0
        for word, (t_mean, t_std, t_peaks) in LIP_TEMPLATES.items():
            dist = (
                abs(mean_ap - t_mean) / (t_mean + 1e-6)
                + abs(std_ap - t_std) / (t_std + 1e-6)
                + abs(peaks - t_peaks) / max(t_peaks, 1)
            )
            score = 1.0 / (1.0 + dist)
            if score > best_score:
                best_word, best_score = word, score

        if best_score < self._min_confidence:
            return None, best_score
        return best_word, best_score

    def process(self, face_landmarks: Optional[List[Tuple[int, int]]]) -> LipReadingResult:
        if not face_landmarks or len(face_landmarks) <= max(OUTER_LIP_IDX):
            return LipReadingResult(lip_points=[], aperture=0.0, word=None, confidence=0.0)

        aperture = self._compute_aperture(face_landmarks)
        self._window.append(aperture)
        word, confidence = self._match_template()
        lip_points = self.extract_lip_points(face_landmarks)

        return LipReadingResult(
            lip_points=lip_points, aperture=aperture, word=word, confidence=confidence
        )

    def reset(self) -> None:
        self._window.clear()
