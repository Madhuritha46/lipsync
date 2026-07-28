"""
detectors/hand_detector.py
Detects hands with MediaPipe Hands and classifies a fixed set of industrial
communication gestures using rule-based geometric analysis of the 21 hand
landmarks (finger-up/down state, thumb orientation, etc.).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import GESTURE_MEANINGS
from utils.logger import get_logger

logger = get_logger(__name__)

# Landmark indices (MediaPipe Hands, 21 points)
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


@dataclass
class HandGestureResult:
    found: bool
    landmarks: Optional[List[Tuple[int, int]]]
    handedness: Optional[str]
    gesture: str          # raw gesture key, e.g. "OPEN_PALM"
    meaning: str           # industrial meaning, e.g. "STOP"
    confidence: float


class HandDetector:
    """Wraps MediaPipe Hands + a rule-based gesture classifier."""

    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info("HandDetector initialised (max_hands=%d)", max_num_hands)

    def detect(self, frame_bgr: np.ndarray) -> List[HandGestureResult]:
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        outputs: List[HandGestureResult] = []
        if not results.multi_hand_landmarks:
            return outputs

        handedness_list = results.multi_handedness or []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            label = None
            score = 0.0
            if idx < len(handedness_list):
                classification = handedness_list[idx].classification[0]
                label = classification.label
                score = classification.score

            gesture, gesture_conf = self._classify(pts)
            meaning = GESTURE_MEANINGS.get(gesture, "UNKNOWN")

            outputs.append(HandGestureResult(
                found=True,
                landmarks=pts,
                handedness=label,
                gesture=gesture,
                meaning=meaning,
                confidence=gesture_conf,
            ))
        return outputs

    # -- rule-based classification -----------------------------------------
    def _finger_up(self, pts: List[Tuple[int, int]], tip: int, pip: int) -> bool:
        """A finger is 'up' if its tip is above (smaller y) its PIP joint."""
        return pts[tip][1] < pts[pip][1]

    def _thumb_up(self, pts: List[Tuple[int, int]]) -> bool:
        """Thumb is 'out' if tip is farther from the wrist (x-axis) than IP joint."""
        wrist_x = pts[WRIST][0]
        return abs(pts[THUMB_TIP][0] - wrist_x) > abs(pts[THUMB_IP][0] - wrist_x)

    def _classify(self, pts: List[Tuple[int, int]]) -> Tuple[str, float]:
        if len(pts) < 21:
            return "UNKNOWN", 0.0

        index_up = self._finger_up(pts, INDEX_TIP, INDEX_PIP)
        middle_up = self._finger_up(pts, MIDDLE_TIP, MIDDLE_PIP)
        ring_up = self._finger_up(pts, RING_TIP, RING_PIP)
        pinky_up = self._finger_up(pts, PINKY_TIP, PINKY_PIP)
        thumb_out = self._thumb_up(pts)

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

        # Thumb tip clearly above wrist and all other fingers curled -> thumbs up
        if thumb_out and fingers_up == 0 and pts[THUMB_TIP][1] < pts[WRIST][1]:
            return "THUMBS_UP", 0.9

        # Open palm: all 4 fingers + thumb extended
        if fingers_up == 4 and thumb_out:
            return "OPEN_PALM", 0.9

        # Fist: no fingers up, thumb tucked
        if fingers_up == 0 and not thumb_out:
            return "FIST", 0.85

        # Pointing: only index finger up
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "POINTING", 0.85

        # Peace / victory sign: index + middle up, others down
        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE", 0.85

        # Four fingers up, thumb tucked
        if fingers_up == 4 and not thumb_out:
            return "FOUR_FINGERS", 0.8

        # OK sign: thumb tip close to index tip, other 3 fingers up
        thumb_index_dist = np.linalg.norm(
            np.array(pts[THUMB_TIP]) - np.array(pts[INDEX_TIP])
        )
        hand_scale = np.linalg.norm(np.array(pts[WRIST]) - np.array(pts[MIDDLE_PIP])) + 1e-6
        if thumb_index_dist / hand_scale < 0.35 and middle_up and ring_up and pinky_up:
            return "OK_SIGN", 0.8

        # Call-me sign: thumb + pinky out, others curled
        if thumb_out and pinky_up and not index_up and not middle_up and not ring_up:
            return "CALL_ME", 0.75

        return "UNKNOWN", 0.3

    def close(self) -> None:
        self._hands.close()
