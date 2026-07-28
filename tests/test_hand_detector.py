"""
tests/test_hand_detector.py
Unit tests for the rule-based gesture classifier. These test the pure
geometry logic with synthetic landmark points - no webcam/mediapipe runtime
needed, so they run anywhere (including CI).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectors.hand_detector import HandDetector  # noqa: E402


def make_points(overrides: dict) -> list:
    """Build a baseline 21-point hand skeleton (all fingers curled/fist-like),
    then apply per-index (x, y) overrides for the test scenario."""
    base = [(100, 200)] * 21
    base[0] = (100, 220)   # wrist
    pts = list(base)
    for idx, (x, y) in overrides.items():
        pts[idx] = (x, y)
    return pts


def test_open_palm():
    detector = HandDetector.__new__(HandDetector)  # skip mediapipe init
    overrides = {
        0: (100, 220),   # wrist
        4: (60, 150), 3: (80, 170),     # thumb out (tip further from wrist.x than ip)
        8: (100, 100), 6: (100, 150),   # index up
        12: (105, 95), 10: (105, 150),  # middle up
        16: (110, 100), 14: (110, 150), # ring up
        20: (115, 110), 18: (115, 150), # pinky up
    }
    pts = make_points(overrides)
    gesture, conf = detector._classify(pts)
    assert gesture == "OPEN_PALM"
    assert conf > 0.5


def test_fist():
    detector = HandDetector.__new__(HandDetector)
    overrides = {
        0: (100, 220),
        4: (99, 210), 3: (90, 205),      # thumb tucked (tip closer to wrist.x than ip)
        8: (100, 180), 6: (100, 150),    # index down (tip below pip)
        12: (105, 180), 10: (105, 150),
        16: (110, 180), 14: (110, 150),
        20: (115, 180), 18: (115, 150),
    }
    pts = make_points(overrides)
    gesture, conf = detector._classify(pts)
    assert gesture == "FIST"


def test_pointing():
    detector = HandDetector.__new__(HandDetector)
    overrides = {
        0: (100, 220),
        4: (99, 210), 3: (90, 205),      # thumb tucked
        8: (100, 100), 6: (100, 150),    # index up
        12: (105, 180), 10: (105, 150),  # middle down
        16: (110, 180), 14: (110, 150),
        20: (115, 180), 18: (115, 150),
    }
    pts = make_points(overrides)
    gesture, conf = detector._classify(pts)
    assert gesture == "POINTING"


def test_too_few_points_returns_unknown():
    detector = HandDetector.__new__(HandDetector)
    gesture, conf = detector._classify([(0, 0)] * 5)
    assert gesture == "UNKNOWN"
    assert conf == 0.0


if __name__ == "__main__":
    test_open_palm()
    test_fist()
    test_pointing()
    test_too_few_points_returns_unknown()
    print("All hand_detector tests passed.")
