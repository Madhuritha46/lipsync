"""
tests/test_lip_detector.py
Unit tests for the heuristic lip-reading matcher's pure logic (aperture
computation + template matching), without needing a real camera/face mesh.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from detectors.lip_detector import LipDetector  # noqa: E402


def test_insufficient_landmarks_returns_empty_result():
    detector = LipDetector(window_size=5, min_confidence=0.0)
    result = detector.process(None)
    assert result.word is None
    assert result.aperture == 0.0


def test_window_must_fill_before_matching():
    detector = LipDetector(window_size=6, min_confidence=0.0)
    # Feed fewer frames than window_size -> should not match yet
    for _ in range(3):
        detector._window.append(0.2)
    word, score = detector._match_template()
    assert word is None


def test_template_match_after_window_fills():
    detector = LipDetector(window_size=24, min_confidence=0.0)
    # Synthetic single-hump pattern approximating the "STOP" template
    # (mean ~0.20, std ~0.05, 1 peak)
    values = 0.17 + 0.06 * np.sin(np.linspace(0, np.pi, 24))
    for v in values:
        detector._window.append(float(v))
    word, score = detector._match_template()
    assert word is not None
    assert 0.0 <= score <= 1.0


def test_compute_aperture_geometry():
    detector = LipDetector()
    # Build a fake 468-point landmark list where only the lip-relevant
    # indices matter for this calculation.
    landmarks = [(0, 0)] * 469
    landmarks[13] = (100, 50)   # upper lip top
    landmarks[14] = (100, 70)   # lower lip bottom -> mouth height = 20
    landmarks[61] = (80, 60)    # left mouth corner
    landmarks[291] = (120, 60)  # right mouth corner -> mouth width = 40
    aperture = detector._compute_aperture(landmarks)
    assert abs(aperture - 0.5) < 1e-6  # 20 / 40


if __name__ == "__main__":
    test_insufficient_landmarks_returns_empty_result()
    test_window_must_fill_before_matching()
    test_template_match_after_window_fills()
    test_compute_aperture_geometry()
    print("All lip_detector tests passed.")
