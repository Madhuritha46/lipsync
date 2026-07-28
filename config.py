"""
config.py
Central configuration for LipSync.
All tunable parameters, vocabularies, and constants live here so
detectors/services never hardcode magic numbers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "database", "lipgesture.db")
EXPORT_DIR = os.path.join(BASE_DIR, "history_exports")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@dataclass
class DetectionConfig:
    """Confidence thresholds for all detectors. Adjustable from Settings page."""
    person_min_confidence: float = 0.5
    face_min_confidence: float = 0.5
    face_min_tracking_confidence: float = 0.5
    hand_min_confidence: float = 0.6
    hand_min_tracking_confidence: float = 0.5
    lip_min_confidence: float = 0.45
    max_num_hands: int = 2


@dataclass
class AppConfig:
    """Global application settings, editable at runtime via Settings page."""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    subtitle_font_size: int = 28
    theme: str = "dark"
    language: str = "en"
    tts_enabled: bool = True
    tts_volume: float = 1.0
    tts_muted: bool = False
    detection: DetectionConfig = field(default_factory=DetectionConfig)


# ---------------------------------------------------------------------------
# Lip-reading vocabulary (heuristic mouth-shape matcher, NOT a trained model).
# Each word is described by a template of mouth-aperture behaviour over a
# short rolling window. See detectors/lip_detector.py for how this is used.
# ---------------------------------------------------------------------------
LIP_VOCABULARY: List[str] = [
    "HELLO",
    "GOOD MORNING",
    "STOP",
    "HELP",
    "YES",
    "NO",
    "THANK YOU",
]

# mean_aperture (0-1 normalised), std_aperture, num_peaks (mouth open/close cycles)
LIP_TEMPLATES: Dict[str, Tuple[float, float, int]] = {
    "HELLO":        (0.35, 0.12, 2),
    "GOOD MORNING": (0.40, 0.15, 4),
    "STOP":         (0.20, 0.05, 1),
    "HELP":         (0.30, 0.10, 2),
    "YES":          (0.25, 0.08, 1),
    "NO":           (0.22, 0.09, 1),
    "THANK YOU":    (0.33, 0.13, 3),
}

# ---------------------------------------------------------------------------
# Hand-gesture vocabulary for industrial communication.
# ---------------------------------------------------------------------------
GESTURE_MEANINGS: Dict[str, str] = {
    "THUMBS_UP": "OK",
    "OPEN_PALM": "STOP",
    "FIST": "EMERGENCY",
    "POINTING": "MOVE",
    "PEACE": "WAIT",
    "OK_SIGN": "OK",
    "CALL_ME": "COME",
    "FOUR_FINGERS": "GO",
    "UNKNOWN": "UNKNOWN",
}

# Combination rules: (gesture_meaning, lip_word) -> combined sentence
COMBINATION_RULES: Dict[Tuple[str, str], str] = {
    ("STOP", "HELP"): "STOP AND HELP ME",
    ("STOP", "YES"): "STOP THE MACHINE",
    ("HELP", "HELP"): "PLEASE HELP",
    ("EMERGENCY", "HELP"): "EMERGENCY, NEED HELP",
    ("OK", "YES"): "CONFIRMED, PROCEED",
    ("WAIT", "NO"): "WAIT, DO NOT PROCEED",
    ("MOVE", "YES"): "MOVE THE MACHINE",
    ("GO", "YES"): "GO AHEAD",
}

DEFAULT_CONFIG = AppConfig()
