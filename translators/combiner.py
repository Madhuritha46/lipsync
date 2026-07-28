"""
translators/combiner.py
Combines a lip-reading word and a hand-gesture meaning into a single
industrial-communication sentence, using a rule table with a sensible
fallback when no explicit rule matches.
"""

from dataclasses import dataclass
from typing import Optional

from config import COMBINATION_RULES
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CombinedTranslation:
    gesture_meaning: Optional[str]
    lip_word: Optional[str]
    sentence: str


class Combiner:
    """Fuses lip-reading output and gesture output into one sentence."""

    def combine(self, gesture_meaning: Optional[str], lip_word: Optional[str]) -> CombinedTranslation:
        if not gesture_meaning and not lip_word:
            return CombinedTranslation(gesture_meaning, lip_word, "")

        if gesture_meaning and lip_word:
            key = (gesture_meaning, lip_word)
            if key in COMBINATION_RULES:
                sentence = COMBINATION_RULES[key]
            else:
                # Fallback: simple natural concatenation
                sentence = f"{gesture_meaning} {lip_word}".strip()
            return CombinedTranslation(gesture_meaning, lip_word, sentence)

        if gesture_meaning:
            return CombinedTranslation(gesture_meaning, lip_word, gesture_meaning)

        return CombinedTranslation(gesture_meaning, lip_word, lip_word or "")
