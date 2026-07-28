"""
tests/test_combiner.py
Unit tests for translators/combiner.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translators.combiner import Combiner  # noqa: E402


def test_combine_known_rule():
    combiner = Combiner()
    result = combiner.combine("HELP", "HELP")
    assert result.sentence == "PLEASE HELP"


def test_combine_fallback_concatenation():
    combiner = Combiner()
    result = combiner.combine("UNKNOWN_MEANING", "YES")
    assert "UNKNOWN_MEANING" in result.sentence and "YES" in result.sentence


def test_combine_gesture_only():
    combiner = Combiner()
    result = combiner.combine("STOP", None)
    assert result.sentence == "STOP"


def test_combine_lip_only():
    combiner = Combiner()
    result = combiner.combine(None, "HELLO")
    assert result.sentence == "HELLO"


def test_combine_neither():
    combiner = Combiner()
    result = combiner.combine(None, None)
    assert result.sentence == ""


if __name__ == "__main__":
    test_combine_known_rule()
    test_combine_fallback_concatenation()
    test_combine_gesture_only()
    test_combine_lip_only()
    test_combine_neither()
    print("All combiner tests passed.")
