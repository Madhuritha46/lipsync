"""
services/speech_service.py
Offline text-to-speech wrapper using pyttsx3 (no internet required, works
fully locally in VS Code). A new engine instance is created per call because
pyttsx3 engines do not play well with being reused across Streamlit reruns.
"""

from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class SpeechService:
    """Wraps pyttsx3 for local, offline text-to-speech output."""

    def __init__(self):
        self._muted = False
        self._volume = 1.0
        self._voice_id: Optional[str] = None

    def list_voices(self) -> List[dict]:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            result = [{"id": v.id, "name": v.name} for v in voices]
            engine.stop()
            return result
        except Exception as exc:  # pragma: no cover - depends on OS TTS drivers
            logger.warning("Could not list TTS voices: %s", exc)
            return []

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))

    def set_voice(self, voice_id: Optional[str]) -> None:
        self._voice_id = voice_id

    def speak(self, text: str) -> bool:
        """Speak text synchronously. Returns True on success."""
        if self._muted or not text.strip():
            return False
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("volume", self._volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("TTS failed: %s", exc)
            return False
