"""
lipsync.py
LipSync - Real-Time Lip Reading and Hand Gesture Translation
System for Industrial Communication.

Developed by Karthik.

Run with:
    streamlit run lipsync.py

See README.md for full setup instructions.
"""

import threading
import time
from datetime import datetime
from typing import Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from config import DEFAULT_CONFIG, LIP_VOCABULARY, GESTURE_MEANINGS
from database.db_manager import DBManager
from detectors.face_detector import FaceDetector
from detectors.hand_detector import HandDetector
from detectors.lip_detector import LipDetector
from detectors.person_detector import PersonDetector
from services.export_service import ExportService
from services.speech_service import SpeechService
from translators.combiner import Combiner
from ui.components import (
    render_history_table,
    render_navbar,
    render_page_footer,
    render_prediction_cards,
    render_sidebar_brand,
    render_sidebar_footer,
    render_subtitle_box,
)
from ui.styles import CUSTOM_CSS
from utils.logger import get_logger

logger = get_logger("app")

st.set_page_config(
    page_title="LipSync",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# ---------------------------------------------------------------------------
# Shared, thread-safe state between the WebRTC video-processing thread and
# the main Streamlit render thread (they are NOT the same thread).
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.person_status = "NONE"
        self.person_count = 0
        self.face_found = False
        self.face_confidence = 0.0
        self.gesture = ""
        self.gesture_meaning = ""
        self.gesture_confidence = 0.0
        self.lip_word = ""
        self.lip_confidence = 0.0
        self.lip_aperture = 0.0
        self.sentence = ""
        self.sentence_confidence = 0.0
        self.fps = 0.0
        self.last_sentence_saved = ""

    def snapshot(self) -> dict:
        with self.lock:
            return self.__dict__.copy()


@st.cache_resource
def get_shared_state() -> SharedState:
    return SharedState()


@st.cache_resource
def get_db() -> DBManager:
    return DBManager()


@st.cache_resource
def get_speech_service() -> SpeechService:
    return SpeechService()


@st.cache_resource
def get_export_service() -> ExportService:
    return ExportService()


if "app_config" not in st.session_state:
    st.session_state.app_config = DEFAULT_CONFIG

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ---------------------------------------------------------------------------
# Video processor: runs the full detection pipeline on every incoming frame.
# ---------------------------------------------------------------------------
class LipGestureVideoProcessor(VideoProcessorBase):
    def __init__(self):
        cfg = st.session_state.app_config.detection
        self.person_detector = PersonDetector(min_confidence=cfg.person_min_confidence)
        self.face_detector = FaceDetector(
            min_detection_confidence=cfg.face_min_confidence,
            min_tracking_confidence=cfg.face_min_tracking_confidence,
        )
        self.lip_detector = LipDetector(min_confidence=cfg.lip_min_confidence)
        self.hand_detector = HandDetector(
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.hand_min_confidence,
            min_tracking_confidence=cfg.hand_min_tracking_confidence,
        )
        self.combiner = Combiner()
        self.shared = get_shared_state()
        self.db = get_db()
        self._last_time = time.time()

    def _update_fps(self) -> float:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        return 1.0 / dt if dt > 0 else 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        fps = self._update_fps()

        person_result = self.person_detector.detect(img)

        gesture, gesture_meaning, gesture_conf = "", "", 0.0
        lip_word, lip_conf, lip_aperture = "", 0.0, 0.0
        face_found, face_conf = False, 0.0

        for (x, y, w, h) in person_result.boxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 140, 0), 2)

        if person_result.status == "NONE":
            cv2.putText(img, "No Person Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif person_result.status == "MULTIPLE":
            cv2.putText(img, "Multiple Persons Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            # Exactly one person -> run full pipeline
            face_result = self.face_detector.detect(img)
            face_found = face_result.found
            face_conf = face_result.confidence

            if face_found:
                x, y, w, h = face_result.bbox
                cv2.rectangle(img, (x, y), (x + w, y + h), (124, 92, 255), 2)
                cv2.putText(img, f"Face {face_conf*100:.0f}%", (x, max(0, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (124, 92, 255), 2)

                lip_result = self.lip_detector.process(face_result.landmarks)
                lip_aperture = lip_result.aperture
                if lip_result.word:
                    lip_word, lip_conf = lip_result.word, lip_result.confidence
                for (lx, ly) in lip_result.lip_points:
                    cv2.circle(img, (lx, ly), 2, (34, 211, 238), -1)

            hand_results = self.hand_detector.detect(img)
            if hand_results:
                best = max(hand_results, key=lambda h: h.confidence)
                gesture = best.gesture
                gesture_meaning = best.meaning
                gesture_conf = best.confidence
                for (hx, hy) in (best.landmarks or []):
                    cv2.circle(img, (hx, hy), 3, (244, 114, 182), -1)

        combined = self.combiner.combine(
            gesture_meaning if gesture_meaning else None,
            lip_word if lip_word else None,
        )
        sentence_conf = max(gesture_conf, lip_conf) if (gesture_meaning or lip_word) else 0.0

        if combined.sentence:
            cv2.rectangle(img, (0, img.shape[0] - 50), (img.shape[1], img.shape[0]), (10, 10, 15), -1)
            cv2.putText(img, combined.sentence, (16, img.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        with self.shared.lock:
            self.shared.person_status = person_result.status
            self.shared.person_count = person_result.count
            self.shared.face_found = face_found
            self.shared.face_confidence = face_conf
            self.shared.gesture = gesture
            self.shared.gesture_meaning = gesture_meaning
            self.shared.gesture_confidence = gesture_conf
            self.shared.lip_word = lip_word
            self.shared.lip_confidence = lip_conf
            self.shared.lip_aperture = lip_aperture
            self.shared.sentence = combined.sentence
            self.shared.sentence_confidence = sentence_conf
            self.shared.fps = fps

            should_save = (
                combined.sentence
                and combined.sentence != self.shared.last_sentence_saved
                and sentence_conf >= 0.4
            )
            if should_save:
                self.shared.last_sentence_saved = combined.sentence

        if should_save:
            try:
                self.db.insert_prediction(gesture_meaning, lip_word, combined.sentence, sentence_conf)
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to save prediction: %s", exc)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def get_cpu_usage() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_brand()
    st.session_state.page = st.radio(
        "Navigate",
        ["Dashboard", "Live Detection", "Hand Signs", "Lip Reading",
         "Subtitle History", "Settings", "About"],
        label_visibility="collapsed",
    )
    render_sidebar_footer()


shared = get_shared_state()
db = get_db()
speech = get_speech_service()
exporter = get_export_service()
cfg = st.session_state.app_config


def render_live_camera():
    webrtc_ctx = webrtc_streamer(
        key="lipgesture-stream",
        video_processor_factory=LipGestureVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    return webrtc_ctx


# ---------------------------------------------------------------------------
# PAGE: Dashboard / Live Detection (same core view, kept as two entries per
# the sidebar spec; Dashboard adds the statistics panel, Live Detection is
# a focused full-width camera view).
# ---------------------------------------------------------------------------
def page_dashboard(show_stats: bool = True):
    snap = shared.snapshot()
    render_navbar(
        camera_status="LIVE" if snap["fps"] > 0 else "OFFLINE",
        fps=snap["fps"],
        cpu=get_cpu_usage(),
        current_time=datetime.now().strftime("%H:%M:%S"),
    )

    main_col, side_col = st.columns([2.2, 1]) if show_stats else (st.container(), None)

    with main_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        render_live_camera()
        st.markdown("</div>", unsafe_allow_html=True)

        if snap["person_status"] == "NONE":
            st.warning("No Person Detected")
        elif snap["person_status"] == "MULTIPLE":
            st.warning(f"Multiple Persons Detected ({snap['person_count']})")

        render_subtitle_box(snap["sentence"], snap["sentence_confidence"])
        render_prediction_cards(
            snap["gesture_meaning"], snap["gesture_confidence"],
            snap["lip_word"], snap["lip_confidence"],
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🔊 Speak Sentence", use_container_width=True):
                if not cfg.tts_muted and snap["sentence"]:
                    speech.set_muted(cfg.tts_muted)
                    speech.set_volume(cfg.tts_volume)
                    speech.speak(snap["sentence"])
        with c2:
            if st.button("📋 Copy Latest", use_container_width=True):
                st.code(snap["sentence"] or "(empty)")
        with c3:
            if st.button("🗑️ Clear History", use_container_width=True):
                db.clear_history()
                st.success("History cleared")

    if show_stats and side_col is not None:
        with side_col:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header" style="font-size:1.1rem;">📊 Statistics</div>',
                        unsafe_allow_html=True)
            records = db.fetch_history(limit=50)
            recent_words = [r.lip_word for r in records if r.lip_word][:8]
            recent_gestures = [r.gesture for r in records if r.gesture][:8]
            st.markdown("**Recent Words**")
            st.write(", ".join(recent_words) if recent_words else "—")
            st.markdown("**Recent Gestures**")
            st.write(", ".join(recent_gestures) if recent_gestures else "—")
            st.markdown("**Confidence Trend**")
            if records:
                st.line_chart([r.confidence for r in reversed(records[:20])])
            else:
                st.write("No data yet")
            st.markdown("</div>", unsafe_allow_html=True)


def page_hand_signs():
    st.markdown('<div class="section-header">🖐️ Hand Signs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Industrial hand-gesture vocabulary recognised live.</div>',
                unsafe_allow_html=True)
    snap = shared.snapshot()
    render_prediction_cards(snap["gesture_meaning"], snap["gesture_confidence"], "", 0.0)
    st.markdown("### Supported Gestures")
    cols = st.columns(3)
    for i, (gkey, meaning) in enumerate(GESTURE_MEANINGS.items()):
        if gkey == "UNKNOWN":
            continue
        with cols[i % 3]:
            st.markdown(
                f'<div class="glass-card"><b>{gkey.replace("_"," ").title()}</b>'
                f'<div style="color:var(--text-secondary);">Meaning: {meaning}</div></div>',
                unsafe_allow_html=True,
            )


def page_lip_reading():
    st.markdown('<div class="section-header">👄 Lip Reading</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Heuristic mouth-shape matcher over a fixed vocabulary '
        "(see README for how this differs from a trained deep-learning model).</div>",
        unsafe_allow_html=True,
    )
    snap = shared.snapshot()
    render_prediction_cards("", 0.0, snap["lip_word"], snap["lip_confidence"])
    st.markdown(f"**Live mouth aperture:** {snap['lip_aperture']:.3f}")
    st.markdown("### Vocabulary")
    st.write(", ".join(LIP_VOCABULARY))


def page_subtitle_history():
    st.markdown('<div class="section-header">📜 Subtitle History</div>', unsafe_allow_html=True)
    records = db.fetch_history(limit=300)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🗑️ Clear", use_container_width=True):
            db.clear_history()
            st.rerun()
    with c2:
        st.download_button("⬇️ TXT", data=exporter.to_txt(records),
                            file_name="subtitle_history.txt", use_container_width=True)
    with c3:
        st.download_button("⬇️ CSV", data=exporter.to_csv(records),
                            file_name="subtitle_history.csv", use_container_width=True)
    with c4:
        try:
            pdf_bytes = exporter.to_pdf(records)
            st.download_button("⬇️ PDF", data=pdf_bytes, file_name="subtitle_history.pdf",
                                use_container_width=True)
        except Exception:
            st.button("⬇️ PDF (install fpdf2)", disabled=True, use_container_width=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_history_table(records)
    st.markdown("</div>", unsafe_allow_html=True)


def page_settings():
    st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)
    st.caption("Detection-sensitivity changes apply the next time you start the camera stream.")

    d = cfg.detection
    col1, col2 = st.columns(2)
    with col1:
        d.person_min_confidence = st.slider("Person Detection Confidence", 0.1, 1.0, d.person_min_confidence)
        d.face_min_confidence = st.slider("Face Detection Confidence", 0.1, 1.0, d.face_min_confidence)
        d.lip_min_confidence = st.slider("Lip Reading Confidence", 0.1, 1.0, d.lip_min_confidence)
        d.hand_min_confidence = st.slider("Gesture Confidence", 0.1, 1.0, d.hand_min_confidence)
    with col2:
        cfg.camera_index = st.number_input("Camera Index", 0, 10, cfg.camera_index)
        cfg.subtitle_font_size = st.slider("Subtitle Font Size", 16, 48, cfg.subtitle_font_size)
        cfg.language = st.selectbox("Language", ["en", "hi", "ta"], index=["en", "hi", "ta"].index(cfg.language))
        cfg.theme = st.selectbox("Theme", ["dark"], index=0, disabled=True)

    st.markdown("### Speech Settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg.tts_enabled = st.checkbox("Enable Speech Output", value=cfg.tts_enabled)
    with c2:
        cfg.tts_muted = st.checkbox("Mute", value=cfg.tts_muted)
    with c3:
        cfg.tts_volume = st.slider("Volume", 0.0, 1.0, cfg.tts_volume)

    voices = speech.list_voices()
    if voices:
        names = [v["name"] for v in voices]
        chosen = st.selectbox("Voice", names)
        speech.set_voice(next(v["id"] for v in voices if v["name"] == chosen))
    else:
        st.caption("No system TTS voices detected (this is OS-dependent; pyttsx3 uses your OS voices).")

    st.session_state.app_config = cfg
    st.success("Settings applied.")


def page_about():
    st.markdown('<div class="section-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="glass-card">
<b>Project:</b> LipSync — Real-Time Lip Reading and Hand Gesture
Translation System for Industrial Communication<br>
<b>Developed by:</b> Karthik<br>
<b>Version:</b> 1.0.0<br>
<b>License:</b> MIT (for educational/mini-project use)<br><br>
<b>Technologies Used:</b> Python, OpenCV, MediaPipe, Streamlit, streamlit-webrtc,
NumPy, Pandas, SQLite, pyttsx3, fpdf2
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="glass-card">
<b>Note on Lip Reading:</b> This project uses a heuristic mouth-shape matcher
over a small fixed vocabulary rather than a trained deep-learning speech
model, since accurate visual speech recognition requires large labelled
video datasets and GPU training infrastructure that are out of scope for a
local mini project. See README.md for details and ideas on how to extend
this into a trained model later.
</div>
""",
        unsafe_allow_html=True,
    )


page = st.session_state.page
if page == "Dashboard":
    page_dashboard(show_stats=True)
elif page == "Live Detection":
    page_dashboard(show_stats=False)
elif page == "Hand Signs":
    page_hand_signs()
elif page == "Lip Reading":
    page_lip_reading()
elif page == "Subtitle History":
    page_subtitle_history()
elif page == "Settings":
    page_settings()
elif page == "About":
    page_about()

render_page_footer()
