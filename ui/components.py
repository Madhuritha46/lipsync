"""
ui/components.py
Small reusable Streamlit rendering helpers used across pages, so
lipsync.py stays focused on layout/orchestration rather than markup
strings.
"""

import streamlit as st
from typing import List
from database.db_manager import HistoryRecord


def status_pill(label: str, value: str, status: str = "online") -> str:
    """Return HTML for a small metric pill with a status dot."""
    return (
        f'<span class="metric-pill">'
        f'<span class="status-dot status-{status}"></span>{label}: <b>{value}</b>'
        f'</span>'
    )


def render_navbar(camera_status: str, fps: float, cpu: float, current_time: str) -> None:
    status = "online" if camera_status == "LIVE" else "offline"
    html = (
        '<div class="glass-card" style="display:flex;justify-content:space-between;'
        'align-items:center;flex-wrap:wrap;gap:10px;">'
        '<div><span class="gradient-text" style="font-size:1.3rem;">LipSync</span> '
        '<span style="color:var(--text-secondary);font-size:0.85rem;"> &nbsp;Real-Time Lip Reading '
        '&amp; Hand Gesture Translation</span></div>'
        '<div>'
        + status_pill("Camera", camera_status, status)
        + status_pill("FPS", f"{fps:.1f}")
        + status_pill("CPU", f"{cpu:.0f}%")
        + status_pill("Time", current_time)
        + "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    st.markdown(
        '<div class="brand-block">'
        '<div class="brand-icon">🎥</div>'
        '<div><div class="brand-name">LipSync</div>'
        '<div class="brand-sub">Industrial Communication AI</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    st.markdown(
        '<div class="sidebar-footer">'
        'Version 1.0.0<br>'
        'Developed by <b>Karthik</b>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_page_footer() -> None:
    st.markdown(
        '<div class="page-footer">LipSync &copy; 2026 &middot; '
        'Developed by <b>Karthik</b> &middot; '
        'Built for industrial communication accessibility</div>',
        unsafe_allow_html=True,
    )


def render_subtitle_box(text: str, confidence: float) -> None:
    display_text = text if text else "…waiting for speech / gestures…"
    st.markdown(
        f'<div class="subtitle-box"><div style="font-size:1.6rem;">{display_text}</div>'
        f'<div style="margin-top:8px;color:var(--text-secondary);font-size:0.85rem;">'
        f"Confidence: {confidence*100:.0f}%</div></div>",
        unsafe_allow_html=True,
    )


def render_prediction_cards(gesture: str, gesture_conf: float,
                             lip_word: str, lip_conf: float) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="glass-card">'
            f'<div class="card-label">Hand Gesture</div>'
            f'<div class="card-value">{gesture or "—"}</div>'
            f'<div class="card-sub">Confidence: {gesture_conf*100:.0f}%</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="glass-card">'
            f'<div class="card-label">Lip Reading</div>'
            f'<div class="card-value">{lip_word or "—"}</div>'
            f'<div class="card-sub">Confidence: {lip_conf*100:.0f}%</div></div>',
            unsafe_allow_html=True,
        )


def render_history_table(records: List[HistoryRecord]) -> None:
    if not records:
        st.info("No history yet. Predictions will appear here once detection starts.")
        return
    for r in records:
        st.markdown(
            f'<div class="history-row">🕒 {r.timestamp} &nbsp;|&nbsp; '
            f'<b>{r.sentence}</b> &nbsp;|&nbsp; gesture={r.gesture}, lip={r.lip_word} '
            f"&nbsp;|&nbsp; conf={r.confidence:.2f}</div>",
            unsafe_allow_html=True,
        )
