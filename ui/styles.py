"""
ui/styles.py
Custom CSS injected into the Streamlit app. Professional dark UI with a
restrained accent palette, consistent spacing, and a proper product-style
header / sidebar treatment.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --bg-primary: #0a0c11;
    --bg-secondary: #10131a;
    --surface: #12151d;
    --surface-hover: #171b25;
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.14);
    --accent-1: #6366f1;
    --accent-2: #06b6d4;
    --accent-gradient: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
    --text-primary: #f1f2f4;
    --text-secondary: #8b92a4;
    --text-muted: #5b6172;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.28);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

code, .stCode, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background:
        radial-gradient(circle at 15% -10%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
        var(--bg-primary);
    color: var(--text-primary);
}

/* ---------------------------------------------------------------- */
/* Sidebar                                                           */
/* ---------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

.brand-block {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 4px 20px 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}
.brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: var(--accent-gradient);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
}
.brand-name {
    font-size: 1.08rem;
    font-weight: 800;
    line-height: 1.15;
    color: var(--text-primary);
}
.brand-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.sidebar-footer {
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.6;
}
.sidebar-footer b {
    color: var(--text-secondary);
}

/* Radio nav styled as a menu list */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 8px 10px;
    border-radius: var(--radius-sm);
    margin-bottom: 2px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: var(--surface-hover);
}

/* ---------------------------------------------------------------- */
/* Cards / surfaces                                                  */
/* ---------------------------------------------------------------- */
.glass-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    box-shadow: var(--shadow-card);
    margin-bottom: 16px;
}

.gradient-text {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-right: 8px;
    font-weight: 500;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-online { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-offline { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
.status-warning { background: var(--warning); box-shadow: 0 0 6px var(--warning); }

.subtitle-box {
    background: linear-gradient(135deg, rgba(99, 91, 241, 0.12), rgba(6, 182, 212, 0.08));
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    padding: 28px;
    text-align: center;
    font-weight: 700;
    letter-spacing: 0.2px;
    animation: fadeInUp 0.35s ease;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.history-row {
    border-bottom: 1px solid var(--border);
    padding: 10px 2px;
    font-size: 0.87rem;
    color: var(--text-secondary);
}
.history-row b { color: var(--text-primary); }

.section-header {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}

.section-sub {
    color: var(--text-secondary);
    margin-bottom: 20px;
    font-size: 0.92rem;
}

.card-label {
    color: var(--text-muted);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.card-value {
    font-size: 1.5rem;
    font-weight: 800;
    margin: 4px 0 2px 0;
}

.card-sub {
    color: var(--text-secondary);
    font-size: 0.8rem;
}

.page-footer {
    text-align: center;
    color: var(--text-muted);
    padding: 28px 0 10px 0;
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 24px;
}
.page-footer b { color: var(--text-secondary); }

hr {
    border-color: var(--border) !important;
}

/* ---------------------------------------------------------------- */
/* Buttons / inputs                                                  */
/* ---------------------------------------------------------------- */
.stButton>button {
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-strong);
    background: var(--surface);
    color: var(--text-primary);
    font-weight: 600;
    transition: all 0.15s ease;
}
.stButton>button:hover {
    border-color: var(--accent-2);
    color: var(--accent-2);
    background: var(--surface-hover);
}
.stButton>button:active {
    transform: scale(0.98);
}

div[data-testid="stDownloadButton"] button {
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-strong);
    font-weight: 600;
}

.stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {
    color: var(--text-secondary) !important;
    font-weight: 500;
}

/* Tighten default Streamlit block spacing for a denser, app-like feel */
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}
</style>
"""
