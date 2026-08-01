"""
styles.py
---------
Design system for the Sentiment Analyzer app: CSS injection and
small render helpers, kept separate from app.py so the UI logic
and the UI *look* can evolve independently.
"""
import streamlit as st

# ----------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------
SENTIMENT_META = {
    "Positive": {
        "emoji": "😊",
        "color": "#34d399",
        "bg": "linear-gradient(135deg, rgba(52,211,153,0.16), rgba(52,211,153,0.04))",
        "border": "#34d399",
    },
    "Negative": {
        "emoji": "😞",
        "color": "#f87171",
        "bg": "linear-gradient(135deg, rgba(248,113,113,0.16), rgba(248,113,113,0.04))",
        "border": "#f87171",
    },
    "Neutral": {
        "emoji": "😐",
        "color": "#fbbf24",
        "bg": "linear-gradient(135deg, rgba(251,191,36,0.16), rgba(251,191,36,0.04))",
        "border": "#fbbf24",
    },
}

CUSTOM_CSS = """
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2.2rem;
        max-width: 780px;
    }

    /* ---------- Header ---------- */
    .app-header {
        text-align: center;
        margin-bottom: 1.6rem;
    }
    .app-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        background: rgba(129,140,248,0.14);
        color: #a5b4fc;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
        border: 1px solid rgba(129,140,248,0.28);
    }
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.4rem;
    }

    /* ---------- Input card ---------- */
    .input-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.4rem 1.4rem 1rem 1.4rem;
        margin-bottom: 1.4rem;
    }
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        font-size: 0.98rem !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #6366f1, #38bdf8);
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(99,102,241,0.45);
    }

    /* ---------- Result card ---------- */
    .result-card {
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        margin-top: 0.6rem;
        border-left: 5px solid;
        animation: fadeIn 0.35s ease-in;
    }
    .result-card h2 {
        margin: 0 0 0.2rem 0;
        font-size: 1.5rem;
    }
    .result-confidence {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: 0.1rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Sample chips ---------- */
    div[data-testid="stExpander"] div.stButton > button {
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.03);
        font-size: 0.85rem;
        padding: 0.4rem 0.9rem;
        width: 100%;
        text-align: left;
    }
    div[data-testid="stExpander"] div.stButton > button:hover {
        border-color: #818cf8;
        color: #a5b4fc;
    }

    /* ---------- History ---------- */
    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.7rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 0.4rem;
        font-size: 0.88rem;
    }
    .history-text {
        color: #cbd5e1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 70%;
    }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <span class="app-badge">NLP · Machine Learning</span>
            <h1 class="app-title">💬 OpinionMetrix AI Sentiment Transformer</h1>
            <p class="app-subtitle">
                Paste any customer review and instantly see whether it reads as
                Positive, Negative, or Neutral — with confidence scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(sentiment: str, confidence: float) -> None:
    meta = SENTIMENT_META.get(
        sentiment, {"emoji": "🤔", "color": "#94a3b8", "bg": "rgba(148,163,184,0.08)", "border": "#94a3b8"}
    )
    st.markdown(
        f"""
        <div class="result-card" style="background:{meta['bg']}; border-color:{meta['border']};">
            <h2 style="color:{meta['color']};">{meta['emoji']} {sentiment}</h2>
            <p class="result-confidence">Confidence score: <b>{confidence:.1%}</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_color(label: str) -> str:
    return SENTIMENT_META.get(label, {}).get("color", "#94a3b8")
