"""
app.py
-------
Phase 11: Deployment.
A Streamlit web application for the Sentiment Analysis project.

Run:
    streamlit run app.py
"""
from db import save_prediction
import streamlit as st
import sys
import os
import pandas as pd
import altair as alt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict_transformer import TransformerSentimentPredictor as SentimentPredictor
from styles import inject_css, render_header, render_result_card, sentiment_color
from download_models import download_models


@st.cache_resource
def setup():
    download_models()


setup()



# ----------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="OpinionMetrix AI Sentiment Transformer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)
inject_css()

if "history" not in st.session_state:
    st.session_state["history"] = []  # list of {"text", "sentiment", "confidence"}
if "sample_text" not in st.session_state:
    st.session_state["sample_text"] = ""


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return SentimentPredictor()


try:
    print("Loading model...")
    predictor = load_predictor()
    model_loaded = True
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ℹ️ About this model")
    st.write(
        "A TF-IDF + tuned scikit-learn classifier (or transformer model, "
        "depending on build) trained on labeled customer reviews to predict "
        "**Positive**, **Negative**, or **Neutral** sentiment."
    )
    st.markdown(
        "Text is cleaned (lowercased, HTML/URL/emoji/number/punctuation "
        "stripped), tokenized, stopword-filtered, and lemmatized before "
        "being vectorized and scored."
    )

    st.divider()
    st.markdown("### 🧪 Try a sample review")
    samples = [
        "Absolutely love this product, best purchase I've made all year!",
        "Terrible quality, broke within a week, avoid this seller.",
        "It's fine, does what it says but nothing more.",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{s}"):
            st.session_state["sample_text"] = s
            st.rerun()

    st.divider()
    st.markdown("### 🕘 Recent predictions")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"][-5:]):
            color = sentiment_color(item["sentiment"])
            st.markdown(
                f"""
                <div class="history-row">
                    <span class="history-text">{item['text']}</span>
                    <span style="color:{color}; font-weight:600;">{item['sentiment']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("Clear history", width='stretch'):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.caption("Your last few predictions will show up here.")


# ----------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------
render_header()

if not model_loaded:
    st.error(
        "Model files not found. Please run `python src/train_model.py` first "
        "to generate the trained model before launching this app."
    )
    st.stop()

st.markdown('<div class="input-card">', unsafe_allow_html=True)
text_input = st.text_area(
    "Enter a customer review",
    height=140,
    placeholder="e.g. The product quality is amazing and delivery was super fast!",
    value=st.session_state["sample_text"],
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("🔍  Predict Sentiment", type="primary", width='stretch')
with col2:
    st.caption(f"{len(text_input)} characters")
st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = predictor.predict(text_input)

        sentiment = result["sentiment"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        save_prediction(text_input, sentiment, probs)

        st.session_state["history"].append(
            {"text": text_input.strip()[:60], "sentiment": sentiment, "confidence": confidence}
        )
        st.session_state["sample_text"] = ""

        render_result_card(sentiment, confidence)

        st.markdown("#### Prediction breakdown")
        df = pd.DataFrame(
            {
                "Label": list(probs.keys()),
                "Probability": list(probs.values()),
            }
        ).sort_values("Probability", ascending=True)

        color_scale = alt.Scale(
            domain=["Positive", "Negative", "Neutral"],
            range=["#34d399", "#f87171", "#fbbf24"],
        )

        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, height=26)
            .encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Label:N", sort="-x", title=None),
                color=alt.Color("Label:N", scale=color_scale, legend=None),
                tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=120)
        )
        st.altair_chart(chart, width='stretch')
