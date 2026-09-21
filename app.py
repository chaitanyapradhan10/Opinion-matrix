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


# ----------------------------------------------------------------------
# Model files setup
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

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
        "A Transformer-based sentiment analysis model trained on labeled "
        "customer reviews to predict **Positive**, **Negative**, or "
        "**Neutral** sentiment."
    )

    st.markdown(
        "The model uses DistilBERT to analyze customer reviews and "
        "generate sentiment probabilities."
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
                    <span class="history-text">
                        {item['text']}
                    </span>

                    <span style="color:{color}; font-weight:600;">
                        {item['sentiment']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Clear history", width="stretch"):

            st.session_state["history"] = []
            st.rerun()

    else:

        st.caption(
            "Your last few predictions will show up here."
        )


# ----------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------
render_header()


# ----------------------------------------------------------------------
# Check model
# ----------------------------------------------------------------------
if not model_loaded:

    st.error(
        "Model files not found. Please make sure the trained Transformer "
        "model exists inside models/transformer/."
    )

    st.stop()


# ----------------------------------------------------------------------
# Input section
# ----------------------------------------------------------------------
st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

text_input = st.text_area(
    "Enter a customer review",
    height=140,
    placeholder=(
        "e.g. The product quality is amazing "
        "and delivery was super fast!"
    ),
    value=st.session_state["sample_text"],
    label_visibility="collapsed",
)


col1, col2 = st.columns([1, 3])


with col1:

    predict_clicked = st.button(
        "🔍  Predict Sentiment",
        type="primary",
        width="stretch",
    )


with col2:

    st.caption(
        f"{len(text_input)} characters"
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if predict_clicked:

    if not text_input.strip():

        st.warning(
            "Please enter a review to analyze."
        )

    else:

        with st.spinner("Analyzing sentiment..."):

            result = predictor.predict(text_input)


        # --------------------------------------------------------------
        # Get prediction results
        # --------------------------------------------------------------
        sentiment = result["sentiment"]

        confidence = result["confidence"]

        probs = result["probabilities"]


        # --------------------------------------------------------------
        # Normalize probability keys
        #
        # The Transformer predictor returns:
        #
        # {
        #     "positive": ...,
        #     "neutral": ...,
        #     "negative": ...
        # }
        #
        # We make sure all three labels are available.
        # --------------------------------------------------------------
        probabilities = {
            "positive": float(
                probs.get("positive", 0)
            ),
            "neutral": float(
                probs.get("neutral", 0)
            ),
            "negative": float(
                probs.get("negative", 0)
            ),
        }


        # --------------------------------------------------------------
        # Save prediction to MySQL
        # --------------------------------------------------------------
        save_prediction(
            text_input,
            sentiment,
            probabilities,
        )


        # --------------------------------------------------------------
        # Update history
        # --------------------------------------------------------------
        st.session_state["history"].append(
            {
                "text": text_input.strip()[:60],
                "sentiment": sentiment,
                "confidence": confidence,
            }
        )


        st.session_state["sample_text"] = ""


        # --------------------------------------------------------------
        # Result card
        # --------------------------------------------------------------
        render_result_card(
            sentiment,
            confidence,
        )


        # --------------------------------------------------------------
        # Prediction breakdown
        # --------------------------------------------------------------
        st.markdown(
            "#### Prediction breakdown"
        )


        # Create DataFrame
        df = pd.DataFrame(
            {
                "Label": [
                    "positive",
                    "neutral",
                    "negative",
                ],
                "Probability": [
                    probabilities["positive"],
                    probabilities["neutral"],
                    probabilities["negative"],
                ],
            }
        )


        # Sort so highest probability appears appropriately
        df = df.sort_values(
            "Probability",
            ascending=True,
        )


        # --------------------------------------------------------------
        # Color scale
        #
        # IMPORTANT:
        # The model uses lowercase labels.
        # --------------------------------------------------------------
        color_scale = alt.Scale(
            domain=[
                "positive",
                "negative",
                "neutral",
            ],
            range=[
                "#34d399",
                "#f87171",
                "#fbbf24",
            ],
        )


        # --------------------------------------------------------------
        # Altair chart
        # --------------------------------------------------------------
        chart = (
            alt.Chart(df)
            .mark_bar(
                cornerRadiusTopRight=8,
                cornerRadiusBottomRight=8,
                height=26,
            )
            .encode(

                x=alt.X(
                    "Probability:Q",
                    title="Probability",
                    axis=alt.Axis(
                        format="%",
                        tickCount=11,
                    ),
                    scale=alt.Scale(
                        domain=[0, 1]
                    ),
                ),

                y=alt.Y(
                    "Label:N",
                    sort="-x",
                    title=None,
                ),

                color=alt.Color(
                    "Label:N",
                    scale=color_scale,
                    legend=None,
                ),

                tooltip=[
                    alt.Tooltip(
                        "Label:N",
                        title="Sentiment",
                    ),
                    alt.Tooltip(
                        "Probability:Q",
                        title="Probability",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                height=150
            )
        )


        st.altair_chart(
            chart,
            width="stretch",
        )