
import joblib
import streamlit as st
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.pkl"

st.set_page_config(
    page_title="SpamShield | AI Spam Detector",
    page_icon="🛡️",
    layout="centered"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🛡️ SpamShield")
st.subheader("AI-Powered Spam Message Detector")

st.write(
    "Check suspicious SMS messages and email text "
    "using a machine learning model."
)

st.info(
    "Enter a message below, or select an example "
    "to test the classifier."
)

samples = {
    "Choose an example": "",
    "Potential spam": (
        "Congratulations! You won a free prize. "
        "Click here to claim now!"
    ),
    "Normal message": (
        "Hi, are we meeting at college tomorrow "
        "at 10 AM?"
    ),
    "Promotional offer": (
        "Limited offer! Claim your cash reward now."
    )
}

selected = st.selectbox(
    "Try a sample message",
    list(samples.keys())
)

message = st.text_area(
    "Your message",
    value=samples[selected],
    placeholder="Type or paste your message here...",
    height=150
)

if st.button("🔍 Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = int(model.predict([message])[0])

        label = "Spam" if prediction == 1 else "Not Spam"

        st.session_state.history.insert(
            0,
            {
                "Message": message,
                "Prediction": label
            }
        )

        if prediction == 1:
            st.error("🚨 Prediction: Spam")
            st.write(
                "The model identifies patterns associated "
                "with spam messages."
            )
        else:
            st.success("✅ Prediction: Not Spam")
            st.write(
                "The model identifies patterns associated "
                "with legitimate messages."
            )

        st.caption(
            "This is a model prediction, not a guarantee "
            "that the message is safe or harmful."
        )

st.divider()

st.subheader("📊 Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("Your predictions will appear here.")

st.divider()

st.caption(
    "SpamShield | Python • Scikit-learn • NLP • Streamlit"
)
