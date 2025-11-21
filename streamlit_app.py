import streamlit as st
import requests

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"   # change when deployed

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🤖", layout="centered")

st.title("🧪 Toxic Comment Classification")
st.write("Enter a comment below to test your FastAPI model.")

# ---------------------------------------------
# USER INPUT
# ---------------------------------------------
user_input = st.text_area("✏️ Enter your comment:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a comment first.")
    else:
        with st.spinner("Contacting API..."):
            try:
                payload = {"comment": user_input}
                response = requests.post(API_URL, json=payload)

                if response.status_code != 200:
                    st.error(f"API Error: {response.text}")
                else:
                    result = response.json()

                    # -----------------------------
                    # DISPLAY RESULTS
                    # -----------------------------
                    st.subheader("🟦 Overall Prediction")
                    if result["is_toxic"]:
                        st.error("⚠️ This comment is toxic")
                    else:
                        st.success("✅ This comment is not toxic")

                    st.write("---")
                    st.subheader("📌 Explanation")
                    st.write(result["explanation"])

                    st.write("---")
                    st.subheader("📊 Label-wise Predictions")

                    for pred in result["predictions"]:
                        label = pred["label"].replace("_", " ").title()
                        prob = pred["probability"]
                        is_pos = pred["is_positive"]

                        if is_pos:
                            st.markdown(f"🔴 **{label}** — {prob}")
                        else:
                            st.markdown(f"🟢 {label} — {prob}")

            except Exception as e:
                st.error(f"Request failed: {e}")

st.write("---")
st.caption("Built with Streamlit + FastAPI 🚀")
