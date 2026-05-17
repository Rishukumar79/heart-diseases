"""
Streamlit Frontend UI
Run: streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests
import json
import os
import sys

# Allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="centered"
)

# ── Header ──────────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")
st.markdown(
    "Enter patient vitals below. The ML model will predict heart disease risk "
    "using a trained **RandomForest** classifier."
)
st.divider()

# ── Input Form ──────────────────────────────────────────
with st.form("patient_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", 20, 80, 50)
        sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {0: "Typical angina", 1: "Atypical angina",
                                    2: "Non-anginal pain", 3: "Asymptomatic"}[x]
        )
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholestoral = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        fasting_blood_sugar = st.radio(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality", 2: "LVH"}[x]
        )

    with col2:
        max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.radio(
            "Exercise-induced Angina",
            options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
        )
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x]
        )
        num_major_vessels = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Fixed defect", 3: "Reversable defect"}[x]
        )

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

# ── Prediction ───────────────────────────────────────────
if submitted:
    payload = {
        "age": age, "sex": sex, "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp, "cholestoral": cholestoral,
        "fasting_blood_sugar": fasting_blood_sugar, "restecg": restecg,
        "max_hr": max_hr, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "num_major_vessels": num_major_vessels, "thal": thal
    }

    with st.spinner("Running prediction..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.divider()
                risk = result["risk"]
                prob = result["probability"]

                if result["prediction"] == 1:
                    st.error(f"⚠️ **{risk}** — Probability: `{prob:.1%}`")
                    st.warning("Please consult a cardiologist for further evaluation.")
                else:
                    st.success(f"✅ **{risk}** — Probability: `{prob:.1%}`")
                    st.info("No significant heart disease risk detected.")

                # Probability gauge
                st.progress(prob)
                st.caption(f"Confidence score: {prob:.4f}")

                # Show raw response in expander
                with st.expander("Raw API response"):
                    st.json(result)
            else:
                st.error(f"API error: {response.status_code} — {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is the FastAPI server running?")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ── Footer ──────────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ | MLOps Portfolio Project | Heart Disease Prediction v1.0")
