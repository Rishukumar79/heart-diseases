import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "..", "models", "best_model.pkl")
    scaler_path = os.path.join(base, "..", "models", "scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

st.title("🫀 Heart Disease Risk Predictor")
st.divider()

with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 80, 50)
        sex = st.radio("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        chest_pain_type = st.selectbox("Chest Pain Type", [0,1,2,3],
            format_func=lambda x: {0:"Typical angina",1:"Atypical angina",
                                    2:"Non-anginal",3:"Asymptomatic"}[x])
        resting_bp = st.slider("Resting BP", 80, 200, 120)
        cholestoral = st.slider("Cholesterol", 100, 600, 240)
        fasting_blood_sugar = st.radio("Fasting Sugar > 120", [0,1],
            format_func=lambda x: "No" if x==0 else "Yes")
        restecg = st.selectbox("Resting ECG", [0,1,2],
            format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"LVH"}[x])
    with col2:
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.radio("Exercise Angina", [0,1],
            format_func=lambda x: "No" if x==0 else "Yes")
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("ST Slope", [0,1,2],
            format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])
        num_major_vessels = st.selectbox("Major Vessels", [0,1,2,3])
        thal = st.selectbox("Thalassemia", [1,2,3],
            format_func=lambda x: {1:"Normal",2:"Fixed defect",3:"Reversable"}[x])

    submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

if submitted:
    model, scaler = load_model()
    features = np.array([[age, sex, chest_pain_type, resting_bp,
                           cholestoral, fasting_blood_sugar, restecg,
                           max_hr, exang, oldpeak, slope,
                           num_major_vessels, thal]])
    pred = int(model.predict(scaler.transform(features))[0])
    prob = float(model.predict_proba(scaler.transform(features))[0][1])
    st.divider()
    if pred == 1:
        st.error(f"⚠️ High Risk — Probability: {prob:.1%}")
        st.warning("Please consult a cardiologist.")
    else:
        st.success(f"✅ Low Risk — Probability: {prob:.1%}")
        st.info("No significant risk detected.")
    st.progress(prob)

st.divider()
st.caption("MLOps Portfolio Project | Heart Disease Prediction v1.0")
