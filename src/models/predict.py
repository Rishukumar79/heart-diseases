"""
Prediction / Inference Module
Loads saved model + scaler and returns predictions with probability.
"""

import joblib
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")


def load_model():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    return joblib.load(model_path)


def load_scaler():
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run preprocessing first.")
    return joblib.load(scaler_path)


def predict(features: list) -> dict:
    """
    Make prediction from raw feature list.
    features: list of 13 values in order:
    [age, sex, chest_pain_type, resting_bp, cholestoral,
     fasting_blood_sugar, restecg, max_hr, exang,
     oldpeak, slope, num_major_vessels, thal]
    Returns: {"prediction": 0/1, "probability": float, "risk": str}
    """
    model = load_model()
    scaler = load_scaler()

    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred = int(model.predict(x_scaled)[0])
    prob = float(model.predict_proba(x_scaled)[0][1])

    risk = "High Risk" if pred == 1 else "Low Risk"
    logger.info(f"Prediction: {pred}, Probability: {prob:.4f}, Risk: {risk}")

    return {
        "prediction": pred,
        "probability": round(prob, 4),
        "risk": risk
    }
