"""
FastAPI Backend
Exposes /predict endpoint for heart disease prediction.
Run: uvicorn app.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps project — predicts heart disease from patient vitals.",
    version="1.0.0",
)


class PatientData(BaseModel):
    age: float = Field(..., example=63, description="Patient age in years")
    sex: int = Field(..., example=1, description="1=Male, 0=Female")
    chest_pain_type: int = Field(..., example=3, description="Chest pain type (0-3)")
    resting_bp: float = Field(..., example=145, description="Resting blood pressure (mm Hg)")
    cholestoral: float = Field(..., example=233, description="Serum cholesterol (mg/dl)")
    fasting_blood_sugar: int = Field(..., example=1, description="1 if >120 mg/dl, else 0")
    restecg: int = Field(..., example=0, description="Resting ECG results (0-2)")
    max_hr: float = Field(..., example=150, description="Maximum heart rate achieved")
    exang: int = Field(..., example=0, description="Exercise-induced angina (1=Yes, 0=No)")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise")
    slope: int = Field(..., example=0, description="Slope of peak exercise ST segment")
    num_major_vessels: int = Field(..., example=0, description="Number of major vessels (0-3)")
    thal: int = Field(..., example=1, description="Thal: 1=Normal, 2=Fixed defect, 3=Reversable")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk: str
    message: str


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Heart Disease Prediction API is running 🫀"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def make_prediction(data: PatientData):
    """
    Predict heart disease risk from patient vitals.
    Returns prediction (0/1), probability, and risk level.
    """
    try:
        features = [
            data.age, data.sex, data.chest_pain_type, data.resting_bp,
            data.cholestoral, data.fasting_blood_sugar, data.restecg,
            data.max_hr, data.exang, data.oldpeak, data.slope,
            data.num_major_vessels, data.thal
        ]
        result = predict(features)
        return PredictionResponse(
            **result,
            message=f"Prediction complete. Patient is at {result['risk']}."
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
