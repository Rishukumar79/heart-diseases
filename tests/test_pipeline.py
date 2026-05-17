"""
Unit Tests for Heart Disease MLOps Project
Run: pytest tests/ -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Data Tests ────────────────────────────────────────────
class TestDataIngestion:
    def test_data_loads_successfully(self):
        from src.data.ingest import load_data
        df = load_data()
        assert df is not None
        assert df.shape[0] > 0

    def test_data_has_correct_columns(self):
        from src.data.ingest import load_data, validate_data
        df = load_data()
        assert validate_data(df) is True

    def test_target_is_binary(self):
        from src.data.ingest import load_data
        df = load_data()
        assert set(df["target"].unique()).issubset({0, 1})

    def test_no_required_nulls(self):
        from src.data.ingest import load_data
        df = load_data()
        # Key columns must not be null
        assert df["age"].isnull().sum() == 0
        assert df["target"].isnull().sum() == 0


# ── Preprocessing Tests ───────────────────────────────────
class TestPreprocessing:
    def setup_method(self):
        from src.data.ingest import load_data
        from src.data.preprocess import preprocess
        df = load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess(df)

    def test_split_shapes(self):
        assert self.X_train.shape[1] == 13  # 13 features
        assert len(self.y_train) == self.X_train.shape[0]

    def test_classes_in_split(self):
        """Both classes should exist after stratified split."""
        assert 0 in self.y_train and 1 in self.y_train
        assert 0 in self.y_test and 1 in self.y_test

    def test_features_scaled(self):
        """After StandardScaler, mean ~0, std ~1."""
        mean = np.abs(self.X_train.mean(axis=0))
        assert np.all(mean < 1.5), "Features not properly scaled"

    def test_no_nan_after_preprocessing(self):
        assert not np.isnan(self.X_train).any()
        assert not np.isnan(self.X_test).any()


# ── Prediction Tests ──────────────────────────────────────
class TestPrediction:
    SAMPLE_PATIENT = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]

    def test_predict_returns_dict(self):
        from src.models.predict import predict
        result = predict(self.SAMPLE_PATIENT)
        assert isinstance(result, dict)

    def test_predict_keys(self):
        from src.models.predict import predict
        result = predict(self.SAMPLE_PATIENT)
        assert "prediction" in result
        assert "probability" in result
        assert "risk" in result

    def test_prediction_is_binary(self):
        from src.models.predict import predict
        result = predict(self.SAMPLE_PATIENT)
        assert result["prediction"] in [0, 1]

    def test_probability_in_range(self):
        from src.models.predict import predict
        result = predict(self.SAMPLE_PATIENT)
        assert 0.0 <= result["probability"] <= 1.0

    def test_risk_label(self):
        from src.models.predict import predict
        result = predict(self.SAMPLE_PATIENT)
        assert result["risk"] in ["High Risk", "Low Risk"]


# ── API Tests ──────────────────────────────────────────────
class TestAPI:
    """FastAPI endpoint tests using TestClient."""

    def test_root_endpoint(self):
        from fastapi.testclient import TestClient
        from app.api import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from app.api import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_predict_endpoint_valid_input(self):
        from fastapi.testclient import TestClient
        from app.api import app
        client = TestClient(app)
        payload = {
            "age": 63, "sex": 1, "chest_pain_type": 3, "resting_bp": 145,
            "cholestoral": 233, "fasting_blood_sugar": 1, "restecg": 0,
            "max_hr": 150, "exang": 0, "oldpeak": 2.3,
            "slope": 0, "num_major_vessels": 0, "thal": 1
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data

    def test_predict_endpoint_invalid_input(self):
        from fastapi.testclient import TestClient
        from app.api import app
        client = TestClient(app)
        response = client.post("/predict", json={"age": "not_a_number"})
        assert response.status_code == 422  # Unprocessable entity
