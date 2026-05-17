"""
Model Training Module with MLflow Experiment Tracking
Trains RandomForest and XGBoost, logs everything to MLflow.
"""

import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import logging
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# XGBoost optional — gracefully handle if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")
MLFLOW_TRACKING_URI = "mlruns"  # local folder — change to remote URI in production


def evaluate(model, X_test, y_test) -> dict:
    """Calculate all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    logger.info(f"Metrics: {metrics}")
    logger.info("\n" + classification_report(y_test, y_pred))
    return metrics


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train RandomForest with MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("heart-disease-prediction")

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_split": 4,
        "random_state": 42,
        "class_weight": "balanced",
    }

    with mlflow.start_run(run_name="RandomForest"):
        logger.info("Training RandomForest...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        # Log params and metrics to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "heart.csv")

        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

    # Save model locally
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "best_model.pkl"))
    _save_metrics(metrics, "RandomForest")
    return model, metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with MLflow tracking."""
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not installed. Skipping.")
        return None, {}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("heart-disease-prediction")

    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    with mlflow.start_run(run_name="XGBoost"):
        logger.info("Training XGBoost...")
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        metrics = evaluate(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "xgboost_model")
        mlflow.set_tag("model_type", "XGBoost")

    return model, metrics


def _save_metrics(metrics: dict, model_name: str):
    """Save metrics as JSON for monitoring."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics["model_name"] = model_name
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to models/metrics.json")


if __name__ == "__main__":
    from src.data.preprocess import load_processed
    X_train, X_test, y_train, y_test = load_processed()
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    print("Best RF Metrics:", rf_metrics)
