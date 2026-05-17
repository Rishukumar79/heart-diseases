import numpy as np
import joblib
import logging
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

def train_random_forest(X_train, X_test, y_train, y_test):
    params = {"n_estimators": 200, "max_depth": 8,
              "min_samples_split": 4, "random_state": 42, "class_weight": "balanced"}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics: {metrics}")
    return model, metrics

def train_xgboost(X_train, X_test, y_train, y_test):
    return None, {}
