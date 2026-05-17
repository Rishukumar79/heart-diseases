"""
Preprocessing Module
Handles cleaning, encoding, scaling, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")

FEATURE_COLS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholestoral",
    "fasting_blood_sugar", "restecg", "max_hr", "exang",
    "oldpeak", "slope", "num_major_vessels", "thal"
]
TARGET_COL = "target"


def preprocess(df: pd.DataFrame, fit_scaler: bool = True):
    """
    Full preprocessing pipeline:
    1. Drop nulls
    2. Separate features and target
    3. Train/test split (stratified)
    4. StandardScaler fit/transform
    Returns: X_train, X_test, y_train, y_test
    """
    logger.info("Starting preprocessing...")

    # Drop nulls if any
    df = df.dropna().reset_index(drop=True)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Stratified split — class balance maintain hota hai
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Scaling
    scaler = StandardScaler()
    if fit_scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Save scaler for inference
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        logger.info("Scaler saved.")
    else:
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Save processed splits
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    logger.info("Processed data saved.")

    return X_train, X_test, y_train, y_test


def load_processed():
    """Load saved processed splits."""
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from ingest import load_data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    print("Classes in train:", np.unique(y_train, return_counts=True))
