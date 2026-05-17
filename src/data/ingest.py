"""
Data Ingestion Module
Loads raw CSV data and performs initial validation.
"""

import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/heart.csv")


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load heart disease dataset from CSV."""
    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Data loaded. Shape: {df.shape}")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Basic data validation checks."""
    required_cols = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholestoral",
        "fasting_blood_sugar", "restecg", "max_hr", "exang",
        "oldpeak", "slope", "num_major_vessels", "thal", "target"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    if df.isnull().sum().sum() > 0:
        logger.warning("Null values found. Will handle in preprocessing.")
    logger.info("Data validation passed.")
    return True


if __name__ == "__main__":
    df = load_data()
    validate_data(df)
    print(df.head())
    print(df["target"].value_counts())
