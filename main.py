"""
Main Pipeline Runner
Runs the complete ML pipeline: ingest → preprocess → train
"""

import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))


def main():
    logger.info("=" * 50)
    logger.info("🫀 HEART DISEASE MLOps PIPELINE STARTING")
    logger.info("=" * 50)

    # Step 1: Data Ingestion
    logger.info("Step 1/3: Data Ingestion")
    from src.data.ingest import load_data, validate_data
    df = load_data()
    if not validate_data(df):
        logger.error("Data validation failed. Exiting.")
        sys.exit(1)

    # Step 2: Preprocessing
    logger.info("Step 2/3: Preprocessing")
    from src.data.preprocess import preprocess
    X_train, X_test, y_train, y_test = preprocess(df, fit_scaler=True)

    # Step 3: Training
    logger.info("Step 3/3: Model Training")
    from src.models.train import train_random_forest, train_xgboost
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)

    logger.info("=" * 50)
    logger.info("PIPELINE COMPLETE!")
    logger.info(f"RandomForest — AUC: {rf_metrics.get('roc_auc')}, F1: {rf_metrics.get('f1_score')}")
    if xgb_metrics:
        logger.info(f"XGBoost — AUC: {xgb_metrics.get('roc_auc')}, F1: {xgb_metrics.get('f1_score')}")
    logger.info("Model saved to models/best_model.pkl")
    logger.info("MLflow experiments tracked in mlruns/")
    logger.info("=" * 50)
    logger.info("Run API:        uvicorn app.api:app --reload")
    logger.info("Run Streamlit:  streamlit run app/streamlit_app.py")
    logger.info("Run MLflow UI:  mlflow ui")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
