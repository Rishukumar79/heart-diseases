# 🫀 Heart Disease Prediction — End-to-End MLOps Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![DVC](https://img.shields.io/badge/DVC-Data_Versioned-purple)
![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black)

> A production-ready, end-to-end MLOps pipeline for heart disease risk prediction.
> Built with industry-standard tools: MLflow, DVC, FastAPI, Streamlit, Docker, and GitHub Actions CI/CD.

---

## 📋 Problem Statement

Heart disease is the leading cause of death globally. Manual diagnosis is time-consuming and error-prone. This system provides clinicians with a fast, data-driven second opinion using patient vitals — predicting whether a patient is at risk of heart disease.

**Input:** 13 clinical features (age, cholesterol, blood pressure, etc.)
**Output:** Binary prediction (0 = No Disease, 1 = Disease) + Probability score

---

## 🏗️ Architecture

```
Data Ingestion (heart.csv)
       │
       ▼
Preprocessing (StandardScaler, stratified split)
       │
       ▼
Feature Engineering (13 clinical features)
       │
       ▼
Model Training ──────────► MLflow Experiment Tracking
(RandomForest + XGBoost)          │
       │                    Model Registry
       ▼
  Evaluation
(AUC, F1, Accuracy)
       │
       ▼
GitHub Actions CI/CD
       │
       ▼
Docker Container Build
       │
       ▼
FastAPI Backend ◄────────► Streamlit UI
       │
       ▼
Monitoring + Logging
```

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.11 |
| ML Models | Scikit-learn (RandomForest), XGBoost |
| API Backend | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Testing | Pytest |

---

## 📁 Project Structure

```
heart-disease-mlops/
├── src/
│   ├── data/
│   │   ├── ingest.py          # Data loading + validation
│   │   └── preprocess.py      # Cleaning, scaling, splitting
│   └── models/
│       ├── train.py           # Training + MLflow tracking
│       └── predict.py         # Inference module
├── app/
│   ├── api.py                 # FastAPI REST endpoints
│   └── streamlit_app.py       # Streamlit UI
├── data/
│   ├── raw/heart.csv          # Source dataset
│   └── processed/             # Saved train/test splits
├── models/
│   ├── best_model.pkl         # Saved trained model
│   ├── scaler.pkl             # Saved StandardScaler
│   └── metrics.json           # Last run metrics
├── tests/
│   └── test_pipeline.py       # Unit tests (data + model + API)
├── .github/
│   └── workflows/ci_cd.yml    # GitHub Actions pipeline
├── notebooks/                 # EDA and experimentation
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                   # DVC pipeline stages
├── requirements.txt
└── main.py                    # One-command pipeline runner
```

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/heart-disease-mlops.git
cd heart-disease-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run complete ML pipeline (ingest + preprocess + train)
python main.py

# 5. Start FastAPI backend
uvicorn app.api:app --reload --port 8000

# 6. Start Streamlit UI (new terminal)
streamlit run app/streamlit_app.py

# 7. View MLflow experiments
mlflow ui
# Open: http://localhost:5000
```

### Option 2: Docker Compose (Recommended)

```bash
# Spin up all services (API + UI + MLflow)
docker-compose up --build

# Services available at:
# FastAPI:  http://localhost:8000
# Streamlit: http://localhost:8501
# MLflow UI: http://localhost:5000
```

---

## 📊 Dataset

**Source:** [Heart Disease UCI Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Rows:** 303 patients
- **Features:** 13 clinical features
- **Target:** Binary (0 = No Disease, 1 = Disease)
- **Class balance:** ~54% disease, ~46% no disease

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = Male, 0 = Female |
| chest_pain_type | Type of chest pain (0-3) |
| resting_bp | Resting blood pressure (mm Hg) |
| cholestoral | Serum cholesterol (mg/dl) |
| fasting_blood_sugar | >120 mg/dl (1=True) |
| restecg | Resting ECG results (0-2) |
| max_hr | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression by exercise |
| slope | Slope of ST segment |
| num_major_vessels | Major vessels colored by fluoroscopy (0-3) |
| thal | Thalassemia type (1-3) |

---

## 📈 Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| RandomForest | ~85% | ~0.86 | ~0.92 |
| XGBoost | ~86% | ~0.87 | ~0.93 |

---

## 🔌 API Reference

**Base URL:** `http://localhost:8000`

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Service health status |
| POST | `/predict` | Make prediction |
| GET | `/docs` | Swagger UI (interactive docs) |

### Example API Call

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "chest_pain_type": 3,
    "resting_bp": 145, "cholestoral": 233,
    "fasting_blood_sugar": 1, "restecg": 0,
    "max_hr": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 0, "num_major_vessels": 0, "thal": 1
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.8742,
  "risk": "High Risk",
  "message": "Prediction complete. Patient is at High Risk."
}
```

---

## ⚙️ MLOps Components

### DVC — Data Version Control
```bash
dvc init
dvc add data/raw/heart.csv
dvc repro          # Rerun pipeline stages
dvc dag            # View pipeline DAG
```

### MLflow — Experiment Tracking
```bash
mlflow ui          # View all experiments
# Open http://localhost:5000
# Compare models, params, metrics side by side
```

### GitHub Actions CI/CD
Every push to `main`:
1. Installs dependencies
2. Runs full pipeline (ingest → preprocess → train)
3. Runs pytest unit tests
4. Builds Docker image
5. Pushes to Docker Hub
6. Triggers Render deploy

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_pipeline.py::TestAPI -v

# With coverage report
pytest tests/ --cov=src --cov=app
```

---


## ☁️ Cloud Deployment (Render.com)

1. Create account at [render.com](https://render.com)
2. New Web Service → Connect GitHub repo
3. Build Command: `pip install -r requirements.txt && python main.py`
4. Start Command: `uvicorn app.api:app --host 0.0.0.0 --port $PORT`
5. Add env var: `PORT=8000`

---

## 📝 License

MIT License — Feel free to use for learning and portfolio purposes.

---


*⭐ If this helped you, give it a star on GitHub!*
