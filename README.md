# Enterprise MLOps Pipeline

> **A production-grade, fully automated ML system built from scratch on Windows — data versioning, experiment tracking, orchestration, CI/CD, REST API, dashboards, drift monitoring, and Slack alerts. All free-tier hosted.**

[![CI/CD](https://github.com/Tamaghnatech/mlops-pipeline/actions/workflows/train.yml/badge.svg)](https://github.com/Tamaghnatech/mlops-pipeline/actions)
[![DagsHub](https://img.shields.io/badge/DagsHub-MLflow%20%7C%20DVC-orange)](https://dagshub.com/Tamaghnatech/mlops-pipeline)
[![API](https://img.shields.io/badge/API-Live%20on%20Render-6c63ff)](https://churn-prediction-api-ykim.onrender.com/docs)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit%20Cloud-ff4b4b)](https://mlops-pipeline-d3n72rexzfzdhs7jjbpaal.streamlit.app)

---

## Live URLs

| Service | URL |
|---------|-----|
| Prediction API (Swagger) | https://churn-prediction-api-ykim.onrender.com/docs |
| Prediction Dashboard | https://mlops-pipeline-d3n72rexzfzdhs7jjbpaal.streamlit.app |
| Monitoring Dashboard | https://mlops-pipeline-uyz7jrqinegdhqd9h2gru5.streamlit.app |
| MLflow Experiments | https://dagshub.com/Tamaghnatech/mlops-pipeline/experiments |
| DagsHub Repository | https://dagshub.com/Tamaghnatech/mlops-pipeline |

> **Note:** The Render API free tier spins down after 15 minutes of inactivity. First request takes ~60 seconds to wake up. Open the /docs URL before demoing.

---

## What This Project Does

This is a customer churn prediction system for a telecom company. Given a customer's account details (tenure, contract type, services, charges), the model predicts the probability they will cancel their subscription.

But more than the model — this project demonstrates how to build the **complete infrastructure** around a model: the data pipeline, the experiment tracking, the API, the monitoring, the automation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Data Layer (DVC + DagsHub)                 │
│   Raw CSV versioned like code — anyone can dvc pull     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│           Experiment Tracking (MLflow + DagsHub)        │
│   Every run logged: params, metrics, model artifact     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│           Orchestration (Prefect)                       │
│   Pipeline as a flow with tasks, retries, logging       │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│           CI/CD (GitHub Actions)                        │
│   Push code → pipeline auto-runs in the cloud           │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│           Serving (FastAPI on Render)                   │
│   POST /predict → churn probability + risk level        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│           UI (Streamlit Cloud)                          │
│   Single prediction + batch CSV scoring + charts        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│           Monitoring (Evidently AI + Slack)             │
│   Drift detected → Slack alert → auto-retrain           │
└─────────────────────────────────────────────────────────┘
```

---

## Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11.0 | Core language |
| scikit-learn | 1.4.2 | Random Forest model |
| MLflow | 2.12.1 | Experiment tracking |
| DVC | 3.49.0 | Data versioning |
| Prefect | 2.19.3 | Pipeline orchestration |
| FastAPI | 0.111.0 | REST API serving |
| Streamlit | 1.33.0 | Visual dashboards |
| Evidently AI | 0.4.16 | Drift monitoring |
| GitHub Actions | — | CI/CD automation |
| Render | — | API cloud hosting |
| DagsHub | — | MLflow + DVC hosting |

---

## Dataset

**Telco Customer Churn** (IBM Sample Dataset)
- 7,043 rows × 21 columns
- Target: Churn (Yes/No) — 26.5% positive rate
- Features: demographics, services subscribed, contract type, payment method, monthly charges

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 79.7% |
| F1 Score | 56.5% |
| ROC AUC | 82.7% |

The F1 score is lower than accuracy due to class imbalance (only 26% of customers churn). ROC AUC of 82.7% means the model has strong discriminative ability.

---

## Quick Start (Local)

```bash
# 1. Clone and setup
git clone https://github.com/Tamaghnatech/mlops-pipeline
cd mlops-pipeline
python -m venv venv
venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
pip install griffe==0.49.0 pathspec==0.12.1

# 2. Pull data from DagsHub
dvc pull

# 3. Set MLflow credentials
$env:MLFLOW_TRACKING_USERNAME = "your_dagshub_username"
$env:MLFLOW_TRACKING_PASSWORD = "your_dagshub_token"

# 4. Train the model
python src/train/train.py

# 5. Start the API
uvicorn src.serve.app:app --port 8001

# 6. Start the UI (new terminal)
streamlit run app_ui.py

# 7. Run drift monitor
$env:SLACK_WEBHOOK_URL = "your_slack_webhook"
python monitoring/drift_monitoring.py --simulate
```

---

## API Usage

```bash
# Predict churn for a single customer
curl -X POST "https://churn-prediction-api-ykim.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0
  }'
```

Response:
```json
{
  "churn_probability": 0.5973,
  "will_churn": true,
  "risk_level": "Medium"
}
```

---

## Project Structure

```
mlops-pipeline/
├── .github/workflows/
│   └── train.yml              # CI/CD pipeline
├── config/
│   └── params.yaml            # Hyperparameters (tracked by DVC)
├── data/
│   └── raw/churn.csv.dvc      # DVC pointer to dataset
├── models/
│   └── model.pkl              # Trained model
├── monitoring/
│   ├── drift_monitoring.py    # Evidently AI drift detection
│   └── dashboard.py           # Monitoring Streamlit dashboard
├── pipelines/
│   └── training_pipeline.py   # Prefect flow
├── reports/
│   ├── metrics.json           # Latest training metrics
│   └── drift/                 # Drift reports and history
├── src/
│   ├── train/train.py         # Training script with MLflow
│   └── serve/app.py           # FastAPI serving
├── app_ui.py                  # Prediction Streamlit dashboard
├── requirements.txt           # Full dependencies
├── requirements-serve.txt     # Slim serving dependencies
├── requirements-ui.txt        # Slim UI dependencies
├── render.yaml                # Render deployment config
└── .python-version            # Python 3.11.0
```

---

## CI/CD Flow

Every push to `master` that changes files in `src/`, `config/`, `pipelines/`, or `.github/workflows/`:

1. GitHub spins up a fresh Ubuntu machine
2. Installs Python 3.11 and all dependencies
3. Pulls the dataset from DagsHub via DVC
4. Runs the Prefect training pipeline
5. Logs metrics to MLflow on DagsHub
6. Green checkmark = new model trained and logged

---

## Drift Monitoring Loop

```
Production data collected
        ↓
Evidently AI compares distributions
        ↓
drift_share > 20% threshold?
        ↓ YES
Slack alert fired automatically
        ↓
Prefect retraining pipeline triggered
        ↓
New model trained and saved
        ↓
Monitoring dashboard updated (commit + push history JSON)
```

---

## Lessons Learned

1. **Pin every package version** — `pip install mlflow` today installs a different version than tomorrow. Pinning prevents unexpected breakage.

2. **Windows has unique gotchas** — PowerShell vs CMD, curl aliasing, CRLF line endings. Know your environment.

3. **Version conflicts are normal** — We hit 3 separate version conflicts (uvicorn, griffe, pathspec). The fix is always the same: read the error, find the incompatible constraint, downgrade one package.

4. **Separate serving requirements** — Your training container needs XGBoost, DVC, MLflow, Prefect. Your serving container only needs FastAPI, scikit-learn, and numpy. Keep them separate.

5. **F1 > Accuracy for imbalanced data** — 79.7% accuracy sounds good but a model predicting "never churn" would get 73% accuracy. ROC AUC and F1 are more meaningful.

6. **Free tier is surprisingly capable** — This entire system runs for free. Render, Streamlit Cloud, DagsHub, Prefect Cloud, GitHub Actions, Slack — all free.

---

## Documentation

See `MLOps_Pipeline_Complete_Documentation.docx` for the full build guide including:
- Every error encountered and its exact fix
- All design decisions and trade-offs
- Step-by-step recreation guide
- Architecture deep-dive

---

## Author

**Tamaghna Nag** — Machine Learning Engineer  
London, UK | [GitHub](https://github.com/Tamaghnatech) | [DagsHub](https://dagshub.com/Tamaghnatech)

---

*Built with persistence. Debugged with patience. The best way to learn MLOps is to break it and fix it.*