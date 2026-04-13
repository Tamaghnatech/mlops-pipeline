import pickle
import mlflow
import mlflow.sklearn
import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# App setup
app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn probability",
    version="1.0.0"
)

# Load model on startup
model = None
encoders = {}

def load_model():
    global model
    model_path = Path("models/model.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("Model loaded from local file")
    else:
        raise RuntimeError("No model found. Run training first.")

@app.on_event("startup")
def startup_event():
    load_model()

# Request schema — matches our dataset columns
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

# Routes
@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([customer.dict()])
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= 0.5)
        return {
            "churn_probability": round(float(proba), 4),
            "will_churn": bool(prediction),
            "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_,
        "classes": model.classes_.tolist()
    }