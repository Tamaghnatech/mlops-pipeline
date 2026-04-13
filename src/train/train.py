import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import yaml
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load params
with open("config/params.yaml") as f:
    params = yaml.safe_load(f)

# MLflow setup
mlflow.set_tracking_uri("https://dagshub.com/Tamaghnatech/mlops-pipeline.mlflow")
mlflow.set_experiment("churn-prediction")

# Load data
df = pd.read_csv("data/raw/churn.csv")

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop(columns=["customerID"], inplace=True)

# Encode categoricals
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["data"]["test_size"],
    random_state=params["data"]["random_state"]
)

# Train with MLflow tracking
with mlflow.start_run():
    n_estimators = params["model"]["n_estimators"]
    max_depth    = params["model"]["max_depth"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=params["data"]["random_state"]
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    # Save metrics locally
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1": f1, "roc_auc": roc_auc}, f, indent=2)

print("Training complete!")