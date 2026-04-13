"""
Evidently AI drift monitor.
Compares training data distribution vs new incoming data.
Generates HTML report + triggers retraining if drift detected.
"""
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports/drift")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))


def load_and_encode(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.drop(columns=["customerID"], errors="ignore", inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    # Drop target column for drift analysis
    ref = reference.drop(columns=["Churn"], errors="ignore")
    cur = current.drop(columns=["Churn"], errors="ignore")

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftPreset(),
    ])
    report.run(reference_data=ref, current_data=cur)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    report.save_html(str(html_path))
    log.info(f"Drift report saved: {html_path}")

    result = report.as_dict()
    drift_result = result["metrics"][0]["result"]

    summary = {
        "timestamp": ts,
        "dataset_drift_detected": drift_result.get("dataset_drift", False),
        "drift_share": round(drift_result.get("drift_share", 0.0), 4),
        "n_drifted_features": drift_result.get("number_of_drifted_columns", 0),
        "n_total_features": drift_result.get("number_of_columns", 0),
        "report_path": str(html_path),
        "threshold_used": DRIFT_THRESHOLD,
    }
    return summary


def send_slack_alert(summary: dict):
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook:
        log.info("No Slack webhook — skipping alert")
        return
    import requests
    msg = (f":warning: *Data Drift Detected!*\n"
           f"Drift share: *{summary['drift_share']:.1%}*\n"
           f"Drifted features: {summary['n_drifted_features']}/{summary['n_total_features']}\n"
           f"Threshold: {summary['threshold_used']:.0%}\n"
           f"Action: Retraining triggered automatically.")
    requests.post(webhook, json={"text": msg}, timeout=10)
    log.info("Slack alert sent")


def trigger_retraining():
    log.warning("Drift threshold exceeded — triggering retraining pipeline!")
    try:
        result = subprocess.run(
            [sys.executable, "pipelines/training_pipeline.py"],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            log.info("Retraining completed successfully!")
            return True
        else:
            log.error(f"Retraining failed:\n{result.stderr}")
            return False
    except Exception as e:
        log.error(f"Retraining error: {e}")
        return False


def simulate_drift(df: pd.DataFrame, drift_level: float = 0.3) -> pd.DataFrame:
    """Simulate data drift by adding noise — for testing purposes."""
    drifted = df.copy()
    numeric_cols = drifted.select_dtypes(include=np.number).columns.tolist()
    cols_to_drift = numeric_cols[:int(len(numeric_cols) * drift_level)]
    for col in cols_to_drift:
        std = drifted[col].std()
        drifted[col] = drifted[col] + np.random.normal(0, std * 1.5, len(drifted))
    log.info(f"Simulated drift on {len(cols_to_drift)} features")
    return drifted


def main(simulate: bool = False, auto_retrain: bool = True):
    log.info("=== Starting Drift Monitor ===")

    reference_path = "data/raw/churn.csv"
    if not Path(reference_path).exists():
        log.error("Reference data not found. Run DVC pull first.")
        sys.exit(1)

    reference_df = load_and_encode(reference_path)
    log.info(f"Reference data: {len(reference_df):,} rows")

    if simulate:
        log.info("Simulating production data with drift...")
        # Take a sample and add drift
        current_df = simulate_drift(
            reference_df.sample(500, random_state=99),
            drift_level=0.4
        )
        log.info(f"Simulated current data: {len(current_df):,} rows")
    else:
        # In production: load from your prediction log DB
        # For now use a recent sample of reference data (no drift)
        current_df = reference_df.sample(500, random_state=42)
        log.info(f"Current data sample: {len(current_df):,} rows")

    summary = run_drift_report(reference_df, current_df)

    log.info(f"Drift share: {summary['drift_share']:.1%}")
    log.info(f"Drifted features: {summary['n_drifted_features']}/{summary['n_total_features']}")
    log.info(f"Drift detected: {summary['dataset_drift_detected']}")

    # Save summary
    summary_path = REPORTS_DIR / "latest_drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary saved: {summary_path}")

    # Check threshold and act
    if summary["drift_share"] > DRIFT_THRESHOLD:
        log.warning(f"DRIFT ALERT: {summary['drift_share']:.1%} > threshold {DRIFT_THRESHOLD:.0%}")
        send_slack_alert(summary)
        if auto_retrain:
            retrained = trigger_retraining()
            summary["retraining_triggered"] = True
            summary["retraining_success"] = retrained
        else:
            summary["retraining_triggered"] = False
    else:
        log.info(f"No significant drift — {summary['drift_share']:.1%} < threshold {DRIFT_THRESHOLD:.0%}")
        summary["retraining_triggered"] = False

    summary_path.write_text(json.dumps(summary, indent=2))
    # Append to history log
    history_path = REPORTS_DIR / "drift_history.json"
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except:
            history = []
    history.append(summary)
    history_path.write_text(json.dumps(history, indent=2))
    log.info("=== Drift Monitor Complete ===")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true",
                        help="Simulate drift for testing")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Detect drift but skip retraining")
    args = parser.parse_args()
    main(simulate=args.simulate, auto_retrain=not args.no_retrain)