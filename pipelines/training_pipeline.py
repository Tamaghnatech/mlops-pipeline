import subprocess
import sys
from prefect import flow, task, get_run_logger

@task(name="train-model", retries=2, retry_delay_seconds=10)
def train_model():
    logger = get_run_logger()
    logger.info("Starting model training...")
    result = subprocess.run(
        [sys.executable, "src/train/train.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Training failed:\n{result.stderr}")
    logger.info(result.stdout)
    return "Training complete"

@flow(name="churn-training-pipeline", log_prints=True)
def training_pipeline():
    logger = get_run_logger()
    logger.info("=== Churn Training Pipeline Started ===")
    result = train_model()
    logger.info(f"Pipeline finished: {result}")
    return result

if __name__ == "__main__":
    training_pipeline()