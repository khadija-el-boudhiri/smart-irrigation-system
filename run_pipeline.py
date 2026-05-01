import os
import subprocess

from pipelines.training_pipeline import training_pipeline
from src.mlflow_config import DEFAULT_ZENML_TRACKER_NAME, get_tracking_uri


def connect_mlflow_tracker() -> None:
    """Configure ZenML stack to use the existing MLflow tracker."""
    tracking_uri = get_tracking_uri()
    tracker_name = os.getenv("ZENML_TRACKER_NAME", DEFAULT_ZENML_TRACKER_NAME)

    subprocess.run(["zenml", "experiment-tracker", "register", tracker_name, "--flavor=mlflow", f"--tracking_uri={tracking_uri}"], check=False)
    subprocess.run(["zenml", "stack", "update", "default", "-e", tracker_name], check=True)


if __name__ == "__main__":
    connect_mlflow_tracker()
    training_pipeline()
