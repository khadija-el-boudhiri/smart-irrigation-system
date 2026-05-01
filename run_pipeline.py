import os
import subprocess

from pipelines.training_pipeline import training_pipeline
from src.mlflow_config import DEFAULT_ZENML_TRACKER_NAME, get_tracking_uri


def connect_mlflow_tracker() -> None:
    """Configure a modifiable ZenML stack to use the MLflow tracker."""
    tracking_uri = get_tracking_uri()
    tracker_name = os.getenv("ZENML_TRACKER_NAME", DEFAULT_ZENML_TRACKER_NAME)
    stack_name = os.getenv("ZENML_STACK_NAME", "local_mlflow_stack")

    # Re-registering existing components is safe with check=False.
    subprocess.run(
        [
            "zenml",
            "experiment-tracker",
            "register",
            tracker_name,
            "--flavor=mlflow",
            f"--tracking_uri={tracking_uri}",
        ],
        check=False,
    )
    subprocess.run(
        ["zenml", "stack", "copy", "default", stack_name],
        check=False,
    )
    subprocess.run(
        ["zenml", "stack", "update", stack_name, "-e", tracker_name],
        check=True,
    )
    subprocess.run(
        ["zenml", "stack", "set", stack_name],
        check=True,
    )


if __name__ == "__main__":
    connect_mlflow_tracker()
    training_data_path = os.getenv("TRAIN_DATA_PATH", "data/processed/features_ready.csv")
    training_pipeline(data_path=training_data_path)
