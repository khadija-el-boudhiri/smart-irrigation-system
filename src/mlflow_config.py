import os

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "Smart Irrigation Multi Models"
DEFAULT_REGISTERED_MODEL_NAME = "PlantWaterModel"
DEFAULT_ZENML_TRACKER_NAME = "mlflow_tracker"


def get_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def get_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)


def configure_mlflow() -> tuple[str, str]:
    tracking_uri = get_tracking_uri()
    experiment_name = get_experiment_name()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri, experiment_name
