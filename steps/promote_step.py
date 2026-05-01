from typing import Any, Dict

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.mlflow_config import (
    DEFAULT_REGISTERED_MODEL_NAME,
    configure_mlflow,
    get_tracking_uri,
)
from zenml import step


@step
def promote_step(
    models: Dict[str, Any],
    best_model_name: str,
    metrics: Dict[str, float],
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> str:
    """Log and register the best model in MLflow and promote it to Production."""
    configure_mlflow()

    best_model = models[best_model_name]

    with mlflow.start_run(run_name=f"zenml-{best_model_name}"):
        mlflow.log_param("best_model_name", best_model_name)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    client = MlflowClient(tracking_uri=get_tracking_uri())
    latest_versions = client.get_latest_versions(registered_model_name)
    latest_version_number = max(int(version.version) for version in latest_versions)

    client.transition_model_version_stage(
        name=registered_model_name,
        version=latest_version_number,
        stage="Production",
    )

    return str(latest_version_number)
