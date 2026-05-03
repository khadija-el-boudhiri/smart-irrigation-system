"""Register a fitted sklearn model in MLflow and set the production alias (matches API: models:/...@production)."""

from __future__ import annotations

import numbers
from typing import Any, Dict, Mapping, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.mlflow_config import configure_mlflow, get_tracking_uri
from src.schema import MODEL_FEATURES


def publish_sklearn_to_registry(
    model: Any,
    *,
    run_name: str,
    metrics: Mapping[str, float],
    registered_model_name: str,
    extra_params: Optional[Dict[str, str]] = None,
    production_alias: str = "production",
) -> str:
    """
    Single MLflow run: log metrics/params, log_model with registry, then set alias
    (same mechanism as src/promote_model.py for serving).
    """
    configure_mlflow()
    extra_params = extra_params or {}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("features", ",".join(MODEL_FEATURES))
        for key, value in extra_params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                mlflow.log_metric(key, float(value))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )
        run_id = mlflow.active_run().info.run_id

    client = MlflowClient(tracking_uri=get_tracking_uri())
    matching = [
        mv
        for mv in client.search_model_versions(f"name='{registered_model_name}'")
        if mv.run_id == run_id
    ]
    if not matching:
        raise RuntimeError(
            f"No registry version found for model {registered_model_name!r} run {run_id}"
        )
    version = max(int(mv.version) for mv in matching)

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=production_alias,
        version=str(version),
    )
    return str(version)
