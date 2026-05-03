from typing import Any, Dict

from src.mlflow_config import DEFAULT_REGISTERED_MODEL_NAME
from src.publish_model import publish_sklearn_to_registry
from zenml import step


@step
def promote_step(
    models: Dict[str, Any],
    best_model_name: str,
    metrics: Dict[str, float],
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> str:
    """Register best model and set MLflow production alias (aligned with src/publish_model.py)."""
    return publish_sklearn_to_registry(
        models[best_model_name],
        run_name=f"zenml-{best_model_name}",
        metrics=metrics,
        registered_model_name=registered_model_name,
        extra_params={"best_model_name": best_model_name},
    )
