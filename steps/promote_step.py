from typing import Any, Dict

import pandas as pd
from src.evaluate import evaluate_model
from src.mlflow_config import DEFAULT_REGISTERED_MODEL_NAME
from src.publish_model import publish_sklearn_to_registry
from zenml import step


@step
def promote_step(
    models: Dict[str, Any],
    best_model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Dict[str, float],
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> str:
    """Register best model and set MLflow production alias (aligned with src/publish_model.py)."""
    best_model = models[best_model_name]
    holdout_acc, holdout_f1, _, _ = evaluate_model(best_model, X_test, y_test)
    enriched_metrics = dict(metrics)
    enriched_metrics["holdout_accuracy"] = float(holdout_acc)
    enriched_metrics["holdout_f1"] = float(holdout_f1)

    return publish_sklearn_to_registry(
        best_model,
        run_name=f"zenml-{best_model_name}",
        metrics=enriched_metrics,
        registered_model_name=registered_model_name,
        extra_params={"best_model_name": best_model_name},
    )
