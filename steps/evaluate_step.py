from typing import Any, Dict, Tuple

import pandas as pd
from src.evaluate import evaluate_model
from zenml import step


@step
def evaluate_step(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, Dict[str, float]]:
    """Evaluate trained models and return the best one by accuracy."""
    metrics: Dict[str, float] = {}
    best_model_name = ""
    best_accuracy = -1.0

    for model_name, model in models.items():
        accuracy, f1, _, _ = evaluate_model(model, X_test, y_test)

        metrics[f"{model_name}_accuracy"] = float(accuracy)
        metrics[f"{model_name}_f1"] = float(f1)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name

    return best_model_name, metrics
