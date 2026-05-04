from typing import Any, Dict, Tuple

import pandas as pd
from zenml import step

from src.evaluate import evaluate_model, pick_best_model_by_accuracy


@step
def evaluate_step(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[str, Dict[str, float]]:
    """Evaluate fitted models on holdout data and pick the best by accuracy."""
    rows = []
    for model_name, model in models.items():
        accuracy, f1, _, _ = evaluate_model(model, X_test, y_test)
        rows.append((model_name, float(accuracy), float(f1)))

    print("\nHoldout Evaluation Comparison")
    print(f"{'Model':<24} {'Accuracy':>10} {'F1':>10}")
    print("-" * 46)
    for model_name, accuracy, f1 in rows:
        print(f"{model_name:<24} {accuracy:>10.4f} {f1:>10.4f}")

    best_model_name, metrics = pick_best_model_by_accuracy(models, X_test, y_test)
    return best_model_name, metrics

