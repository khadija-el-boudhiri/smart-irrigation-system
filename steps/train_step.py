from typing import Any, Dict, Tuple

import pandas as pd
from zenml import step

from src.evaluate import pick_best_model_by_cv
from src.model_training import (
    fit_all_pipelines,
    prepare_split_and_unfitted_pipelines,
)
from src.schema import TARGET_COLUMN


@step
def train_step(
    data: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series, str, Dict[str, float]]:
    """
    Split, 5-fold CV model selection (adjusted mean F1), then fit all pipelines on train.
    ZenML requires a literal tuple return for multiple artifacts.
    """
    X_train, X_test, y_train, y_test, pipelines = prepare_split_and_unfitted_pipelines(
        data, target_column
    )
    best_model_name, metrics = pick_best_model_by_cv(pipelines, X_train, y_train)
    fit_all_pipelines(pipelines, X_train, y_train)
    return pipelines, X_test, y_test, best_model_name, metrics
