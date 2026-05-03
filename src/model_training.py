"""Single training path shared by src/train_models.py (manual MLflow) and ZenML steps."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from src.preprocess import preprocess_data
except ModuleNotFoundError:
    from preprocess import preprocess_data


def build_unfitted_pipelines() -> Dict[str, Pipeline]:
    """Classifier definitions: scaler + estimator (same for script and ZenML)."""
    estimators: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
        ),
    }
    return {
        name: Pipeline([("scaler", StandardScaler()), ("model", est)])
        for name, est in estimators.items()
    }


def fit_all_pipelines(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    for pipeline in pipelines.values():
        pipeline.fit(X_train, y_train)


def prepare_split_and_unfitted_pipelines(
    data: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Pipeline]]:
    """Validate + split + build unfitted pipelines (CV and fit happen downstream)."""
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column)
    pipelines = build_unfitted_pipelines()
    return X_train, X_test, y_train, y_test, pipelines


def prepare_data_and_fit_models(
    data: pd.DataFrame,
    target_column: str,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.Series]:
    """
    Validate + split + build + fit on full train (no CV selection).
    Prefer prepare_split_* + pick_best_model_by_cv + fit_all for ZenML/manual with CV.
    """
    X_train, X_test, y_train, y_test, pipelines = prepare_split_and_unfitted_pipelines(
        data, target_column
    )
    fit_all_pipelines(pipelines, X_train, y_train)
    return pipelines, X_test, y_test
