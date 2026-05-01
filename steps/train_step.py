from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from zenml import step


@step
def train_step(
    data: pd.DataFrame,
    target_column: str = "status",
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.Series]:
    """Train all candidate models and return trained estimators."""
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
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

    for model in models.values():
        model.fit(X_train, y_train)

    return models, X_test, y_test
