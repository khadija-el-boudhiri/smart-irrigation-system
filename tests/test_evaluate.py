import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluate import (  # noqa: E402
    CV_STABILITY_F1_STD_THRESHOLD,
    adjusted_selection_score,
    cross_validate_pipeline_metrics,
    evaluate_model,
    pick_best_model_by_accuracy,
    pick_best_model_by_cv,
)
from src.schema import MODEL_FEATURES, TARGET_COLUMN  # noqa: E402


def test_adjusted_selection_score_no_penalty():
    mean_f1 = 0.78
    std_f1 = CV_STABILITY_F1_STD_THRESHOLD * 0.5
    score = adjusted_selection_score(mean_f1, std_f1)
    assert score == pytest.approx(mean_f1)


def test_adjusted_selection_score_with_penalty():
    mean_f1 = 0.9
    std_f1 = 0.1
    expected = mean_f1 - (std_f1 - CV_STABILITY_F1_STD_THRESHOLD)
    score = adjusted_selection_score(mean_f1, std_f1)
    assert score == pytest.approx(expected)
    assert score < mean_f1


def test_adjusted_selection_score_zero_std():
    mean_f1 = 0.65
    score = adjusted_selection_score(mean_f1, 0.0)
    assert score == pytest.approx(mean_f1)


@pytest.fixture
def synthetic_xy():
    rng = np.random.default_rng(42)
    n = 50
    X = rng.uniform(low=0.0, high=1.0, size=(n, len(MODEL_FEATURES)))
    logits = 3.0 * X[:, 0] + 0.5 * rng.standard_normal(n)
    y = (logits > np.median(logits)).astype(np.int64)
    X_df = pd.DataFrame(X, columns=MODEL_FEATURES)
    y_series = pd.Series(y, name=TARGET_COLUMN)
    return X_df, y_series


@pytest.fixture
def train_test_bundle(synthetic_xy):
    X, y = synthetic_xy
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


@pytest.fixture
def fitted_eval_pipeline(train_test_bundle):
    X_train, X_test, y_train, y_test = train_test_bundle
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=100, random_state=0)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


@pytest.fixture
def cv_xy(train_test_bundle):
    X_train, _, y_train, _ = train_test_bundle
    return X_train, y_train


@pytest.fixture
def pipelines_dict():
    return {
        "log_reg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=100, random_state=1)),
            ]
        ),
        "dummy_most_frequent": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
    }


@pytest.fixture
def fitted_models_dict(train_test_bundle):
    X_train, X_test, y_train, y_test = train_test_bundle
    models = {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=100, random_state=0)),
            ]
        ),
        "dummy": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        ),
    }
    for pipe in models.values():
        pipe.fit(X_train, y_train)
    return models, X_test, y_test


def test_evaluate_model_returns_correct_types(fitted_eval_pipeline):
    model, X_test, y_test = fitted_eval_pipeline
    accuracy, f1, matrix, report = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float)
    assert isinstance(f1, float)
    assert isinstance(matrix, np.ndarray)
    assert isinstance(report, str)


def test_evaluate_model_accuracy_range(fitted_eval_pipeline):
    model, X_test, y_test = fitted_eval_pipeline
    accuracy, _, _, _ = evaluate_model(model, X_test, y_test)
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_model_f1_range(fitted_eval_pipeline):
    model, X_test, y_test = fitted_eval_pipeline
    _, f1, _, _ = evaluate_model(model, X_test, y_test)
    assert 0.0 <= f1 <= 1.0


def test_cross_validate_returns_four_keys(cv_xy, pipelines_dict):
    X_train, y_train = cv_xy
    pipe = pipelines_dict["log_reg"]
    result = cross_validate_pipeline_metrics(pipe, X_train, y_train)
    assert set(result.keys()) == {
        "cv_mean_accuracy",
        "cv_std_accuracy",
        "cv_mean_f1",
        "cv_std_f1",
    }


def test_cross_validate_values_in_range(cv_xy, pipelines_dict):
    X_train, y_train = cv_xy
    pipe = pipelines_dict["log_reg"]
    result = cross_validate_pipeline_metrics(pipe, X_train, y_train)
    for key, value in result.items():
        assert 0.0 <= value <= 1.0, f"{key}={value} out of range"


def test_pick_best_model_by_cv_returns_name(cv_xy, pipelines_dict):
    X_train, y_train = cv_xy
    name, _ = pick_best_model_by_cv(pipelines_dict, X_train, y_train)
    assert isinstance(name, str)
    assert name in pipelines_dict


def test_pick_best_model_by_cv_returns_metrics(cv_xy, pipelines_dict):
    X_train, y_train = cv_xy
    _, metrics = pick_best_model_by_cv(pipelines_dict, X_train, y_train)
    assert "selection_adjusted_f1_best" in metrics
    assert isinstance(metrics["selection_adjusted_f1_best"], float)


def test_pick_best_model_by_accuracy_returns_name(fitted_models_dict):
    models, X_test, y_test = fitted_models_dict
    name, _ = pick_best_model_by_accuracy(models, X_test, y_test)
    assert isinstance(name, str)
    assert name in models


def test_pick_best_model_by_accuracy_returns_metrics(fitted_models_dict):
    models, X_test, y_test = fitted_models_dict
    _, metrics = pick_best_model_by_accuracy(models, X_test, y_test)
    for model_name in models:
        assert f"{model_name}_accuracy" in metrics
        assert f"{model_name}_f1" in metrics
        assert isinstance(metrics[f"{model_name}_accuracy"], float)
        assert isinstance(metrics[f"{model_name}_f1"], float)
