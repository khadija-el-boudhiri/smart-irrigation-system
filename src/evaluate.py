from typing import Any, Dict, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate

CV_N_SPLITS = 5
CV_RANDOM_STATE = 42
"""If fold-to-fold F1 std exceeds this, excess is subtracted from mean F1 for model ranking."""
CV_STABILITY_F1_STD_THRESHOLD = 0.02


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, f1, matrix, report


def cross_validate_pipeline_metrics(
    pipeline: Any,
    X_train,
    y_train,
    n_splits: int = CV_N_SPLITS,
) -> Dict[str, float]:
    """5-fold stratified CV; returns mean/std accuracy and weighted F1 across folds."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    scores = cross_validate(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1_weighted",
        },
        n_jobs=-1,
    )
    acc = scores["test_accuracy"]
    f1 = scores["test_f1"]
    ddof = 1 if len(f1) > 1 else 0
    return {
        "cv_mean_accuracy": float(np.mean(acc)),
        "cv_std_accuracy": float(np.std(acc, ddof=ddof)),
        "cv_mean_f1": float(np.mean(f1)),
        "cv_std_f1": float(np.std(f1, ddof=ddof)),
    }


def adjusted_selection_score(cv_mean_f1: float, cv_std_f1: float) -> float:
    """
    Rank models by mean CV F1, penalizing instability when std(F1) > threshold.
    """
    penalty = max(0.0, cv_std_f1 - CV_STABILITY_F1_STD_THRESHOLD)
    return cv_mean_f1 - penalty


def pick_best_model_by_cv(
    pipelines: Dict[str, Any],
    X_train,
    y_train,
) -> Tuple[str, Dict[str, float]]:
    """
    Cross-validate each unfitted pipeline on X_train, log-style flat metrics per model,
    select best by adjusted_selection_score (cv_mean_f1 with stability penalty).
    """
    flat_metrics: Dict[str, float] = {}
    best_model_name = ""
    best_adjusted = -1.0

    for model_name, pipeline in pipelines.items():
        cv_m = cross_validate_pipeline_metrics(pipeline, X_train, y_train)
        for key, value in cv_m.items():
            flat_metrics[f"{model_name}_{key}"] = float(value)

        adj = adjusted_selection_score(cv_m["cv_mean_f1"], cv_m["cv_std_f1"])
        flat_metrics[f"{model_name}_selection_score"] = float(adj)

        if adj > best_adjusted:
            best_adjusted = adj
            best_model_name = model_name

    flat_metrics["selection_adjusted_f1_best"] = float(best_adjusted)
    return best_model_name, flat_metrics


def pick_best_model_by_accuracy(
    models: Dict[str, Any],
    X_test,
    y_test,
) -> Tuple[str, Dict[str, float]]:
    """Legacy: single holdout accuracy (not used for production selection)."""
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
