import os

import mlflow
import mlflow.sklearn

try:
    from src.preprocess import load_ready_dataset
    from src.evaluate import (
        adjusted_selection_score,
        cross_validate_pipeline_metrics,
        evaluate_model,
    )
    from src.model_training import (
        fit_all_pipelines,
        prepare_split_and_unfitted_pipelines,
    )
    from src.schema import EXPERIMENT_NAME, MODEL_FEATURES, TARGET_COLUMN
    from src.mlflow_config import get_tracking_uri
except ModuleNotFoundError:
    from preprocess import load_ready_dataset
    from evaluate import (
        adjusted_selection_score,
        cross_validate_pipeline_metrics,
        evaluate_model,
    )
    from model_training import fit_all_pipelines, prepare_split_and_unfitted_pipelines
    from schema import EXPERIMENT_NAME, MODEL_FEATURES, TARGET_COLUMN
    from mlflow_config import get_tracking_uri

mlflow.set_tracking_uri(get_tracking_uri())
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_PATH = os.environ.get(
    "TRAIN_DATA_PATH", "data/processed/features_ready.csv"
)

data = load_ready_dataset(DATA_PATH)
X_train, X_test, y_train, y_test, pipelines = prepare_split_and_unfitted_pipelines(
    data, TARGET_COLUMN
)

best_adjusted = -1.0
best_model_name = None

for name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=name):
        cv_m = cross_validate_pipeline_metrics(pipeline, X_train, y_train)
        mlflow.log_param("model_name", name)
        mlflow.log_param("features", ",".join(MODEL_FEATURES))
        mlflow.log_metric("cv_mean_f1", cv_m["cv_mean_f1"])
        mlflow.log_metric("cv_std_f1", cv_m["cv_std_f1"])
        mlflow.log_metric("cv_mean_accuracy", cv_m["cv_mean_accuracy"])
        mlflow.log_metric("cv_std_accuracy", cv_m["cv_std_accuracy"])

        adj = adjusted_selection_score(cv_m["cv_mean_f1"], cv_m["cv_std_f1"])
        mlflow.log_metric("selection_score", adj)

        pipeline.fit(X_train, y_train)
        accuracy, f1, matrix, report = evaluate_model(pipeline, X_test, y_test)
        mlflow.log_metric("holdout_accuracy", accuracy)
        mlflow.log_metric("holdout_f1", f1)

        mlflow.sklearn.log_model(pipeline, "model")

        print("Model:", name)
        print("CV mean F1:", cv_m["cv_mean_f1"], "std:", cv_m["cv_std_f1"])
        print("CV mean acc:", cv_m["cv_mean_accuracy"], "std:", cv_m["cv_std_accuracy"])
        print("Selection score (adj. F1):", adj)
        print("Holdout accuracy:", accuracy, "F1:", f1)
        print("Confusion Matrix:")
        print(matrix)
        print(report)
        print("------------------------")

        if adj > best_adjusted:
            best_adjusted = adj
            best_model_name = name

print("Best model (by CV mean F1 with stability penalty):", best_model_name)
print("Best selection score:", best_adjusted)
