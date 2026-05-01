import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from src.preprocess import preprocess_data
    from src.evaluate import evaluate_model
    from src.schema import EXPERIMENT_NAME, MODEL_FEATURES, TARGET_COLUMN
except ModuleNotFoundError:
    from preprocess import preprocess_data
    from evaluate import evaluate_model
    from schema import EXPERIMENT_NAME, MODEL_FEATURES, TARGET_COLUMN

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

DATA_PATH = os.environ.get(
    "TRAIN_DATA_PATH", "data/processed/features_ready.csv"
)

data = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test = preprocess_data(data, TARGET_COLUMN)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )
}

best_accuracy = 0
best_model_name = None

for name, estimator in models.items():
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", estimator)]
    )
    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)

        accuracy, f1, matrix, report = evaluate_model(pipeline, X_test, y_test)

        mlflow.log_param("model_name", name)
        mlflow.log_param("features", ",".join(MODEL_FEATURES))
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, "model")

        print("Model:", name)
        print("Accuracy:", accuracy)
        print("F1-score:", f1)
        print("Confusion Matrix:")
        print(matrix)
        print(report)
        print("------------------------")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

print("Best model:", best_model_name)
print("Best accuracy:", best_accuracy)