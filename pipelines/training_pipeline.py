from zenml import pipeline

from src.mlflow_config import DEFAULT_REGISTERED_MODEL_NAME
from steps.evaluate_step import evaluate_step
from steps.load_data import load_data_step
from steps.promote_step import promote_step
from steps.train_step import train_step


@pipeline(enable_cache=False)
def training_pipeline(
    data_path: str = "features_ready.csv",
    target_column: str = "status",
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
):
    """End-to-end training/evaluation/promotion pipeline."""
    data = load_data_step(path=data_path)
    models, X_test, y_test = train_step(data=data, target_column=target_column)
    best_model_name, metrics = evaluate_step(models=models, X_test=X_test, y_test=y_test)
    promote_step(
        models=models,
        best_model_name=best_model_name,
        metrics=metrics,
        registered_model_name=registered_model_name,
    )
