from zenml import pipeline

from src.mlflow_config import DEFAULT_REGISTERED_MODEL_NAME
from src.schema import TARGET_COLUMN
from steps.load_data import load_data_step
from steps.promote_step import promote_step
from steps.train_step import train_step


@pipeline(enable_cache=False)
def training_pipeline(
    data_path: str = "data/processed/features_ready.csv",
    target_column: str = TARGET_COLUMN,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
):
    """End-to-end training (CV selection) and promotion."""
    data = load_data_step(path=data_path)
    models, _, _, best_model_name, metrics = train_step(
        data=data, target_column=target_column
    )
    promote_step(
        models=models,
        best_model_name=best_model_name,
        metrics=metrics,
        registered_model_name=registered_model_name,
    )

