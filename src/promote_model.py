import argparse

import mlflow
from mlflow import MlflowClient

try:
    from src.schema import EXPERIMENT_NAME, REGISTERED_MODEL_NAME
    from src.mlflow_config import get_tracking_uri
except ModuleNotFoundError:
    from schema import EXPERIMENT_NAME, REGISTERED_MODEL_NAME
    from mlflow_config import get_tracking_uri


def promote_best_model(target_alias: str = "production") -> str:
    mlflow.set_tracking_uri(get_tracking_uri())
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.selection_score DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError("No runs found to promote.")

    best_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except Exception:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=best_run_id,
    )

    target_alias = target_alias.lower()
    if target_alias not in {"staging", "production"}:
        raise ValueError("target_alias must be 'staging' or 'production'.")

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=target_alias,
        version=model_version.version,
    )

    return (
        f"Promoted run {best_run_id} as version {model_version.version} "
        f"to alias '{target_alias}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote best MLflow run to model registry")
    parser.add_argument(
        "--target",
        default="production",
        choices=["staging", "production"],
        help="Target alias in the model registry.",
    )
    args = parser.parse_args()
    print(promote_best_model(args.target))
