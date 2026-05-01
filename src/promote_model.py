from mlflow.tracking import MlflowClient
from src.mlflow_config import DEFAULT_REGISTERED_MODEL_NAME, get_tracking_uri

client = MlflowClient(tracking_uri=get_tracking_uri())

# Passer en Staging pour tests
client.transition_model_version_stage(
    name=DEFAULT_REGISTERED_MODEL_NAME,
    version=1,
    stage="Staging"
)

# Après validation → Production
client.transition_model_version_stage(
    name=DEFAULT_REGISTERED_MODEL_NAME,
    version=1,
    stage="Production"
)