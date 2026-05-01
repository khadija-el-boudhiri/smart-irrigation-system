import os

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.schema import MODEL_FEATURES, TARGET_COLUMN
except ModuleNotFoundError:
    from schema import MODEL_FEATURES, TARGET_COLUMN


def load_ready_dataset(path: str = "data/processed/features_ready.csv") -> pd.DataFrame:
    """DataOps: load the already prepared dataset from disk."""
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> None:
    """MLOps: ensure required training columns are present."""
    required_columns = MODEL_FEATURES + [target_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in training data: {missing_columns}")


def preprocess_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    """
    Split raw features for training. Scaling is done inside each model Pipeline
    in train_models.py so the same preprocessing is saved with the model for the API.
    """
    validate_schema(df, target_column)

    X = df[MODEL_FEATURES]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    path = os.environ.get(
        "TRAIN_DATA_PATH", "data/processed/features_ready.csv"
    )
    df = load_ready_dataset(path)
    validate_schema(df)
    print("Dataset loaded and validated.")
    print(f"Rows: {len(df)}")
    print(f"Features used by model: {MODEL_FEATURES}")
