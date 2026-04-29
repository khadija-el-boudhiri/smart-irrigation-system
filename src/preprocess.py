import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    Shared by the team:
    - DataOps: works on a clean tabular dataset
    - MLOps: returns train/test + fitted scaler
    - DevOps: deterministic split (fixed random_state)
    """
    validate_schema(df, target_column)

    X = df[MODEL_FEATURES]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_ready_dataset()
    validate_schema(df)
    print("Dataset loaded and validated.")
    print(f"Rows: {len(df)}")
    print(f"Features used by model: {MODEL_FEATURES}")