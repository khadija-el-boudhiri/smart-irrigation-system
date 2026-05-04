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


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with feature values outside expected physical bounds."""
    n_before = len(df)
    in_range = (
        (df["soil_pct"] >= 0)
        & (df["soil_pct"] <= 100)
        & (df["temperature"] >= 10)
        & (df["temperature"] <= 42)
        & (df["pressure"] >= 9780)
        & (df["pressure"] <= 10120)
        & (df["altitude"] >= 0)
        & (df["altitude"] <= 500)
    )
    df_clean = df.loc[in_range].copy()
    n_dropped = n_before - len(df_clean)
    n_remain = len(df_clean)
    print(
        f"Dropped {n_dropped} row(s) out of range; {n_remain} row(s) remain."
    )
    return df_clean


def validate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain any NaN."""
    n_before = len(df)
    has_missing = df.isna().any(axis=1)
    n_drop = int(has_missing.sum())
    df_clean = df.loc[~has_missing].copy()
    print(
        f"{n_drop} row(s) had missing values and were dropped."
    )
    return df_clean


def validate_class_balance(df: pd.DataFrame, target_column: str = "status") -> None:
    """Print class distribution; warn if either class is under 30%."""
    total = len(df)
    if total == 0:
        print(f"Class distribution ({target_column}): no rows.")
        return
    counts = df[target_column].value_counts()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))
    p0 = 100.0 * n0 / total
    p1 = 100.0 * n1 / total
    print(
        f"Class distribution ({target_column}): class 0 = {p0:.2f}% ({n0} rows), "
        f"class 1 = {p1:.2f}% ({n1} rows)"
    )
    if p0 < 30 or p1 < 30:
        print(
            "WARNING: dataset is heavily imbalanced — model quality may be affected."
        )


def preprocess_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    """
    Split raw features for training. Scaling is done inside each model Pipeline
    in train_models.py so the same preprocessing is saved with the model for the API.
    """
    validate_schema(df, target_column)
    df = validate_missing(df)
    df = validate_ranges(df)
    validate_class_balance(df, target_column)

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
    df = validate_missing(df)
    df = validate_ranges(df)
    validate_class_balance(df)
    print("All validation checks passed.")
    print("Dataset loaded and validated.")
    print(f"Rows: {len(df)}")
    print(f"Features used by model: {MODEL_FEATURES}")
