from zenml import step

import pandas as pd

from src.preprocess import load_ready_dataset


@step
def load_data_step(path: str = "data/processed/features_ready.csv") -> pd.DataFrame:
    """Load the prepared dataset (same helper as src/preprocess.py)."""
    return load_ready_dataset(path)
