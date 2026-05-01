from zenml import step

import pandas as pd


@step
def load_data_step(path: str = "features_ready.csv") -> pd.DataFrame:
    """Load the prepared dataset used for model training."""
    return pd.read_csv(path)
