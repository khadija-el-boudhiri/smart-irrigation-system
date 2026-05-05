import sys
from pathlib import Path

import pandas as pd
import pytest

# Project root so `import src.preprocess` works when pytest is run from repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def valid_irrigation_df() -> pd.DataFrame:
    """
    Ten rows with all MODEL_FEATURES and TARGET_COLUMN, values inside preprocess ranges.
    Class balance 50/50 so preprocess_data can stratify the train/test split.
    """
    return pd.DataFrame(
        {
            "soil_pct": [25.0, 40.5, 55.0, 72.3, 88.0, 15.2, 33.7, 61.0, 49.1, 95.4],
            "temperature": [18.0, 22.5, 28.0, 35.2, 12.1, 20.0, 30.0, 38.0, 15.5, 25.0],
            "pressure": [
                9900,
                10050,
                10100,
                9850,
                9980,
                10000,
                9790,
                10115,
                9920,
                10020,
            ],
            "altitude": [50, 120, 0, 300, 450, 200, 80, 500, 10, 250],
            "status": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
