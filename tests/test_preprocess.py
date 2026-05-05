import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    preprocess_data,
    validate_class_balance,
    validate_missing,
    validate_ranges,
    validate_schema,
)
from src.schema import MODEL_FEATURES, TARGET_COLUMN


def test_validate_schema_passes(valid_irrigation_df):
    validate_schema(valid_irrigation_df)


def test_validate_schema_missing_col(valid_irrigation_df):
    df = valid_irrigation_df.drop(columns=["temperature"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df)


def test_validate_ranges_drops_rows(valid_irrigation_df):
    df = valid_irrigation_df.copy()
    df.loc[0, "soil_pct"] = 999.0
    out = validate_ranges(df)
    assert len(out) == len(valid_irrigation_df) - 1
    assert (out["soil_pct"] <= 100).all()


def test_validate_ranges_all_valid(valid_irrigation_df):
    out = validate_ranges(valid_irrigation_df.copy())
    assert len(out) == len(valid_irrigation_df)


def test_validate_missing_drops_rows(valid_irrigation_df):
    df = valid_irrigation_df.copy()
    df.loc[2, "temperature"] = np.nan
    out = validate_missing(df)
    assert len(out) == len(valid_irrigation_df) - 1
    assert not out.isna().any().any()


def test_validate_missing_none_dropped(valid_irrigation_df):
    out = validate_missing(valid_irrigation_df.copy())
    assert len(out) == len(valid_irrigation_df)


def test_validate_class_balance_warns(capsys):
    # 95% class 0, 5% class 1 → minority share < 30% triggers imbalance warning.
    n0, n1 = 95, 5
    df = pd.DataFrame(
        {
            "soil_pct": [50.0] * (n0 + n1),
            "temperature": [25.0] * (n0 + n1),
            "pressure": [10000] * (n0 + n1),
            "altitude": [100] * (n0 + n1),
            "status": [0] * n0 + [1] * n1,
        }
    )
    validate_class_balance(df, TARGET_COLUMN)
    captured = capsys.readouterr().out
    assert "WARNING: dataset is heavily imbalanced" in captured


def test_preprocess_data_split_shapes(valid_irrigation_df):
    X_train, X_test, y_train, y_test = preprocess_data(valid_irrigation_df.copy())
    n = len(valid_irrigation_df)
    assert X_train.shape == (int(n * 0.8), len(MODEL_FEATURES))
    assert X_test.shape == (int(n * 0.2), len(MODEL_FEATURES))
    assert y_train.shape == (int(n * 0.8),)
    assert y_test.shape == (int(n * 0.2),)
    assert list(X_train.columns) == MODEL_FEATURES


def test_preprocess_data_no_nan(valid_irrigation_df):
    X_train, X_test, y_train, y_test = preprocess_data(valid_irrigation_df.copy())
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert not y_train.isna().any()
    assert not y_test.isna().any()
