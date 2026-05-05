import pandas as pd
import pytest
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def valid_payload():
    return {"soil_pct": 35.2, "temperature": 28.0, "pressure": 9984.5, "altitude": 12.1}

@pytest.fixture
def missing_field_payload():
    return {"temperature": 28.0, "pressure": 9984.5, "altitude": 12.1}

@pytest.fixture
def out_of_range_payload():
    return {"soil_pct": 999, "temperature": 28.0, "pressure": 9984.5, "altitude": 12.1}

@pytest.fixture
def valid_irrigation_df():
    """Create a valid DataFrame with enough rows for stratified split."""
    return pd.DataFrame({
        'soil_pct': [35.2, 40.5, 28.3, 45.0, 32.0, 38.0, 42.0, 30.0, 48.0, 25.0],
        'temperature': [28.0, 32.5, 25.0, 30.0, 27.0, 29.0, 31.0, 26.0, 33.0, 24.0],
        'pressure': [9984.5, 10012.3, 9975.0, 9990.0, 10000.0, 9980.0, 10005.0, 9960.0, 10020.0, 9950.0],
        'altitude': [12.1, 15.0, 10.5, 14.0, 11.0, 13.0, 16.0, 9.0, 17.0, 8.0],
        'status': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 5 of each class
    })
