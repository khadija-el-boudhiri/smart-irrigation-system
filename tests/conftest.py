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
    """Create a valid DataFrame for testing preprocessing functions."""
    return pd.DataFrame({
        'soil_moisture': [35.2, 40.5, 28.3],
        'temperature': [28.0, 32.5, 25.0],
        'pressure': [9984.5, 10012.3, 9975.0],
        'altitude': [12.1, 15.0, 10.5],
        'irrigation_need': [1, 0, 1]
    })
