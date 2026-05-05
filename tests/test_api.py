"""
API tests: `load_model()` still runs at import, but `mlflow.pyfunc.load_model` is patched so no
tracking server, SQLite URI, or model artifact is touched; both apps receive the same mock with
`predict` returning ``[1]``.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _valid_predict_json():
    return {
        "soil_pct": 45.0,
        "temperature": 24.5,
        "pressure": 10000.0,
        "altitude": 120.0,
    }


@pytest.fixture(scope="module")
def api_modules():
    """Load api.app and api.fastapi_app once, with mlflow.pyfunc.load_model patched."""
    for name in ("api.app", "api.fastapi_app"):
        sys.modules.pop(name, None)

    mock_model = MagicMock()
    mock_model.predict.return_value = [1]

    with patch("mlflow.pyfunc.load_model", return_value=mock_model):
        import api.app as flask_app_module  # noqa: PLC0415
        import api.fastapi_app as fastapi_app_module  # noqa: PLC0415

    return {
        "flask": flask_app_module,
        "fastapi": fastapi_app_module,
        "mock_model": mock_model,
    }


@pytest.fixture
def flask_client(api_modules):
    return api_modules["flask"].app.test_client()


@pytest.fixture
def fastapi_client(api_modules):
    # Starlette's TestClient shells out to httpx (not always installed with FastAPI).
    pytest.importorskip("httpx")
    from starlette.testclient import TestClient  # noqa: PLC0415

    return TestClient(api_modules["fastapi"].app)


def test_flask_home(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body is not None
    assert "message" in body


def test_flask_predict_valid(flask_client, api_modules):
    resp = flask_client.post("/predict", json=_valid_predict_json())
    assert resp.status_code == 200
    body = resp.get_json()
    assert body is not None
    assert "needs_irrigation" in body
    assert isinstance(body["needs_irrigation"], bool)
    assert body["needs_irrigation"] is True
    api_modules["mock_model"].predict.assert_called()


def test_flask_predict_missing_field(flask_client):
    payload = _valid_predict_json()
    del payload["altitude"]
    resp = flask_client.post("/predict", json=payload)
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert "error" in body


def test_flask_predict_model_not_loaded(flask_client, api_modules, monkeypatch):
    monkeypatch.setattr(api_modules["flask"], "model", None)
    resp = flask_client.post("/predict", json=_valid_predict_json())
    assert resp.status_code == 503
    body = resp.get_json()
    assert body is not None
    assert "error" in body


def test_fastapi_home(fastapi_client):
    resp = fastapi_client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_fastapi_predict_valid(fastapi_client, api_modules):
    resp = fastapi_client.post("/predict", json=_valid_predict_json())
    assert resp.status_code == 200
    data = resp.json()
    assert "needs_irrigation" in data
    assert isinstance(data["needs_irrigation"], bool)
    assert data["needs_irrigation"] is True
    api_modules["mock_model"].predict.assert_called()


def test_fastapi_predict_missing_field(fastapi_client):
    payload = _valid_predict_json()
    del payload["pressure"]
    resp = fastapi_client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_fastapi_predict_model_not_loaded(fastapi_client, api_modules, monkeypatch):
    monkeypatch.setattr(api_modules["fastapi"], "model", None)
    resp = fastapi_client.post("/predict", json=_valid_predict_json())
    assert resp.status_code == 503
    assert "error" in resp.json()


def _app_has_get_path(app, path: str) -> bool:
    for route in app.routes:
        p = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if p == path and "GET" in methods:
            return True
    return False


def test_fastapi_health(fastapi_client, api_modules):
    app = api_modules["fastapi"].app
    if not _app_has_get_path(app, "/health"):
        pytest.skip("/health is not registered in api.fastapi_app yet")
    resp = fastapi_client.get("/health")
    assert resp.status_code == 200
