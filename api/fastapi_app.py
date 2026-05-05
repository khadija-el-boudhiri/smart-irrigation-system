import os
from pathlib import Path
import sys

import mlflow
import mlflow.pyfunc
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

try:
    from src.schema import MODEL_FEATURES, REGISTERED_MODEL_NAME
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.schema import MODEL_FEATURES, REGISTERED_MODEL_NAME

app = FastAPI()
try:
    app.mount("/static", StaticFiles(directory="api"), name="static")
except Exception:
    pass
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
model = None


class IrrigationInput(BaseModel):
    soil_pct: float
    temperature: float
    pressure: float
    altitude: float


def load_model() -> None:
    global model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    model_uri = os.getenv(
        "MLFLOW_MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}@production"
    )
    mlflow.set_tracking_uri(tracking_uri)
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception:
        model = None


@app.get("/")
def home():
    return {"message": "Smart Irrigation FastAPI is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui")
def ui():
    if os.path.exists("api/index.html"):
        return FileResponse("api/index.html")
    else:
        return JSONResponse(status_code=503, content={"message": "UI not available yet"})


@app.post("/predict")
def predict(payload: IrrigationInput):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model is not loaded yet"},
        )

    data = payload.model_dump()
    
    # Validate field ranges (must match Flask validation)
    ranges = {
        "soil_pct": (0, 100),
        "temperature": (10, 42),
        "pressure": (9780, 10120),
        "altitude": (0, 500)
    }
    for field in MODEL_FEATURES:
        if field in ranges:
            min_val, max_val = ranges[field]
            if not (min_val <= data[field] <= max_val):
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Field {field} is out of valid range"},
                )
    
    features = [[data[col] for col in MODEL_FEATURES]]
    prediction = model.predict(features)
    return {"needs_irrigation": bool(prediction[0])}


load_model()

if __name__ == "__main__":
    uvicorn.run("api.fastapi_app:app", host="0.0.0.0", port=8000)
