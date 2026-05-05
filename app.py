from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import time

app = FastAPI(title="Smart Irrigation API")
Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    time.sleep(0.05)
    return {"prediction": 1, "confidence": 0.92, "model": "irrigation-model"}
