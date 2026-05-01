import os
import mlflow
import mlflow.pyfunc
from flask import Flask, jsonify, request
try:
    from src.schema import MODEL_FEATURES, REGISTERED_MODEL_NAME
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.schema import MODEL_FEATURES, REGISTERED_MODEL_NAME

app = Flask(__name__)
model = None


def load_model():
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


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Smart Irrigation API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    if model is None:
        return jsonify({"error": "Model is not loaded yet"}), 503

    missing = [field for field in MODEL_FEATURES if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    features = [[data[col] for col in MODEL_FEATURES]]
    prediction = model.predict(features)
    return jsonify({"needs_irrigation": bool(prediction[0])})


load_model()

if __name__ == "__main__":
    app.run(debug=True)
