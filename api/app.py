import os
import mlflow.pyfunc
from flask import Flask, jsonify, request

app = Flask(__name__)
model = None

def load_model():
    global model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/PlantWaterModel/Production")
    mlflow.pyfunc.set_tracking_uri(tracking_uri)
    model = mlflow.pyfunc.load_model(model_uri)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Smart Irrigation API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    if model is None:
        return jsonify({"error": "Model is not loaded yet"}), 503

    required_features = ["hum_sol", "lux", "temp", "hum_air"]
    missing = [field for field in required_features if field not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    features = [[data["hum_sol"], data["lux"], data["temp"], data["hum_air"]]]
    prediction = model.predict(features)
    return jsonify({"besoin_eau": bool(prediction[0])})


load_model()

if __name__ == "__main__":
    app.run(debug=True)