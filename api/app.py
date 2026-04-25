from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Smart Irrigation API is running"})

# NEW endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    soil_moisture = data.get("soil_moisture")

    if soil_moisture is None:
        return jsonify({"error": "soil_moisture is required"}), 400

    # simple rule-based "fake model"
    if soil_moisture < 40:
        decision = "IRRIGATE"
    else:
        decision = "DO_NOT_IRRIGATE"

    return jsonify({
        "soil_moisture": soil_moisture,
        "decision": decision
    })

if __name__ == "__main__":
    app.run(debug=True)