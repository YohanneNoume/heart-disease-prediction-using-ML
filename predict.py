# Heart Disease Prediction Service (Flask API)

import pickle
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "models/model.bin"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")

app = Flask("heart-disease-prediction")

BINARY_MAP = {
    "Sex": {"F": 0, "M": 1},
    "ExerciseAngina": {"N": 0, "Y": 1}
}

CATEGORICAL_COLS = ["ChestPainType", "RestingECG", "ST_Slope"]

def prepare_features(data):
    """
    Converts raw JSON input into model-ready feature vector
    """
    df = pd.DataFrame([data])

    # Encode binary categorical features
    for col, mapping in BINARY_MAP.items():
        df[col] = df[col].map(mapping)

    # One-hot encode categorical features
    df = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=True
    )

    # Align features with training data
    model_features = model.get_booster().feature_names
    df = df.reindex(columns=model_features, fill_value=0)

    return df

@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    if customer is None:
        return jsonify({"error": "Invalid JSON input"}), 400

    X = prepare_features(customer)

    probability = model.predict_proba(X)[0, 1]
    prediction = int(probability >= 0.5)

    if prediction == 1:
        risk_label = "High Risk"
        risk_message = "Patient is at risk of heart disease"
    else:
        risk_label = "Low Risk"
        risk_message = "Patient is not at high risk of heart disease"

    result = {
        "heart_disease_probability": round(float(probability), 3),
        "heart_disease_prediction": prediction,
        "risk_label": risk_label,
        "risk_message": risk_message
    }

    return jsonify(result)


@app.route("/", methods=["GET"])
def home():
    return """
    <h2> Heart Disease Prediction API</h2>
    <p>Status: <b>Running</b></p>
    <p><b>POST</b> <code>/predict</code> with patient data in JSON format.</p>
    <p>Example fields:</p>
    <ul>
        <li>Age</li>
        <li>Sex</li>
        <li>ChestPainType</li>
        <li>RestingBP</li>
        <li>Cholesterol</li>
        <li>MaxHR</li>
        <li>Oldpeak</li>
        <li>ST_Slope</li>
    </ul>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)