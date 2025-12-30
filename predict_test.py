import pickle
import pandas as pd

with open("models/model.bin", "rb") as f:
    model = pickle.load(f)

# Load feature columns used during training
with open("models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

print("Model loaded successfully")
patient = {
    "Age": 63,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 145,
    "Cholesterol": 233,
    "FastingBS": 1,
    "RestingECG": "Normal",
    "MaxHR": 150,
    "ExerciseAngina": "N",
    "Oldpeak": 2.3,
    "ST_Slope": "Flat"
}

df = pd.DataFrame([patient])

binary_map = {"M": 1, "F": 0, "Y": 1, "N": 0}
df["Sex"] = df["Sex"].map(binary_map)
df["ExerciseAngina"] = df["ExerciseAngina"].map(binary_map)

df = pd.get_dummies(
    df,
    columns=["ChestPainType", "RestingECG", "ST_Slope"],
    drop_first=True
)

df = df.reindex(columns=feature_columns, fill_value=0)

prediction = model.predict(df)[0]
probability = model.predict_proba(df)[0][1]

if prediction == 1:
    risk_label = "High Risk"
    message = "Patient is at risk of heart disease."
else:
    risk_label = "Low Risk"
    message = "Patient is not at high risk of heart disease."

print("\n--- Heart Disease Prediction Result ---")
print(f"Prediction (0 = No Disease, 1 = Disease): {int(prediction)}")
print(f"Probability of Heart Disease: {round(probability, 3)}")
print(f"Risk Level: {risk_label}")
print(f"Interpretation: {message}")