# train.py
# Train Heart Disease Prediction Model (XGBoost)


import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# Load data
DATA_PATH = "data/heart.csv"   
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# Basic cleaning
df = df.drop_duplicates().reset_index(drop=True)

# Separate target
target = "HeartDisease"
y = df[target].values
X = df.drop(columns=[target])

# Scale numerical features
num_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Encode categorical features

binary_map = {"F": 0, "M": 1, "N": 0, "Y": 1}
X["Sex"] = X["Sex"].map(binary_map)
X["ExerciseAngina"] = X["ExerciseAngina"].map(binary_map)
X = pd.get_dummies(
    X,
    columns=["ChestPainType", "RestingECG", "ST_Slope"],
    drop_first=True
)
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"

with open(FEATURES_PATH, "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print(f"Feature columns saved to {FEATURES_PATH}")


# Train / Validation / Test split
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25,
    random_state=42, stratify=y_full_train
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Train XGBoost model
model = XGBClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# Validation evaluation
val_preds = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]
val_acc = accuracy_score(y_val, val_preds)
val_auc = roc_auc_score(y_val, val_probs)

print("\nValidation Results")
print("------------------")
print(f"Accuracy: {val_acc:.3f}")
print(f"ROC-AUC : {val_auc:.3f}")

# Final evaluation on test set
test_preds = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)

print("\nTest Results")
print("------------")
print(f"Accuracy: {test_acc:.3f}")
print(f"ROC-AUC : {test_auc:.3f}")

# Save model
MODEL_PATH = MODEL_DIR / "model.bin"

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"\n Model saved to {MODEL_PATH}")