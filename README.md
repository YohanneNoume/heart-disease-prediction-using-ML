# Heart Disease Prediction System (ML + Cloud Deployment)

## Problem Description

Cardiovascular diseases are one of the leading causes of death worldwide. Early detection of heart disease can significantly improve treatment outcomes and reduce mortality.

This project implements an **end-to-end machine learning system** that predicts whether a patient is at risk of heart disease based on clinical attributes.

The system:

* Analyzes patient medical data
* Predicts heart disease risk using a trained ML model
* Exposes predictions via a **Flask REST API**
* Is **containerized with Docker**
* Is **deployed to the cloud using Render**
* Is **tested on Postman** 

This solution can be used as a decision-support tool or integrated into healthcare applications for early risk assessment.

---

## Dataset

* **Dataset:** Heart Disease Dataset
* The dataset was acquired from kaggle: fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [December 2025] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.
* **Location:** `data/heart.csv`

### Features include:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-induced Angina
* ST Depression (Oldpeak)
* ST Slope

### Target Variable:

* `HeartDisease`

  * `0` → No Heart Disease
  * `1` → Heart Disease

The dataset is included in the repository to ensure **full reproducibility**.

---

## Exploratory Data Analysis (EDA)

EDA is performed in **`notebook.ipynb`** and includes:

* Dataset overview and data types
* Missing value and duplicate checks
* Distribution analysis of numerical features
* Target variable distribution analysis
* Feature correlation analysis
* Feature importance analysis (tree-based models)
* Confusion matrix and ROC curve analysis

These analyses guided model selection and feature handling.

---

## Model Training & Selection

Multiple machine learning models were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* **XGBoost (final model)**

### Final Model: XGBoost

XGBoost was selected due to:

* Highest ROC-AUC score
* Strong precision–recall balance
* Robust handling of non-linear relationships

<img width="985" height="766" alt="image" src="https://github.com/user-attachments/assets/68707437-5691-4538-8a0f-702bc79d0fa9" />


### Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrices and ROC curves

---

## Project Structure

```
heart-disease-prediction-using-ML/
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── model.bin
│   └── feature_columns.pkl
│
├── notebook.ipynb
├── train.py
├── predict.py
├── predict_test.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

---

## Reproducibility

## 1. Clone the repository
```bash
git clone https://github.com/YohanneNoume/heart-disease-prediction-using-ML.git
cd heart-disease-prediction-using-ML
```

### 2. Create a virtual environment

This project was developed using a Conda environment, but it can also be run
using a standard Python virtual environment (`venv`).

### Option 1: Using Conda 

```bash
conda create -n heart-project python=3.11
conda activate heart-project
pip install -r requirements.txt
```
### Option 2: Using Python env
```bash
# Create venv
python -m venv venv

# To activate venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Trains the final machine learning model using the prepared dataset, evaluates performance, and saves the trained model and feature configuration to disk.

```bash
python train.py
```

This generates:

* `models/model.bin`
* `models/feature_columns.pkl`

---

## Model Deployment (Flask API)

The trained model is served using **Flask** and **Gunicorn**.
Loads the trained model and serves it through a Flask REST API that accepts patient data and returns heart disease risk predictions.

### Run the API locally

```bash
python predict.py
```

---

## Testing the Model 

Sends a sample patient record through the trained model to verify predictions locally without using the API.
A test client script is provided.

```bash
python predict_test.py
```

### Sample Output
<img width="686" height="177" alt="image" src="https://github.com/user-attachments/assets/0c8104fd-4cc8-4a66-acde-8e2fecb9c48b" />

---


## Docker Containerization

### Build the Docker image

```bash
docker build -t heart-disease-predictor .
```

### Run the container

```bash
docker run -p 9696:9696 heart-disease-predictor
```

The API will be accessible at:

```
http://localhost:9696
```

Gunicorn is used as the production WSGI server.

### Output
<img width="1594" height="1029" alt="image" src="https://github.com/user-attachments/assets/8a88c883-8e1f-4270-8bc4-eb574ff4c31e" />

---

## Cloud Deployment
The application is deployed using Render (Docker-based service).

Live URL
https://heart-disease-prediction-using-ml-7xw3.onrender.com/


Deployment validation includes:

* API accessible via browser
* /predict endpoint tested via Postman
* Docker container running successfully

<img width="1053" height="600" alt="image" src="https://github.com/user-attachments/assets/d2f93c46-c23d-4dfe-9db0-88b2a617959e" />

---

## API Testing with Postman

The deployed API was tested using **Postman** to verify correct cloud deployment and prediction behavior.

### Step 1: Open Postman

Download and open Postman from:
[https://www.postman.com/downloads/](https://www.postman.com/downloads/)

---

### Step 2: Create a New Request

* Click **New → HTTP Request**
* Set the request method to **POST**
* Enter the deployed API URL:

```
https://heart-disease-prediction-using-ml-7xw3.onrender.com/predict
```

---

### Step 3: Configure Headers

Go to the **Headers** tab and add:

| Key          | Value            |
| ------------ | ---------------- |
| Content-Type | application/json |

---

### Step 4: Add Request Body

1. Go to the **Body** tab
2. Select **raw**
3. Choose **JSON** from the dropdown
4. Paste the following example JSON:

```json
{
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
```

---

### Step 5: Send the Request

Click **Send**.

---

### Step 6: Expected Response

If the API is running correctly, you should receive a JSON response similar to:

```json
{
  "heart_disease_prediction": 1,
  "heart_disease_probability": 0.725,
  "risk_label": "High Risk",
  "risk_message": "Patient is at risk of heart disease"
}
```

---

### Notes

* `heart_disease_prediction`

  * `0` → No Heart Disease
  * `1` → Heart Disease
* The `risk_label` and `risk_message` fields make the output **human-readable**.
* Screenshots from Postman are included in the submission as proof of successful cloud deployment.
<img width="1371" height="952" alt="image" src="https://github.com/user-attachments/assets/041734a3-269f-4c92-bffd-1f2b624a3fdb" />

---























