"""
Heart Disease Prediction - Flask Application
"""
import json
import joblib
import numpy as np
import json, joblib, numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load saved artefacts ───────────────────────────────────────────────
scaler        = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

MODELS = {
    "Random Forest":      joblib.load("models/random_forest.pkl"),
    "Logistic Regression":joblib.load("models/logistic_regression.pkl"),
    "Decision Tree":      joblib.load("models/decision_tree.pkl"),
    "Gradient Boosting":  joblib.load("models/gradient_boosting.pkl"),
}
SCALED_MODELS = {"Logistic Regression"}

with open("models/results.json") as f:
    MODEL_RESULTS = json.load(f)

# Feature metadata for the UI
FEATURE_META = {
    "age":      {"label": "Age",                    "type": "number", "min": 1,   "max": 120, "step": 1,   "default": 55},
    "sex":      {"label": "Sex",                    "type": "select", "options": [("0","Female"),("1","Male")], "default": "1"},
    "cp":       {"label": "Chest Pain Type",        "type": "select", "options": [("0","Typical Angina"),("1","Atypical Angina"),("2","Non-anginal"),("3","Asymptomatic")], "default": "0"},
    "trestbps": {"label": "Resting Blood Pressure", "type": "number", "min": 80,  "max": 220, "step": 1,   "default": 130, "unit": "mm Hg"},
    "chol":     {"label": "Serum Cholesterol",      "type": "number", "min": 100, "max": 600, "step": 1,   "default": 240, "unit": "mg/dl"},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl","type":"select","options":[("0","No"),("1","Yes")], "default": "0"},
    "restecg":  {"label": "Resting ECG Results",    "type": "select", "options": [("0","Normal"),("1","ST-T Abnormality"),("2","Left Ventricular Hypertrophy")], "default": "0"},
    "thalach":  {"label": "Max Heart Rate Achieved","type": "number", "min": 60,  "max": 220, "step": 1,   "default": 150, "unit": "bpm"},
    "exang":    {"label": "Exercise Induced Angina","type": "select", "options": [("0","No"),("1","Yes")], "default": "0"},
    "oldpeak":  {"label": "ST Depression",          "type": "number", "min": 0,   "max": 8,   "step": 0.1, "default": 1.0},
    "slope":    {"label": "Slope of Peak ST",       "type": "select", "options": [("0","Upsloping"),("1","Flat"),("2","Downsloping")], "default": "1"},
    "ca":       {"label": "Major Vessels (0–3)",    "type": "select", "options": [("0","0"),("1","1"),("2","2"),("3","3")], "default": "0"},
    "thal":     {"label": "Thalassemia",            "type": "select", "options": [("1","Normal"),("2","Fixed Defect"),("3","Reversible Defect")], "default": "2"},
}

@app.route("/")
def index():
    return render_template("index.html", features=FEATURE_META)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        row = np.array([[float(data[f]) for f in feature_names]])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    row_sc = scaler.transform(row)
    predictions = {}
    for name, model in MODELS.items():
        X = row_sc if name in SCALED_MODELS else row
        pred  = int(model.predict(X)[0])
        prob  = float(model.predict_proba(X)[0][1])
        predictions[name] = {"prediction": pred, "probability": round(prob * 100, 1)}

    return jsonify(predictions)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", results=MODEL_RESULTS)

@app.route("/api/results")
def api_results():
    return jsonify(MODEL_RESULTS)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
