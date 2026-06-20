# HeartSense ML — Heart Disease Prediction Web App

A complete Machine Learning web application fulfilling all course project requirements.

## Features
- ✅ Web-based interface (Flask)
- ✅ 4 ML algorithms (Random Forest, Logistic Regression, Decision Tree, Gradient Boosting)
- ✅ Fully self-contained — no external APIs, all models trained locally
- ✅ User input form with 13 clinical features
- ✅ Model comparison dashboard (Accuracy, Precision, Recall, F1, ROC-AUC, Training Time, Confusion Matrix)
- ✅ 800-record synthetic heart disease dataset (Cleveland-style)

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (only needed once)
python train_models.py

# 3. Start the web server
python app.py
```

Then open http://localhost:5000 in your browser.

## Project Structure

```
ml_project/
├── app.py               # Flask application
├── train_models.py      # Model training script
├── requirements.txt
├── models/              # Saved .pkl files + results.json
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── gradient_boosting.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── results.json
└── templates/
    ├── base.html
    ├── index.html       # Prediction page
    └── dashboard.html   # Comparison dashboard
```

## Dataset

Synthetic Cleveland Heart Disease-style dataset (800 records).
Features: age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar,
resting ECG, max heart rate, exercise angina, ST depression, slope, vessels, thalassemia.

## Models & Results

| Model               | Accuracy | F1    | ROC-AUC |
|---------------------|----------|-------|---------|
| Random Forest       | ~80%     | ~80%  | ~87%    |
| Logistic Regression | ~80%     | ~79%  | ~87%    |
| Decision Tree       | ~79%     | ~78%  | ~79%    |
| Gradient Boosting   | ~83%     | ~83%  | ~90%    |

## Pages

- `/` — Patient input form + real-time predictions from all 4 models
- `/dashboard` — Full performance comparison dashboard with charts
