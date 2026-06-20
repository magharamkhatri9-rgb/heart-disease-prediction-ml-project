"""
Heart Disease Prediction - Model Training Script
Trains Random Forest and Logistic Regression on the Cleveland Heart Disease dataset.
"""
import numpy as np
import pandas as pd
import json, time, joblib, os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

# ── Synthetic Cleveland-style dataset (500+ records, no external API) ──
np.random.seed(42)
N = 800

age        = np.random.randint(29, 78, N)
sex        = np.random.randint(0, 2, N)
cp         = np.random.randint(0, 4, N)
trestbps   = np.random.randint(90, 200, N)
chol       = np.random.randint(126, 565, N)
fbs        = np.random.randint(0, 2, N)
restecg    = np.random.randint(0, 3, N)
thalach    = np.random.randint(71, 202, N)
exang      = np.random.randint(0, 2, N)
oldpeak    = np.round(np.random.uniform(0, 6.2, N), 1)
slope      = np.random.randint(0, 3, N)
ca         = np.random.randint(0, 4, N)
thal       = np.random.choice([1, 2, 3], N)

# Risk score → target (disease = 1)
risk = (
    (age > 55) * 0.25 +
    (sex == 1) * 0.15 +
    (cp == 0) * 0.20 +
    (trestbps > 140) * 0.10 +
    (chol > 240) * 0.08 +
    (exang == 1) * 0.15 +
    (oldpeak > 2) * 0.10 +
    (ca > 0) * 0.12 +
    np.random.uniform(-0.2, 0.2, N)
)
target = (risk > 0.45).astype(int)

df = pd.DataFrame({
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
    'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
    'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
    'thal': thal, 'target': target
})

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

feature_names = list(X.columns)

# ── Train 4 models ─────────────────────────────────────────────────────
MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
os.makedirs("models", exist_ok=True)

for name, model in MODELS.items():
    uses_scale = name in ("Logistic Regression",)
    Xtr = X_train_sc if uses_scale else X_train.values
    Xte = X_test_sc  if uses_scale else X_test.values

    t0 = time.time()
    model.fit(Xtr, y_train)
    train_time = round(time.time() - t0, 3)

    y_pred  = model.predict(Xte)
    y_prob  = model.predict_proba(Xte)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        "accuracy":      round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision":     round(precision_score(y_test, y_pred) * 100, 2),
        "recall":        round(recall_score(y_test, y_pred)    * 100, 2),
        "f1":            round(f1_score(y_test, y_pred)        * 100, 2),
        "roc_auc":       round(roc_auc_score(y_test, y_prob)   * 100, 2),
        "train_time":    train_time,
        "confusion_matrix": cm.tolist(),
    }
    safe = name.replace(" ", "_").lower()
    joblib.dump(model, f"models/{safe}.pkl")
    print(f"✓ {name}: acc={results[name]['accuracy']}%  time={train_time}s")

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

with open("models/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll models saved to models/")
