"""Prediction helpers: load the model and metrics, risk levels, suggestions, feature importance."""
import json

import joblib
import numpy as np
import pandas as pd

from src.data import FEATURES, ROOT_DIR

MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels",
    "thal": "Thal",
}


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_metrics(path=METRICS_PATH):
    """Return the metrics report written by training, or None if it does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def risk_level(probability):
    # Simple cut-offs for display. They are a heuristic, not clinical thresholds.
    if probability < 0.3:
        return "Low Risk of Heart Disease"
    if probability < 0.6:
        return "Moderate Risk of Heart Disease"
    return "High Risk of Heart Disease"


def predict_risk(model, inputs):
    """Return (probability_of_heart_disease, risk_label)."""
    row = pd.DataFrame([inputs], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    return probability, risk_level(probability)


def suggestions(inputs):
    tips = []
    if inputs["chol"] >= 240:
        tips.append(
            "Cholesterol is above the high-risk threshold — a heart-healthy diet "
            "and consistent exercise could help."
        )
    elif inputs["chol"] >= 200:
        tips.append("Cholesterol is borderline high — keep an eye on it with regular checkups.")

    if inputs["trestbps"] >= 140:
        tips.append(
            "Resting blood pressure is high — less salt, regular activity and "
            "regular BP checks can help."
        )
    if inputs["fbs"] == 1:
        tips.append("Fasting blood sugar is above 120 mg/dl — ask a doctor about a diabetes screening.")
    if inputs["exang"] == 1:
        tips.append("Exercise-induced chest pain was reported — talk to a doctor before intense exercise.")
    if inputs["oldpeak"] >= 2.0:
        tips.append("ST depression (oldpeak) is high — worth reviewing with a cardiologist.")

    if not tips:
        tips.append(
            "None of the values you entered is a major flag. Keep up regular "
            "checkups and a healthy lifestyle."
        )
    return tips


def feature_importance(model):
    """Return a DataFrame (feature, importance) sorted for a horizontal bar chart, or None."""
    estimator = model[-1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_).ravel())
    else:
        return None
    df = pd.DataFrame({
        "feature": [FEATURE_LABELS[f] for f in FEATURES],
        "importance": values,
    })
    return df.sort_values("importance", ascending=True).reset_index(drop=True)
