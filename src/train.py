"""Steps 3-4 (from the video): train Logistic Regression (baseline), Random Forest
and XGBoost, then evaluate them. Saves the best model for the Streamlit app.

Run from the project root:  python -m src.train
"""
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data import FEATURES, ROOT_DIR, TARGET, load_and_clean

RANDOM_STATE = 42
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

# Accuracy alone is not enough here. F2 weights recall (catching sick patients)
# more heavily than precision, because a missed case (false negative) is worse
# than a false alarm.
SCORERS = {
    "f2": make_scorer(fbeta_score, beta=2),
    "recall": "recall",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


def build_models():
    return {
        # Baseline: simple and easy to explain. Scaling helps it weigh features fairly.
        "Logistic Regression (baseline)": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "f2": float(fbeta_score(y_test, y_pred, beta=2)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    df, clean_report = load_and_clean()
    print("== Step 2: cleaning ==")
    for key, value in clean_report.items():
        print(f"  {key}: {value}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results, test_results, fitted = {}, {}, {}

    print("\n== Step 3-4: training and evaluation ==")
    for name, model in build_models().items():
        print(f"  training {name} ...")
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORERS)
        cv_results[name] = {m: float(np.mean(scores[f"test_{m}"])) for m in SCORERS}
        model.fit(X_train, y_train)
        fitted[name] = model
        test_results[name] = evaluate(model, X_test, y_test)

    best_name = max(cv_results, key=lambda n: cv_results[n]["f2"])
    best_model = fitted[best_name]

    header = (
        f"{'Model':<32}{'CV F2':>8}{'CV recall':>11}{'Test acc':>10}"
        f"{'Test recall':>13}{'Test AUC':>10}{'Missed (FN)':>13}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name in cv_results:
        cv_r, t = cv_results[name], test_results[name]
        print(
            f"{name:<32}{cv_r['f2']:>8.3f}{cv_r['recall']:>11.3f}{t['accuracy']:>10.3f}"
            f"{t['recall']:>13.3f}{t['roc_auc']:>10.3f}{t['confusion_matrix']['fn']:>13d}"
        )

    cm = test_results[best_name]["confusion_matrix"]
    print(f"\nBest model (highest CV F2): {best_name}")
    print(f"Confusion matrix on the test set ({best_name}):")
    print(f"                   Predicted 0   Predicted 1")
    print(f"  Actual 0 (no)    {cm['tn']:>11}   {cm['fp']:>11}")
    print(f"  Actual 1 (yes)   {cm['fn']:>11}   {cm['tp']:>11}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    report = {
        "cleaning": clean_report,
        "selection_metric": "mean 5-fold CV F2 (recall weighted more than precision)",
        "best_model": best_name,
        "cross_validation": cv_results,
        "test": test_results,
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved model   -> {model_path}")
    print(f"Saved metrics -> {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
