# HDP SYSTEM — Heart Disease Prediction

An end-to-end machine learning project: clean heart disease data, train Logistic Regression,
Random Forest and XGBoost, pick the best model, and serve it in an interactive Streamlit app.
Predictions can be saved to a database with SQLAlchemy and exported as CSV.

## Features

- **Live prediction:** the risk gauge updates as you change any input.
- **Sample patients:** load a low-risk or high-risk example in one click.
- **Save to history:** log a prediction to the database on demand.
- **Model performance tab:** model comparison, confusion matrix and feature importance.
- **History tab:** saved predictions with summary metrics and CSV download.

## Project structure

```
HDP SYSTEM/
├── app.py                 # Streamlit app (Predict, Model performance, History, About)
├── requirements.txt
├── README.md
├── data/raw/heart.csv     # Dataset: UCI Heart Disease, 303 rows
├── models/                # best_model.joblib (created by training)
├── reports/               # metrics.json (created by training)
└── src/
    ├── data.py            # Steps 1-2: load and clean the data
    ├── train.py           # Steps 3-4: train, evaluate, save the best model
    ├── predict.py         # Model loading, risk levels, suggestions, feature importance
    └── db.py              # SQLAlchemy prediction log
```

## Method (same order as the video)

1. **Data:** `data/raw/heart.csv`, UCI Heart Disease (Cleveland format, 13 features plus `target`).
2. **Clean:** check for missing values and duplicates, fix data types, keep valid labels (`src/data.py`).
3. **Train:** Logistic Regression as the baseline, then Random Forest and XGBoost (`src/train.py`).
4. **Evaluate:** 5-fold cross-validation ranked by **F2 score**, which weights recall more than precision,
   because a missed sick patient is worse than a false alarm. Test-set metrics and the confusion matrix are reported.
5. **Serve:** the best model is used by the Streamlit app (`app.py`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

From the project root:

```bash
python -m src.train
```

This prints the cleaning report, the model comparison and the confusion matrix, and saves
`models/best_model.joblib` and `reports/metrics.json`. The included model is already trained, so this step is optional.

## Run the app

```bash
streamlit run app.py
```

Open (https://hdp-system.streamlit.app/) in your browser.

## Database

By default, saved predictions go to `predictions.db` (SQLite), so nothing else is needed.
To use PostgreSQL, set `DATABASE_URL` before starting the app and install `psycopg2-binary`:

```bash
export DATABASE_URL="postgresql+psycopg2://user:password@localhost:5432/hdp"
```

## Notes and limitations

- The dataset is small (303 rows), so scores can move by a few points between runs. Logistic Regression and
  Random Forest score similarly, so do not read too much into the ranking.
- The risk cut-offs (below 30% low, 30% to 60% moderate, above 60% high) are a display heuristic, not clinical thresholds.
- Categorical codes (`cp`, `restecg`, `slope`, `thal`, `ca`) are shown as the raw dataset numbers. Confirm their meanings
  against the dataset documentation before showing human-readable labels.
- This is an educational project. It is not a medical device and must not be used for diagnosis.

## Data sources

- UCI Machine Learning Repository, Heart Disease: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- Kaggle mirror: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
- The file in this project comes from a GitHub mirror of the UCI Cleveland data (`kb22/Heart-Disease-Prediction`).
