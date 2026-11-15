"""Steps 1-2 (from the video): load the raw heart CSV and clean it."""
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "heart.csv"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"  # 1 = heart disease, 0 = no heart disease


def load_and_clean(path=RAW_DATA_PATH):
    """Load the CSV, fix types, drop missing rows and duplicates.

    Returns (clean_dataframe, report_dict) so the steps can be printed/logged.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig also strips a stray BOM
    raw.columns = [c.strip().lower() for c in raw.columns]

    required = FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {path}: {missing_cols}")

    df = raw[required].apply(pd.to_numeric, errors="coerce")
    report = {"rows_raw": len(raw), "missing_cells": int(df.isna().sum().sum())}

    df = df.dropna()
    report["rows_dropped_missing"] = report["rows_raw"] - len(df)

    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    df[TARGET] = df[TARGET].astype(int)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    report["rows_clean"] = len(df)

    return df, report
