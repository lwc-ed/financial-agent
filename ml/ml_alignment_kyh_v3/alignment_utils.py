import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


COMMON_FEATURES = [
    "daily_expense",
    "daily_income",
    "txn_count",
    "daily_net",
    "dow",
    "is_weekend",
    "day",
    "month",
    "expense_7d_sum",
    "expense_7d_mean",
    "expense_30d_sum",
    "expense_30d_mean",
]


def ensure_datetime(df, col):
    df[col] = pd.to_datetime(df[col])
    return df


def fill_missing_values(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(0)
    return df


def save_feature_schema(cols, path):
    with open(path, "w") as f:
        json.dump(cols, f)


def load_feature_schema(path):
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def split_xy(df, features, target="label"):
    return df[features], df[target]


def train_valid_split_by_time(df, ratio=0.2):
    df = df.sort_values("date")
    split = int(len(df) * (1 - ratio))
    return df[:split], df[split:]