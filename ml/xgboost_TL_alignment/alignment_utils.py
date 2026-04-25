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


def ensure_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def save_feature_schema(feature_cols: List[str], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)


def load_feature_schema(schema_path: Path) -> List[str]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj, output_path: Path) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(input_path: Path):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def split_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "label"
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def train_valid_split_by_time(
    df: pd.DataFrame,
    valid_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - valid_ratio))
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


def train_valid_test_split_by_time(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user chronological 70/15/15 split."""
    train_list, valid_list, test_list = [], [], []
    for _, grp in df.groupby("user_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        train_list.append(grp.iloc[:train_end])
        valid_list.append(grp.iloc[train_end:valid_end])
        test_list.append(grp.iloc[valid_end:])
    train_df = pd.concat(train_list).reset_index(drop=True)
    valid_df = pd.concat(valid_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, valid_df, test_df