from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parents[2]


def build_expanding_window_folds(
    df: pd.DataFrame,
    date_column: str = "date",
    n_folds: int = 3,
    val_fraction: float = 0.1,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build expanding-window folds per user (month-based) and merge users for each fold.

    Fold schedule is:
      fold 1 train_fraction = 0.5, val_fraction = val_fraction
      fold 2 train_fraction = 0.5 + val_fraction, val_fraction = val_fraction
      ...

    Example with n_folds=3 and val_fraction=0.1:
      fold1 train 50%, val next 10%
      fold2 train 60%, val next 10%
      fold3 train 70%, val next 10%

    Validation end is truncated to available months when needed.
    Same month will never be split across train/val in a fold.
    """
    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column}")
    if "user_id" not in df.columns:
        raise ValueError("Missing required column: user_id")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1")

    working_df = df.copy()
    working_df[date_column] = pd.to_datetime(working_df[date_column])
    working_df = working_df.sort_values(["user_id", date_column]).reset_index(drop=True)
    working_df["_month_start"] = working_df[date_column].dt.to_period("M").dt.to_timestamp()

    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    base_train_fraction = 0.5

    grouped = {user_id: user_df.copy() for user_id, user_df in working_df.groupby("user_id", sort=False)}

    for fold_idx in range(n_folds):
        train_fraction = base_train_fraction + fold_idx * val_fraction
        train_chunks = []
        val_chunks = []

        for _, user_df in grouped.items():
            unique_months = user_df["_month_start"].drop_duplicates().sort_values().tolist()
            n_months = len(unique_months)
            if n_months == 0:
                continue

            train_end = int(np.floor(n_months * train_fraction))
            train_end = max(1, min(train_end, n_months))

            val_start = train_end
            val_end = int(np.floor(n_months * (train_fraction + val_fraction)))
            val_end = max(val_start, min(val_end, n_months))

            train_months = set(unique_months[:train_end])
            val_months = set(unique_months[val_start:val_end])

            user_train = user_df[user_df["_month_start"].isin(train_months)]
            user_val = user_df[user_df["_month_start"].isin(val_months)]

            if not user_train.empty:
                train_chunks.append(user_train)
            if not user_val.empty:
                val_chunks.append(user_val)

        fold_train = (
            pd.concat(train_chunks, ignore_index=True).sort_values(["user_id", date_column]).reset_index(drop=True).copy()
            if train_chunks
            else pd.DataFrame(columns=working_df.columns).copy()
        )
        fold_val = (
            pd.concat(val_chunks, ignore_index=True).sort_values(["user_id", date_column]).reset_index(drop=True).copy()
            if val_chunks
            else pd.DataFrame(columns=working_df.columns).copy()
        )
        if "_month_start" in fold_train.columns:
            fold_train = fold_train.drop(columns=["_month_start"])
        if "_month_start" in fold_val.columns:
            fold_val = fold_val.drop(columns=["_month_start"])
        folds.append((fold_train, fold_val))

    return folds


def example_train_with_expanding_window(
    df: pd.DataFrame,
    target_column: str = "future_expense_7d_sum",
    date_column: str = "date",
    n_folds: int = 3,
    val_fraction: float = 0.1,
) -> pd.DataFrame:
    """
    Example training loop using expanding-window folds.
    Returns a DataFrame with per-fold metrics.
    """
    folds = build_expanding_window_folds(
        df=df,
        date_column=date_column,
        n_folds=n_folds,
        val_fraction=val_fraction,
    )

    exclude_cols = {"user_id", date_column, target_column}
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_columns = [col for col in numeric_cols if col not in exclude_cols]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for training.")

    results = []
    for fold_number, (train_df, val_df) in enumerate(folds, start=1):
        if train_df.empty or val_df.empty:
            results.append(
                {
                    "fold": fold_number,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "mae": np.nan,
                    "rmse": np.nan,
                    "note": "Skipped due to empty train or val split",
                }
            )
            continue

        x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
        y_train = train_df[target_column].to_numpy(dtype=np.float32)
        x_val = val_df[feature_columns].to_numpy(dtype=np.float32)
        y_val = val_df[target_column].to_numpy(dtype=np.float32)

        model = HistGradientBoostingRegressor(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
        )
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)

        mae = mean_absolute_error(y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        results.append(
            {
                "fold": fold_number,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "mae": float(mae),
                "rmse": float(rmse),
                "note": "",
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick runnable example:
    # 1) Load your prepared feature table into df
    # 2) Call example_train_with_expanding_window(df)
    example_path = ROOT_DIR / "ml" / "artifacts" / "features_all.parquet"
    df_example = pd.read_parquet(example_path)
    metrics_df = example_train_with_expanding_window(df_example)
    print(metrics_df)
