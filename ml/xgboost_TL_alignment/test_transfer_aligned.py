import sys
from pathlib import Path

# 把專案根目錄加到 Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from alignment_utils import (
    DATA_DIR,
    RESULT_DIR,
    MODEL_DIR,
    load_feature_schema,
    load_pickle,
)

from ml.output_eval_utils import run_output_evaluation


ROOT = Path(__file__).resolve().parent


def evaluate_model(
    model_path: Path,
    scaler,
    df: pd.DataFrame,
    feature_cols,
    name: str,
):
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔案: {model_path}")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"{name} 評估資料缺少特徵欄位: {missing_features}")

    if "label" not in df.columns or "target" not in df.columns:
        raise ValueError(f"{name} 評估資料缺少 label 或 target 欄位")

    X = df[feature_cols].copy()
    y_raw = df["target"].copy()

    X_sc = scaler.transform(X)

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    preds_log = model.predict(X_sc)
    preds_raw = np.expm1(preds_log)

    mae = mean_absolute_error(y_raw, preds_raw)
    rmse = mean_squared_error(y_raw, preds_raw) ** 0.5

    pred_df = pd.DataFrame()
    if "user_id" in df.columns:
        pred_df["user_id"] = df["user_id"].astype(str).values
    if "date" in df.columns:
        pred_df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").values

    pred_df["y_true"] = y_raw.values
    pred_df["y_pred"] = preds_raw

    pred_df.to_csv(
        RESULT_DIR / f"predictions_{name}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metrics = {
        "model_name": name,
        "mae": float(mae),
        "rmse": float(rmse),
        "rows": int(len(df)),
        "feature_count": int(len(feature_cols)),
    }

    return metrics, pred_df


def build_split_metadata_per_user(df: pd.DataFrame, train_ratio: float = 0.8) -> pd.DataFrame:
    """
    建立共用 evaluator 需要的 split_metadata_df
    必要欄位: user_id, date, split

    修正重點：
    - 不能整份 df 直接 80/20 切
    - 要改成每個 user 各自依日期排序後切 train/test
    - 這樣每個 user 才會至少有 train 資料可算 monthly_available_cash
    """
    required_cols = ["user_id", "date"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"建立 split_metadata_df 時缺少欄位: {missing}")

    work_df = df.copy()
    work_df["user_id"] = work_df["user_id"].astype(str)
    work_df["date"] = pd.to_datetime(work_df["date"])

    parts = []

    for user_id, g in work_df.groupby("user_id", sort=False):
        g = g.sort_values("date").copy()
        n = len(g)

        if n == 1:
            # 只有一筆時，只能放 train
            g["split"] = "train"
        else:
            split_idx = int(n * train_ratio)

            # 至少保證 1 筆 train
            if split_idx < 1:
                split_idx = 1

            # 也盡量保留至少 1 筆 test
            if split_idx >= n:
                split_idx = n - 1

            g["split"] = "test"
            g.iloc[:split_idx, g.columns.get_loc("split")] = "train"

        parts.append(g[["user_id", "date", "split"]])

    split_metadata_df = pd.concat(parts, axis=0).reset_index(drop=True)
    split_metadata_df["date"] = split_metadata_df["date"].dt.strftime("%Y-%m-%d")

    return split_metadata_df


def build_test_df_per_user(df: pd.DataFrame, train_ratio: float = 0.8) -> pd.DataFrame:
    """
    與 split_metadata_df 相同邏輯：
    每個 user 各自依日期排序後切 test。
    """
    work_df = df.copy()
    work_df["user_id"] = work_df["user_id"].astype(str)
    work_df["date"] = pd.to_datetime(work_df["date"])

    test_parts = []

    for user_id, g in work_df.groupby("user_id", sort=False):
        g = g.sort_values("date").copy()
        n = len(g)

        if n <= 1:
            # 沒辦法切 test，跳過
            continue

        split_idx = int(n * train_ratio)
        if split_idx < 1:
            split_idx = 1
        if split_idx >= n:
            split_idx = n - 1

        g_test = g.iloc[split_idx:].copy()
        test_parts.append(g_test)

    if not test_parts:
        raise ValueError("測試集為空，請檢查每位 user 的資料量")

    test_df = pd.concat(test_parts, axis=0).reset_index(drop=True)
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d")
    return test_df


def main():
    own_data_path = DATA_DIR / "own_processed_aligned.csv"
    schema_path = DATA_DIR / "common_features.json"

    base_model_path = MODEL_DIR / "xgb_kaggle_aligned_v2.json"
    finetuned_model_path = MODEL_DIR / "xgb_finetuned_own_aligned_v2.json"
    scaler_path = MODEL_DIR / "feature_scaler_v2.pkl"

    if not own_data_path.exists():
        raise FileNotFoundError(f"找不到資料檔案: {own_data_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"找不到 schema 檔案: {schema_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"找不到 scaler 檔案: {scaler_path}")

    df = pd.read_csv(own_data_path)
    df["user_id"] = df["user_id"].astype(str)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    feature_cols = load_feature_schema(schema_path)
    scaler = load_pickle(scaler_path)

    # 依每個 user 各自切 test
    test_df = build_test_df_per_user(df, train_ratio=0.8)

    if len(test_df) == 0:
        raise ValueError("測試集為空，請檢查資料量")

    print("========== TEST INFO ==========")
    print(f"Full data size: {df.shape}")
    print(f"Test size: {test_df.shape}")
    print("Feature columns:")
    print(feature_cols)
    print("Test label summary (log space):")
    print(test_df["label"].describe())
    print("Test target summary (raw space):")
    print(test_df["target"].describe())

    results = []
    prediction_dfs = {}

    if base_model_path.exists():
        metrics_base, pred_df_base = evaluate_model(
            base_model_path, scaler, test_df, feature_cols, "base_aligned"
        )
        results.append(metrics_base)
        prediction_dfs["base_aligned"] = pred_df_base
    else:
        print(f"[WARN] base model not found: {base_model_path}")

    if finetuned_model_path.exists():
        metrics_ft, pred_df_ft = evaluate_model(
            finetuned_model_path, scaler, test_df, feature_cols, "finetuned_aligned"
        )
        results.append(metrics_ft)
        prediction_dfs["finetuned_aligned"] = pred_df_ft
    else:
        print(f"[WARN] finetuned model not found: {finetuned_model_path}")

    output_json = RESULT_DIR / "transfer_test_metrics_v2.json"
    output_txt = RESULT_DIR / "aligned_results_v2.txt"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_txt, "w", encoding="utf-8") as f:
        for item in results:
            f.write(
                f"Model: {item['model_name']}\n"
                f"Rows: {item['rows']}\n"
                f"Feature count: {item['feature_count']}\n"
                f"MAE: {item['mae']:.4f}\n"
                f"RMSE: {item['rmse']:.4f}\n\n"
            )

    print(f"[OK] saved evaluation -> {output_json}")
    print(f"[OK] saved summary -> {output_txt}")
    print(results)

    # 用 per-user split 建正式 metadata
    split_metadata_df = build_split_metadata_per_user(df, train_ratio=0.8)

    # 正式 spec 輸出：優先 finetuned，沒有才退回 base
    if "finetuned_aligned" in prediction_dfs:
        official_prediction_input_df = prediction_dfs["finetuned_aligned"].copy()
        official_model_source = "finetuned_aligned"
    elif "base_aligned" in prediction_dfs:
        official_prediction_input_df = prediction_dfs["base_aligned"].copy()
        official_model_source = "base_aligned"
    else:
        raise RuntimeError("沒有可用模型可做正式輸出，base/finetuned 都不存在")

    # Debug：檢查 prediction users 是否都在 train 出現過
    train_users = set(
        split_metadata_df.loc[split_metadata_df["split"] == "train", "user_id"].astype(str)
    )
    pred_users = set(official_prediction_input_df["user_id"].astype(str))
    missing_users = sorted(pred_users - train_users)
    print("Users in prediction but not in train:", missing_users)

    print(f"[INFO] official spec output uses: {official_model_source}")

    run_output_evaluation(
        model_name="xgboost_TL_alignment",
        prediction_input_df=official_prediction_input_df,
        split_metadata_df=split_metadata_df,
    )

    print("[OK] spec output generated -> ml/model_outputs/xgboost_TL_alignment/")


if __name__ == "__main__":
    main()