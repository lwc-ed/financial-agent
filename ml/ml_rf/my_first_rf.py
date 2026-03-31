import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 設定路徑
ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "ml_hgbr" / "artifacts"
TRAIN_PATH = ARTIFACT_DIR / "features_train.parquet"
VAL_PATH = ARTIFACT_DIR / "features_val.parquet"
TEST_PATH = ARTIFACT_DIR / "features_test.parquet"

# 設定 RF 專屬的輸出路徑
OUTPUT_DIR = ROOT_DIR / "ml_rf" / "rf_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. 定義特徵與目標
TARGET_COLUMN = "future_expense_7d_sum"
DROP_COLUMNS = {"user_id", "date", "month_start"}

def get_all_features(df: pd.DataFrame, target_col: str) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [col for col in numeric_cols if col not in DROP_COLUMNS and col != target_col]

NAIVE_BASELINE_FEATURE = "expense_7d_sum"
MOVING_AVG_BASELINE_FEATURE = "expense_30d_mean"

# 3. 準備資料的函數
def prepare_xy(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_columns].astype(np.float32).to_numpy()
    y = df[target_column].astype(np.float32).to_numpy()
    return x, y

def compute_regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    return {"mae": float(mae), "rmse": float(np.sqrt(mse))}

def main():
    print("🌲 開始執行隨機森林 (Random Forest) 訓練！")

    # 讀取黃金資料
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print(f"✅ 成功讀取資料！訓練集: {len(train_df)} 筆, 測試集: {len(test_df)} 筆")

    FEATURE_COLUMNS = get_all_features(train_df, TARGET_COLUMN)
    print(f"🔍 這次我們火力全開，使用了 {len(FEATURE_COLUMNS)} 個特徵來訓練！")

    x_train, y_train = prepare_xy(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    x_val, y_val = prepare_xy(val_df, FEATURE_COLUMNS, TARGET_COLUMN)
    x_test, y_test = prepare_xy(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    # 4. 訓練隨機森林模型 (自動尋寶 + Log 魔法版)
    print("🌲 開始自動尋寶 (含 Log 魔法)！讓電腦自己測試多組參數...")
    
    candidate_configs = [
        {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 2, "max_features": "log2"},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5, "max_features": None},
        {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 10, "max_features": 0.5},
        {"n_estimators": 400, "max_depth": 15, "min_samples_leaf": 2, "max_features": "sqrt"}
    ]

    best_val_mae = float("inf")
    best_model = None
    best_val_metrics = None

    # 🌟 魔法開始：把訓練集的真實答案 y 取對數 (壓縮它)
    y_train_log = np.log1p(y_train)

    for i, config in enumerate(candidate_configs, 1):
        model = RandomForestRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=42,
            n_jobs=-1
        )
        
        # 讓模型學習「壓縮後」的金額
        model.fit(x_train, y_train_log)
        
        # 模型預測出來的也是壓縮版的金額
        val_preds_log = model.predict(x_val)
        
        # 🌟 魔法結束：把預測結果「解壓縮」回原本的真實金額
        val_preds = np.expm1(val_preds_log)
        
        # 用真實金額來算誤差
        val_mae = mean_absolute_error(y_val, val_preds)
        
        print(f"嘗試第 {i} 組參數: 模擬考誤差={val_mae:.2f}")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_val_metrics = compute_regression_metrics(y_val, val_preds)

    print("🏆 尋寶結束！我們選出了最強的隨機森林模型！")
    
    rf_model = best_model
    val_metrics = best_val_metrics

    # 5. 在測試集上評估 (期末考也要解壓縮！)
    test_preds_log = rf_model.predict(x_test)
    test_predictions = np.expm1(test_preds_log)
    model_metrics = compute_regression_metrics(y_test, test_predictions)

    # 計算 Baseline
    naive_predictions = test_df[NAIVE_BASELINE_FEATURE].astype(np.float32).to_numpy()
    moving_avg_predictions = test_df[MOVING_AVG_BASELINE_FEATURE].astype(np.float32).to_numpy() * 7.0
    naive_metrics = compute_regression_metrics(y_test, naive_predictions)
    moving_avg_metrics = compute_regression_metrics(y_test, moving_avg_predictions)

    beat_moving_avg = model_metrics["mae"] < moving_avg_metrics["mae"] and model_metrics["rmse"] < moving_avg_metrics["rmse"]

    # 6. 輸出報告
    report_lines = [
        "Training Run Summary RF (Log Edition)",
        "model_name: random_forest_log_v1",
        "feature_set: all_numeric",
        f"target_column: {TARGET_COLUMN}",
        f"best_val_metric: {val_metrics['mae']:.6f}",
        f"test_mae: {model_metrics['mae']:.6f}",
        f"test_rmse: {model_metrics['rmse']:.6f}",
        "",
        "Dataset Sizes",
        f"train_rows: {len(train_df)}",
        f"val_rows: {len(val_df)}",
        f"test_rows: {len(test_df)}",
        f"feature_count: {len(FEATURE_COLUMNS)}",
        "",
        "Selected Features",
    ]
    report_lines.extend([f"- {feature}" for feature in FEATURE_COLUMNS])
    report_lines.extend(
        [
            "",
            "Baselines",
            f"naive_7d_sum mae: {naive_metrics['mae']:.6f}",
            f"naive_7d_sum rmse: {naive_metrics['rmse']:.6f}",
            f"moving_avg_30d_x7 mae: {moving_avg_metrics['mae']:.6f}",
            f"moving_avg_30d_x7 rmse: {moving_avg_metrics['rmse']:.6f}",
            "",
            f"beat_moving_avg_30d_x7: {beat_moving_avg}",
            "",
            "Artifacts",
            f"best_model: {OUTPUT_DIR / 'best_rf_model.pkl'}",
            f"report_txt: {OUTPUT_DIR / 'training_report_rf.txt'}",
        ]
    )
    
    report_path = OUTPUT_DIR / "training_report_rf.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # 存檔模型
    model_path = OUTPUT_DIR / "best_rf_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"model": rf_model, "features": FEATURE_COLUMNS}, f)

    print(f"🎉 執行完成！")
    print(f"📊 你的 RF test_mae: {model_metrics['mae']:.6f}")
    print(f"📄 報告已經儲存在: {report_path}")

if __name__ == "__main__":
    main()