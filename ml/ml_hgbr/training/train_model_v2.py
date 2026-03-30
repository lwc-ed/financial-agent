# train_model_v2.py
# 功能：
# 1. 讀取 train / val / test 三份特徵資料
# 2. 根據指定的 feature set 選出要訓練的欄位
# 3. 使用 HistGradientBoostingRegressor (HGBR) 訓練多組參數
# 4. 依 validation set 表現選出最佳模型
# 5. 在 test set 上評估模型，並與 baseline 做比較
# 6. 輸出模型、評估指標、預測結果與訓練報告
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 專案路徑設定
# ROOT_DIR: 專案根目錄
# ML_DIR: ml_hgbr 資料夾
# ARTIFACT_DIR: 模型輸出、特徵檔、報表存放位置
ROOT_DIR = Path(__file__).resolve().parents[2]
ML_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ML_DIR / "artifacts"
DEFAULT_TRAIN_PATH = ARTIFACT_DIR / "features_train.parquet"
DEFAULT_VAL_PATH = ARTIFACT_DIR / "features_val.parquet"
DEFAULT_TEST_PATH = ARTIFACT_DIR / "features_test.parquet"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR

# 目標欄位候選名稱：因為不同版本的前處理可能使用不同命名
TARGET_CANDIDATES = ["future_7d_expense", "future_expense_7d_sum"]

# 這些欄位不作為模型特徵：
# user_id / date / month_start 屬於識別或時間定位資訊，不直接餵進模型
DROP_COLUMNS = {"user_id", "date", "month_start"}

# baseline 用來跟正式模型比較，確認模型是否真的比簡單方法更好
# naive baseline: 直接用過去 7 天總支出當成未來 7 天預測
# moving average baseline: 用過去 30 天平均日支出 * 7 當成未來 7 天預測
NAIVE_BASELINE_FEATURE = "expense_7d_sum"
MOVING_AVG_BASELINE_FEATURE = "expense_30d_mean"

# 修改特徵最主要就是改這裡：
# 1. 若你想調整現有 preset，就直接增減下面 list 裡的欄位名稱
# 2. 若你想新增一組新版本，照樣新增一個 key，例如 "compact_v2": [...]
# 3. 若你想完全手動指定，不改這裡也可以，執行時用：
#    --feature-set custom --feature-columns col1,col2,col3
FEATURE_PRESETS = {
    # user_selected_v1: 目前手動挑選的特徵組合，用來測試精簡特徵版本
    "user_selected_v1": [
        "daily_expense",
        "expense_30d_sum",
        "expense_7d_mean",
        "has_expense",
        "has_income",
        "net_30d_sum",
        "txn_30d_sum"
    ],
}


# 解析命令列參數：讓這支程式可以從 terminal 指定資料路徑、特徵組合、random seed 等設定
def parse_args() -> argparse.Namespace:
    # 建立命令列參數解析器
    parser = argparse.ArgumentParser(
        description="Train model version 2 with a controlled feature subset instead of all 22 features."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH, help="Train parquet path")
    parser.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH, help="Validation parquet path")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="Test parquet path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact output directory")
    parser.add_argument(
        "--target-column",
        default=None,
        help="Regression target column. Defaults to auto-detect from future_7d_expense/future_expense_7d_sum.",
    )
    parser.add_argument(
        "--feature-set",
        default="compact_v1",
        choices=sorted(FEATURE_PRESETS.keys()) + ["custom", "all_numeric"],
        help="Feature preset to use for model v2.",
    )
    parser.add_argument(
        "--feature-columns",
        default="",
        help="Comma-separated feature columns. Only used when --feature-set custom.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    # 回傳使用者在命令列輸入的參數
    return parser.parse_args()


# 決定要預測的 target 欄位
# 若使用者有手動指定 --target-column，就優先使用
# 否則自動從 TARGET_CANDIDATES 裡找三份資料都同時存在的欄位名稱
def resolve_target_column(dataframes: list[pd.DataFrame], requested_target: str | None) -> str:
    # 情況 1：使用者手動指定 target 欄位
    if requested_target is not None:
        missing = [index for index, df in enumerate(dataframes) if requested_target not in df.columns]
        if missing:
            raise ValueError(f"Target column {requested_target!r} is missing from dataset indexes {missing}.")
        return requested_target

    # 情況 2：未手動指定，則依序嘗試候選名稱
    for candidate in TARGET_CANDIDATES:
        if all(candidate in df.columns for df in dataframes):
            return candidate

    raise ValueError(
        "Unable to detect target column. Provide --target-column explicitly. "
        f"Tried {TARGET_CANDIDATES}."
    )


# 從資料中找出所有可用的數值型特徵
# 會排除 target 欄位與 DROP_COLUMNS 中不應作為特徵的欄位
def list_all_numeric_features(df: pd.DataFrame, target_column: str) -> list[str]:
    # 合併所有要排除的欄位名稱
    excluded_columns = DROP_COLUMNS | {target_column}
    numeric_columns = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [column for column in numeric_columns if column not in excluded_columns]


# 根據命令列參數決定本次訓練要使用哪些特徵欄位
# 支援三種模式：
# 1. all_numeric：使用所有數值欄位
# 2. custom：執行時手動傳入欄位名稱
# 3. preset：使用 FEATURE_PRESETS 事先定義好的特徵組合
def resolve_feature_columns(train_df: pd.DataFrame, target_column: str, args: argparse.Namespace) -> list[str]:
    # 先整理出 train_df 中所有可用的數值型特徵
    all_numeric = list_all_numeric_features(train_df, target_column)
    if args.feature_set == "all_numeric":
        # 直接使用所有數值欄位，等同於目前舊版 train_model.py 的做法
        selected = all_numeric
    elif args.feature_set == "custom":
        # 想臨時測試特徵組合時，用命令列傳入，不必改程式碼
        selected = [column.strip() for column in args.feature_columns.split(",") if column.strip()]
        if not selected:
            raise ValueError("--feature-columns is required when --feature-set custom.")
    else:
        # 想固定保存一組可重複使用的特徵組合，就到上面的 FEATURE_PRESETS 修改
        selected = FEATURE_PRESETS[args.feature_set]

    # 檢查選到的特徵是否真的存在於訓練資料中
    missing = [column for column in selected if column not in train_df.columns]
    if missing:
        raise ValueError(f"Selected feature columns are missing from training data: {missing}")

    return selected


# 將 DataFrame 轉成模型可直接使用的 X / y numpy array
# X = 特徵矩陣，y = 預測目標
def prepare_xy(df: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[np.ndarray, np.ndarray]:
    # 轉成 float32 可減少記憶體使用，也方便模型訓練
    x = df[feature_columns].astype(np.float32).to_numpy()
    y = df[target_column].astype(np.float32).to_numpy()
    # 訓練前先檢查是否有 NaN，避免模型訓練時直接報錯或學到有問題的資料
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError("NaN detected in features or target. Clean the source features before training.")
    return x, y


# 計算迴歸模型常用評估指標
# MAE: 平均絕對誤差，越小越好
# RMSE: 均方根誤差，對大誤差更敏感，越小越好
def compute_regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(actual, prediction)
    mse = mean_squared_error(actual, prediction)
    return {
        "mae": float(mae),
        "rmse": float(np.sqrt(mse)),
    }


# 將 Python 物件序列化存成 pickle 檔，方便之後載入最佳模型
def save_pickle(payload: dict, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(payload, file)


# 把不同模型的 test set 預測結果整理成 long format DataFrame
# 方便之後輸出 CSV、做圖、或比較不同模型的誤差
def build_predictions_long_df(
    test_df: pd.DataFrame,
    target_column: str,
    prediction_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    # 每個模型各自產生一份結果表，最後再合併
    frames = []
    # prediction_map 內會放正式模型與 baseline 的預測結果
    for model_name, predictions in prediction_map.items():
        frame = test_df[["user_id", "date", target_column]].copy()
        frame = frame.rename(columns={target_column: "y_true"})
        frame["y_pred"] = predictions
        frame["model_name"] = model_name
        frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
        frame["error"] = frame["y_pred"] - frame["y_true"]
        frames.append(frame[["user_id", "date", "y_true", "y_pred", "model_name", "abs_error", "error"]])
    return pd.concat(frames, ignore_index=True)


# 以多組超參數訓練 HGBR，並依 validation set 表現挑選最佳模型
# 這裡做的是手動 small grid search，不是 sklearn 的 GridSearchCV
def train_hgbr_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    feature_columns: list[str],
    target_column: str,
    output_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    # 手動設定幾組要測試的 HGBR 超參數組合
    candidate_configs = [
        {"learning_rate": 0.03, "max_depth": 3, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.1},
        {"learning_rate": 0.05, "max_depth": 4, "max_leaf_nodes": 31, "min_samples_leaf": 8, "l2_regularization": 0.0},
        {"learning_rate": 0.08, "max_depth": 3, "max_leaf_nodes": 31, "min_samples_leaf": 5, "l2_regularization": 0.1},
        {"learning_rate": 0.1, "max_depth": None, "max_leaf_nodes": 15, "min_samples_leaf": 5, "l2_regularization": 0.0},
    ]

    # history_records: 紀錄每一次 trial 的參數與 validation 表現
    # best_bundle:     保存目前最佳模型與其相關資訊
    history_records = []
    best_bundle = None
    best_val_metrics = None
    best_model_path = output_dir / "best_hgbr_model_v2.pkl"

    # 逐一測試每組候選參數
    for trial_index, config in enumerate(candidate_configs, start=1):
        # 建立一個 HGBR 模型
        # early_stopping=True 代表若 validation 表現連續多輪沒有改善，就提早停止
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            max_leaf_nodes=config["max_leaf_nodes"],
            min_samples_leaf=config["min_samples_leaf"],
            l2_regularization=config["l2_regularization"],
            max_iter=500,
            early_stopping=True,
            validation_fraction=None,
            n_iter_no_change=25,
            random_state=args.random_state,
        )
        # 用 train 訓練，並用 validation set 監控 early stopping
        model.fit(x_train, y_train, X_val=x_val, y_val=y_val)
        # 先看這組模型在 validation set 上的表現
        val_predictions = model.predict(x_val)
        val_metrics = compute_regression_metrics(y_val, val_predictions)

        # 把本次 trial 的結果記錄下來，之後可輸出成 history CSV
        history_records.append(
            {
                "model_name": "hgbr_v2",
                "trial": trial_index,
                "feature_set": args.feature_set,
                "feature_count": len(feature_columns),
                "iterations": int(model.n_iter_),
                "val_mae": float(val_metrics["mae"]),
                "val_rmse": float(val_metrics["rmse"]),
                **config,
            }
        )
        print(
            f"[hgbr_v2] trial={trial_index:02d} feature_set={args.feature_set} "
            f"val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f} "
            f"iters={model.n_iter_} config={config}"
        )

        # 比較這次 trial 是否優於目前最佳模型
        # 先比 MAE；若 MAE 幾乎相同，再比 RMSE
        is_better = best_val_metrics is None or (
            val_metrics["mae"] < best_val_metrics["mae"]
            or (
                np.isclose(val_metrics["mae"], best_val_metrics["mae"])
                and val_metrics["rmse"] < best_val_metrics["rmse"]
            )
        )
        # 若更好，就更新最佳模型資訊並立即存檔
        if is_better:
            best_val_metrics = val_metrics
            best_bundle = {
                "model_name": "hgbr_v2",
                "feature_set": args.feature_set,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "best_val_metric": float(val_metrics["mae"]),
                "best_val_rmse": float(val_metrics["rmse"]),
                "best_config": config,
                "iterations": int(model.n_iter_),
                "model": model,
            }
            save_pickle(best_bundle, best_model_path)

    return best_bundle, pd.DataFrame(history_records)


# 主流程：串接整個訓練、評估與輸出流程
def main() -> None:
    # 讀取命令列參數，並確保輸出資料夾存在
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 讀取 train / val / test 三份資料
    train_df = pd.read_parquet(args.train_path)
    val_df = pd.read_parquet(args.val_path)
    test_df = pd.read_parquet(args.test_path)

    # 自動決定 target 欄位，並選出這次要訓練的特徵欄位
    target_column = resolve_target_column([train_df, val_df, test_df], args.target_column)
    feature_columns = resolve_feature_columns(train_df, target_column, args)

    # 印出目前使用的特徵，方便訓練時檢查
    print("== Model V2 Feature Selection ==")
    print(f"feature_set={args.feature_set}")
    print(f"feature_count={len(feature_columns)}")
    for index, feature_name in enumerate(feature_columns, start=1):
        print(f"{index:02d}. {feature_name}")

    # 把 DataFrame 轉成模型要吃的 numpy 格式
    x_train, y_train = prepare_xy(train_df, feature_columns, target_column)
    x_val, y_val = prepare_xy(val_df, feature_columns, target_column)
    x_test, y_test = prepare_xy(test_df, feature_columns, target_column)

    # 進行多組 HGBR 參數訓練，並取得 validation 表現最好的模型
    best_bundle, history_df = train_hgbr_search(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        args=args,
        feature_columns=feature_columns,
        target_column=target_column,
        output_dir=args.output_dir,
    )

    # 用最佳模型在 test set 上做最終評估
    best_model = best_bundle["model"]
    test_predictions = best_model.predict(x_test)
    model_metrics = compute_regression_metrics(y_test, test_predictions)

    # 建立兩個 baseline 預測，檢查正式模型是否真的優於簡單規則
    naive_predictions = test_df[NAIVE_BASELINE_FEATURE].astype(np.float32).to_numpy()
    moving_avg_predictions = test_df[MOVING_AVG_BASELINE_FEATURE].astype(np.float32).to_numpy() * 7.0
    naive_metrics = compute_regression_metrics(y_test, naive_predictions)
    moving_avg_metrics = compute_regression_metrics(y_test, moving_avg_predictions)

    # 印出 test set 評估結果
    print("== Model V2 Test Metrics ==")
    print(f"hgbr_v2 test_mae={model_metrics['mae']:.6f} test_rmse={model_metrics['rmse']:.6f}")
    print(f"naive_7d_sum mae={naive_metrics['mae']:.6f} rmse={naive_metrics['rmse']:.6f}")
    print(f"moving_avg_30d_x7 mae={moving_avg_metrics['mae']:.6f} rmse={moving_avg_metrics['rmse']:.6f}")

    # 這裡用一個簡單條件判斷：正式模型的 MAE 與 RMSE 是否都優於 moving average baseline
    beat_moving_avg = model_metrics["mae"] < moving_avg_metrics["mae"] and model_metrics["rmse"] < moving_avg_metrics["rmse"]
    print(f"beat_moving_avg_30d_x7={'YES' if beat_moving_avg else 'NO'}")

    # 輸出 trial 歷史紀錄
    history_path = args.output_dir / "training_history_v2.csv"
    history_df.to_csv(history_path, index=False)

    # 輸出 test set 各模型逐筆預測結果
    predictions_df = build_predictions_long_df(
        test_df=test_df,
        target_column=target_column,
        prediction_map={
            "hgbr_v2": test_predictions,
            "naive_7d_sum": naive_predictions,
            "moving_avg_30d_x7": moving_avg_predictions,
        },
    )
    predictions_path = args.output_dir / "predictions_test_v2.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # 輸出本次實驗使用的特徵清單，方便之後回顧
    selected_features_path = args.output_dir / "selected_features_v2.json"
    selected_features_payload = {
        "feature_set": args.feature_set,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": target_column,
    }
    with selected_features_path.open("w", encoding="utf-8") as file:
        json.dump(selected_features_payload, file, ensure_ascii=False, indent=2)

    # 彙整本次實驗的重要指標，輸出成 JSON
    metrics = {
        "model_name": "hgbr_v2",
        "feature_set": args.feature_set,
        "feature_columns": feature_columns,
        "best_val_metric": float(best_bundle["best_val_metric"]),
        "test_mae": float(model_metrics["mae"]),
        "test_rmse": float(model_metrics["rmse"]),
        "baseline_mae": {
            "naive_7d_sum": float(naive_metrics["mae"]),
            "moving_avg_30d_x7": float(moving_avg_metrics["mae"]),
        },
        "baseline_rmse": {
            "naive_7d_sum": float(naive_metrics["rmse"]),
            "moving_avg_30d_x7": float(moving_avg_metrics["rmse"]),
        },
        "beat_moving_avg_30d_x7": bool(beat_moving_avg),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_columns)),
    }
    metrics_path = args.output_dir / "test_metrics_v2.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    # 另外輸出一份純文字報告，方便快速查看訓練結果
    report_lines = [
        "Training Run Summary V2",
        "model_name: hgbr_v2",
        f"feature_set: {args.feature_set}",
        f"target_column: {target_column}",
        f"best_val_metric: {best_bundle['best_val_metric']:.6f}",
        f"test_mae: {model_metrics['mae']:.6f}",
        f"test_rmse: {model_metrics['rmse']:.6f}",
        "",
        "Dataset Sizes",
        f"train_rows: {len(train_df)}",
        f"val_rows: {len(val_df)}",
        f"test_rows: {len(test_df)}",
        f"feature_count: {len(feature_columns)}",
        "",
        "Selected Features",
    ]
    report_lines.extend([f"- {feature}" for feature in feature_columns])
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
            f"best_model: {args.output_dir / 'best_hgbr_model_v2.pkl'}",
            f"history_csv: {history_path}",
            f"predictions_csv: {predictions_path}",
            f"metrics_json: {metrics_path}",
            f"selected_features_json: {selected_features_path}",
        ]
    )
    report_path = args.output_dir / "training_report_v2.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # 顯示所有輸出檔案的位置
    print(f"saved best model to {args.output_dir / 'best_hgbr_model_v2.pkl'}")
    print(f"saved history to {history_path}")
    print(f"saved predictions to {predictions_path}")
    print(f"saved metrics to {metrics_path}")
    print(f"saved selected features to {selected_features_path}")
    print(f"saved report to {report_path}")


# 直接執行這支檔案時，從 main() 開始跑整個流程
if __name__ == "__main__":
    main()
