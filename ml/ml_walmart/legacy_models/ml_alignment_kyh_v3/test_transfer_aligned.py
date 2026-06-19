import json
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


def evaluate_model(model_path, scaler, df, features, name):
    X = df[features].copy()
    y_raw = df["target"].copy()

    X_sc = scaler.transform(X)

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    preds = model.predict(X_sc)
    preds = preds.clip(min=0)

    pred_df = df[["user_id", "date", "target"]].copy()
    pred_df["y_true"] = y_raw.values
    pred_df["y_pred"] = preds
    pred_df.to_csv(
        RESULT_DIR / f"predictions_{name}_v3_huber_raw.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "model_name": name,
        "mae": float(mean_absolute_error(y_raw, preds)),
        "rmse": float(mean_squared_error(y_raw, preds) ** 0.5),
        "rows": int(len(df)),
        "feature_count": int(len(features)),
        "objective": "reg:pseudohubererror",
        "target_space": "raw",
    }


def main():
    own_data_path = DATA_DIR / "own_processed_aligned.csv"
    schema_path = DATA_DIR / "common_features.json"
    scaler_path = MODEL_DIR / "feature_scaler_v3_huber_raw.pkl"
    base_model_path = MODEL_DIR / "xgb_kaggle_aligned_v3_huber_raw.json"
    finetuned_model_path = MODEL_DIR / "xgb_finetuned_own_aligned_v3_huber_raw.json"

    df = pd.read_csv(own_data_path)
    features = load_feature_schema(schema_path)
    scaler = load_pickle(scaler_path)

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    print("========== TEST INFO ==========")
    print(f"Test size: {test_df.shape}")
    print("Feature columns:")
    print(features)
    print("Test target summary:")
    print(test_df["target"].describe())

    results = []

    if base_model_path.exists():
        results.append(
            evaluate_model(base_model_path, scaler, test_df, features, "base_aligned")
        )

    if finetuned_model_path.exists():
        results.append(
            evaluate_model(finetuned_model_path, scaler, test_df, features, "finetuned_aligned")
        )

    output_json = RESULT_DIR / "transfer_test_metrics_v3_huber_raw.json"
    output_txt = RESULT_DIR / "aligned_results_v3_huber_raw.txt"

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


if __name__ == "__main__":
    main()