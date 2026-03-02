from pathlib import Path
import pandas as pd
import numpy as np

IN_PATH = Path("artifacts/daily_ledger_all.parquet")
OUT_PATH = Path("artifacts/features_all.parquet")

# 可改：預警提前幾天（horizon）
HORIZON_DAYS = 7

# rolling window（只用過去資料）
WINDOWS = [7, 30]


def add_rolling_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)

    # 避免洩漏：rolling 都用過去，shift(1) 代表「不包含今天」更保守
    for w in WINDOWS:
        g[f"expense_{w}d_sum"] = g["daily_expense"].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)
        g[f"expense_{w}d_mean"] = g["daily_expense"].rolling(w, min_periods=1).mean().shift(1).fillna(0.0)
        g[f"net_{w}d_sum"] = g["daily_net"].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)
        g[f"txn_{w}d_sum"] = g["txn_count"].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)

    # 目標候選：未來 H 天支出總和（包含明天到第 H 天）
    # 例：H=7，對應 t+1 ~ t+7
    future = (
        g["daily_expense"]
        .shift(-1)
        .rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS)
        .sum()
        .shift(-(HORIZON_DAYS - 1))
    )
    g[f"future_expense_{HORIZON_DAYS}d_sum"] = future

    return g


def main():
    df = pd.read_parquet(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # 季節/假期特徵（近似）：以月份標記寒暑假
    # 你可依研究情境調整月份範圍
    df["month"] = df["date"].dt.month.astype(int)
    df["is_summer_vacation"] = df["month"].isin([7, 8]).astype(int)
    df["is_winter_vacation"] = df["month"].isin([1, 2]).astype(int)

    # 週末特徵
    df["is_weekend"] = df["is_weekend"].astype(int)

    feat = df.groupby("user_id", group_keys=False).apply(add_rolling_features)

    # 未來 H 天無法算的尾端會是 NaN（每個 user 最後 H 天）
    feat = feat.dropna(subset=[f"future_expense_{HORIZON_DAYS}d_sum"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 輸出 parquet（給模型用）
    feat.to_parquet(OUT_PATH, index=False)

    # 同步輸出 csv（方便人工檢查）
    csv_path = OUT_PATH.with_suffix(".csv")
    feat.to_csv(csv_path, index=False)

    print(
        f"Saved: {OUT_PATH} & {csv_path} "
        f"rows={len(feat)} users={feat['user_id'].nunique()} horizon={HORIZON_DAYS}"
    )


if __name__ == "__main__":
    main()