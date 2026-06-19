from pathlib import Path
import pandas as pd


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
IN_PATH = ARTIFACTS_DIR / "daily_ledger_all.parquet"
OUT_PATH = ARTIFACTS_DIR / "features_all.parquet"

HORIZON_DAYS = 7
WINDOWS = [7, 30]


def read_daily_ledger() -> pd.DataFrame:
    if IN_PATH.exists():
        try:
            return pd.read_parquet(IN_PATH)
        except ImportError:
            print("[WARN] parquet engine not installed, fallback to daily_ledger_all.csv")

    csv_path = IN_PATH.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"找不到 {IN_PATH} 或 {csv_path}。請先執行 build_daily_ledgers.py"
    )


def write_features(df: pd.DataFrame) -> None:
    csv_path = OUT_PATH.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(OUT_PATH, index=False)
    except ImportError:
        print(f"[WARN] parquet engine not installed, skipped {OUT_PATH.name}; csv 已儲存")


def add_rolling_features(g: pd.DataFrame, user_id: str) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)

    for w in WINDOWS:
        g[f"expense_{w}d_sum"] = g["daily_expense"].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)
        g[f"expense_{w}d_mean"] = g["daily_expense"].rolling(w, min_periods=1).mean().shift(1).fillna(0.0)
        g[f"net_{w}d_sum"] = g["daily_net"].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)
        g[f"txn_{w}d_sum"] = g["txn_count"].rolling(w, min_periods=1).sum().shift(1).fillna(0.0)

    g["expense_7d_30d_ratio"] = (g["expense_7d_mean"] + 1.0) / (g["expense_30d_mean"] + 1.0)
    g["expense_trend"] = g["expense_7d_mean"] - g["expense_30d_mean"]

    future = (
        g["daily_expense"]
        .shift(-1)
        .rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS)
        .sum()
        .shift(-(HORIZON_DAYS - 1))
    )
    g[f"future_expense_{HORIZON_DAYS}d_sum"] = future
    g["user_id"] = user_id
    return g


def main():
    df = read_daily_ledger()
    df["date"] = pd.to_datetime(df["date"])
    exclude_users = ["user4", "user5", "user6"]
    df = df[~df["user_id"].isin(exclude_users)].reset_index(drop=True)
    print(f"  使用者數: {df['user_id'].nunique()}  (排除 {exclude_users})")

    df["month"] = df["date"].dt.month.astype(int)
    df["is_summer_vacation"] = df["month"].isin([7, 8]).astype(int)
    df["is_winter_vacation"] = df["month"].isin([1, 2]).astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)

    month_end = df["date"] + pd.offsets.MonthEnd(0)
    df["days_to_end_of_month"] = (month_end - df["date"]).dt.days.astype(int)

    result = []
    for user_id, g in df.groupby("user_id", sort=False):
        result.append(add_rolling_features(g.copy(), user_id))

    feat = pd.concat(result, ignore_index=True)
    feat = feat.dropna(subset=[f"future_expense_{HORIZON_DAYS}d_sum"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_features(feat)
    csv_path = OUT_PATH.with_suffix(".csv")

    print(
        f"Saved: {OUT_PATH} & {csv_path} "
        f"rows={len(feat)} users={feat['user_id'].nunique()} horizon={HORIZON_DAYS}"
    )


if __name__ == "__main__":
    main()
