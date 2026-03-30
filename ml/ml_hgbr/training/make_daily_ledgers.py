#conver raw data to dataset(multi-users version)
import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 之後可改成 argparse，但先用常數快速跑通
USER_IDS = [f"user{i}" for i in range(1, 21)]


def load_raw(path: Path) -> pd.DataFrame:
    # Excel 檔（.xlsx/.xls）直接用 read_excel，不走 encoding/sep
    if path.suffix.lower() in [".xlsx", ".xls"]:
        try:
            # .xlsx 通常用 openpyxl；.xls 可能需要 xlrd（若你沒有 .xls 可忽略）
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl" if path.suffix.lower() == ".xlsx" else None)
        except Exception as e:
            raise RuntimeError(
                f"Failed to read Excel file: {path}. "
                f"If it's .xlsx, please `pip install openpyxl`. "
                f"If it's .xls, please `pip install xlrd`. "
                f"Original error: {e}"
            )

        df.columns = [str(c).strip() for c in df.columns]
        required = {"time_stamp", "transaction_type", "amount"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {missing} in {path}. Columns={df.columns.tolist()}")

        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df["transaction_type"] = df["transaction_type"].astype(str).str.strip()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

        if df["time_stamp"].isna().any():
            raise ValueError("time_stamp contains NaT after parsing")
        if df["amount"].isna().any():
            raise ValueError("amount contains NaN after parsing")

        return df

    encodings = ["utf-8", "utf-8-sig", "cp950", "big5", "utf-16", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            # 自動猜分隔符，較不容易被逗號/Tab/分號搞死
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                encoding=enc,
                skipinitialspace=True,
                thousands=",",
            )
            if df.shape[1] == 1:
                raise ValueError("delimiter sniff failed (only 1 column)")

            df.columns = [c.strip() for c in df.columns]
            required = {"time_stamp", "transaction_type", "amount"}
            if not required.issubset(df.columns):
                raise ValueError(f"missing columns: {required - set(df.columns)}")

            df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
            df["transaction_type"] = df["transaction_type"].astype(str).str.strip()
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

            # 基本檢查
            if df["time_stamp"].isna().any():
                raise ValueError("time_stamp contains NaT after parsing")
            if df["amount"].isna().any():
                raise ValueError("amount contains NaN after parsing")

            return df

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Cannot read {path}. Last error: {last_err}")


def to_daily_ledger(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["time_stamp"].dt.floor("D")

    df["income_amt"] = np.where(df["transaction_type"].str.lower() == "income", df["amount"], 0.0)
    df["expense_amt"] = np.where(df["transaction_type"].str.lower() == "expense", df["amount"], 0.0)

    daily = df.groupby("date", as_index=False).agg(
        daily_income=("income_amt", "sum"),
        daily_expense=("expense_amt", "sum"),
        txn_count=("amount", "size"),
        income_txn_count=("income_amt", lambda x: int((x > 0).sum())),
        expense_txn_count=("expense_amt", lambda x: int((x > 0).sum())),
    )

    daily["daily_net"] = daily["daily_income"] - daily["daily_expense"]
    daily["has_income"] = (daily["daily_income"] > 0).astype(int)
    daily["has_expense"] = (daily["daily_expense"] > 0).astype(int)

    daily = daily.sort_values("date").reset_index(drop=True)
    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(all_days).fillna(0.0).reset_index()
    daily = daily.rename(columns={"index": "date"})

    # 重新轉 int
    for c in ["txn_count", "income_txn_count", "expense_txn_count", "has_income", "has_expense"]:
        daily[c] = daily[c].astype(int)

    # 時間特徵
    daily["dow"] = daily["date"].dt.dayofweek.astype(int)
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["day"] = daily["date"].dt.day.astype(int)
    daily["month"] = daily["date"].dt.month.astype(int)

    return daily


def main():
    all_daily = []

    for uid in USER_IDS:
        raw_path = None
        for ext in [".csv", ".CSV", ".tsv", ".TSV", ".xlsx", ".XLSX", ".xls", ".XLS"]:
            candidate = DATA_DIR / f"raw_transactions_{uid}{ext}"
            if candidate.exists():
                raw_path = candidate
                break

        if raw_path is None:
            print(f"[SKIP] {uid}: file not found -> {DATA_DIR}/raw_transactions_{uid}(.csv/.tsv/.xlsx)")
            continue

        df = load_raw(raw_path)
        daily = to_daily_ledger(df)
        daily.insert(0, "user_id", uid)  # 放第一欄

        out_parquet = OUT_DIR / f"daily_ledger_{uid}.parquet"
        out_csv = OUT_DIR / f"daily_ledger_{uid}.csv"

        daily.to_parquet(out_parquet, index=False)
        daily.to_csv(out_csv, index=False)

        print(f"[OK] {uid}: saved {out_parquet} & {out_csv} (days={len(daily)})")

        all_daily.append(daily)

    if not all_daily:
        raise RuntimeError("No user data processed. Check filenames under data/")

    merged = pd.concat(all_daily, ignore_index=True).sort_values(["user_id", "date"]).reset_index(drop=True)
    merged_parquet = OUT_DIR / "daily_ledger_all.parquet"
    merged_csv = OUT_DIR / "daily_ledger_all.csv"

    merged.to_parquet(merged_parquet, index=False)
    merged.to_csv(merged_csv, index=False)

    print(
        f"[DONE] merged saved: {merged_parquet} & {merged_csv} "
        f"rows={len(merged)} users={merged['user_id'].nunique()}"
    )


if __name__ == "__main__":
    main()
