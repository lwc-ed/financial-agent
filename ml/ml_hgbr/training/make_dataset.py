#conver raw data to dataset(only for 1 user)
"""
	•	date
當天日期（天粒度）。後續所有 rolling window（如 7 天平均、30 天總支出）都以它排序。
	•	daily_income
當日「收入」總額（把交易中 transaction_type = Income 的 amount 加總）。
用途：衡量收入節奏、計算淨現金流、做預警時可當保護因子（有收入但支出高 vs 沒收入支出高）。
	•	daily_expense
當日「支出」總額（transaction_type = Expense 的 amount 加總）。
用途：預警核心訊號，後續會做「近 7 天/30 天支出」與「支出異常」判定。
	•	txn_count
當日交易筆數（收入+支出全部筆數）。
用途：行為強度/頻率特徵。某些人衝動消費會表現為「筆數暴增」不一定是金額暴增。
	•	income_txn_count
當日收入交易筆數（Income 有幾筆）。
用途：分辨「一次性入帳」或「多筆入帳」，也可作為收入穩定性線索。
	•	expense_txn_count
當日支出交易筆數（Expense 有幾筆）。
用途：對衝動型支出很有用（例如小額多筆 vs 大額單筆）。
	•	daily_net
當日淨額：
daily\_net = daily\_income - daily\_expense
用途：比單看支出更貼近「錢包壓力」。負值大代表當天支出超過收入很多。
	•	has_income
當日是否有收入（有則 1，無則 0）。
用途：模型可學到「發薪日/入帳日」型態；也能在規則型預警中當 gating（例如有收入日容忍度高）。
	•	has_expense
當日是否有支出（有則 1，無則 0）。
用途：辨識「完全沒花錢」的日子；對缺失日補零後，也能區分「真的沒花」vs「資料缺失」的風險（後續可以再加欄位處理）。
	•	dow（day of week）
星期幾，0=週一 … 6=週日。
用途：消費通常有週期（週末較高），模型容易用它抓規律。
	•	is_weekend
是否週末（週六/週日 = 1，其他 = 0）。
用途：比 dow 更粗粒度的週期特徵，對小數據更穩。
	•	day
月中的第幾天（1–31）。
用途：很多人有「月初/月中/月末」支出節奏，搭配薪資週期有用。
	•	month
月份（1–12）。
用途：季節性（過年、暑假、開學、年末）可能影響消費。
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR.parent / "data" / "raw_transactions_user2.xlsx"
OUT_PARQUET = BASE_DIR / "artifacts" / "daily_ledger_user2.parquet"
OUT_CSV = BASE_DIR / "artifacts" / "daily_ledger_user2.csv"


def _looks_like_binary_spreadsheet(path: Path) -> bool:
    with path.open("rb") as file:
        header = file.read(8)
    return header.startswith(b"PK\x03\x04")


def _normalize_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("原始資料是空的")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = {"time_stamp", "transaction_type", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要欄位: {sorted(missing)}")

    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    if df["time_stamp"].isna().any():
        bad = df[df["time_stamp"].isna()].head(5)
        raise ValueError(f"time_stamp 有無法解析的值（前5筆如下）:\n{bad}")

    df["transaction_type"] = df["transaction_type"].astype(str).str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if df["amount"].isna().any():
        bad = df[df["amount"].isna()].head(5)
        raise ValueError(f"amount 有無法轉成數字的值（前5筆如下）:\n{bad}")

    if "net_cash_flow" in df.columns:
        df["net_cash_flow"] = pd.to_numeric(df["net_cash_flow"], errors="coerce")

    return df


def load_raw(path: str | Path) -> pd.DataFrame:
    """
    讀取 raw data（CSV/TSV），自動嘗試不同分隔符與常見編碼。
    欄位預期：time_stamp, transaction_type, amount, net_cash_flow

    你目前遇到的 `UnicodeDecodeError` 通常是檔案不是 UTF-8（常見：Big5/CP950）。
    這裡會依序嘗試多種 encoding，成功就停。
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到原始資料檔：{path}")

    if _looks_like_binary_spreadsheet(path):
        try:
            return _normalize_raw_df(pd.read_excel(path, engine="openpyxl"))
        except ImportError as exc:
            raise RuntimeError(
                "讀取 Excel 檔需要 openpyxl。請先執行 `pip install openpyxl` 或重裝 `pip install -r ml/requirements.txt`。"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{path} 看起來是 ZIP 型試算表檔，但無法當作 Excel 讀取。\n"
                "如果這是 Numbers 檔，請先匯出成 `.xlsx` 或純文字 `.csv`。"
            ) from exc

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp950",
        "big5",
        "latin1",
    ]

    last_err: Exception | None = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            if df.shape[1] == 1:
                continue
            return _normalize_raw_df(df)

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"無法讀取檔案：{path}。\n"
        f"已嘗試 encodings={encodings} 與自動分隔符偵測。\n"
        f"最後一次錯誤：{last_err}"
    )

def to_daily_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 raw 交易資料聚合成「每日」資料：
    - daily_income：當日收入總和
    - daily_expense：當日支出總和
    - daily_net：income - expense
    - txn_count：交易筆數
    - income_txn_count / expense_txn_count
    - has_income / has_expense
    並補齊缺失日期。
    """
    df = df.copy()
    df["date"] = df["time_stamp"].dt.floor("D")

    # 把 Income/Expense 轉成兩個欄位
    df["income_amt"] = np.where(df["transaction_type"].str.lower() == "income", df["amount"], 0.0)
    df["expense_amt"] = np.where(df["transaction_type"].str.lower() == "expense", df["amount"], 0.0)

    # 基本聚合
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

    # 補齊缺失日期：沒有任何交易的天補 0
    daily = daily.sort_values("date").reset_index(drop=True)
    all_days = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(all_days).fillna(0.0).reset_index()
    daily = daily.rename(columns={"index": "date"})

    # 重新把 count 欄位轉成 int（補齊後會變 float）
    int_cols = ["txn_count", "income_txn_count", "expense_txn_count", "has_income", "has_expense"]
    for c in int_cols:
        daily[c] = daily[c].astype(int)

    # 補充一些常用時間特徵（可選，但通常很有用）
    daily["dow"] = daily["date"].dt.dayofweek.astype(int)          # 0=Mon, 6=Sun
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["day"] = daily["date"].dt.day.astype(int)
    daily["month"] = daily["date"].dt.month.astype(int)

    return daily

def main():
    df = load_raw(RAW_PATH)
    daily = to_daily_ledger(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    daily.to_parquet(OUT_PARQUET, index=False)
    daily.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUT_PARQUET}")
    print(f"Saved: {OUT_CSV}")
    print(f"Date range: {daily['date'].min().date()} ~ {daily['date'].max().date()}")
    print(f"Rows(days): {len(daily)}")
    print("Head:")
    print(daily.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
