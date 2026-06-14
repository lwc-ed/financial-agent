"""
bigru_service.py
================
串接 ml/ml_ibm/bigru_TL_alignment 訓練好的模型，
從使用者 DB 紀錄預測未來 7 天消費並計算財務風險等級。
"""

import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_BIGRU_DIR = _HERE.parent.parent / "ml" / "ml_ibm" / "bigru_TL_alignment"
_ARTIFACTS_DIR = _BIGRU_DIR / "artifacts_bigru_tl"

if str(_BIGRU_DIR) not in sys.path:
    sys.path.insert(0, str(_BIGRU_DIR))

from alignment_utils import ALIGNED_FEATURE_COLS, compute_aligned_features  # noqa: E402
from model_bigru import BiGRUWithAttention  # noqa: E402

# ── Model hyperparams (must match training) ────────────────────────────────────
_INPUT_SIZE = len(ALIGNED_FEATURE_COLS)  # 10
_HIDDEN_SIZE = 48
_NUM_LAYERS = 2
_DROPOUT = 0.4
_INPUT_DAYS = 30

# ── Lazy-loaded module-level cache ─────────────────────────────────────────────
_models: list | None = None
_scaler = None
_device: torch.device | None = None


def _get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    return _device


def _load_assets() -> None:
    global _models, _scaler
    if _models is not None:
        return

    device = _get_device()
    pth_files = sorted(_ARTIFACTS_DIR.glob("finetune_bigru_seed*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"找不到模型權重檔：{_ARTIFACTS_DIR}")

    loaded = []
    for pth in pth_files:
        model = BiGRUWithAttention(_INPUT_SIZE, _HIDDEN_SIZE, _NUM_LAYERS, 1, _DROPOUT).to(device)
        ckpt = torch.load(pth, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        loaded.append(model)
    _models = loaded

    with open(_ARTIFACTS_DIR / "personal_target_scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)

    print(f"[bigru_service] 已載入 {len(_models)} 個 seed 模型，設備={_get_device()}")


def _risk_level(ratio: float) -> int:
    if ratio <= 0.85:
        return 1
    if ratio <= 1.0:
        return 2
    if ratio <= 1.2:
        return 3
    return 4


def get_user_daily_expense(line_user_id: str, db, days: int = 90) -> pd.DataFrame:
    from sqlalchemy import func as sqlfunc
    from backend.models.record import Record

    cutoff = datetime.now() - timedelta(days=days)
    rows = (
        db.query(
            sqlfunc.date(Record.timestamp).label("date"),
            sqlfunc.sum(Record.amount).label("daily_expense"),
        )
        .filter(
            Record.line_user_id == line_user_id,
            Record.type == "expense",
            Record.timestamp >= cutoff,
        )
        .group_by(sqlfunc.date(Record.timestamp))
        .order_by(sqlfunc.date(Record.timestamp))
        .all()
    )

    if not rows:
        return pd.DataFrame(columns=["date", "daily_expense"])

    df = pd.DataFrame(rows, columns=["date", "daily_expense"])
    df["date"] = pd.to_datetime(df["date"])
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range, fill_value=0.0).reset_index()
    df.columns = ["date", "daily_expense"]
    return df


def get_monthly_income_avg(line_user_id: str, db, months: int = 6) -> float:
    from sqlalchemy import func as sqlfunc
    from backend.models.record import Record

    cutoff = datetime.now() - timedelta(days=months * 30)
    total = (
        db.query(sqlfunc.sum(Record.amount))
        .filter(
            Record.line_user_id == line_user_id,
            Record.type == "income",
            Record.timestamp >= cutoff,
        )
        .scalar()
    ) or 0.0
    return float(total) / months


def get_net_cash_flow_30d(line_user_id: str, db) -> float:
    from sqlalchemy import func as sqlfunc
    from backend.models.record import Record

    cutoff = datetime.now() - timedelta(days=30)
    income = (
        db.query(sqlfunc.sum(Record.amount))
        .filter(
            Record.line_user_id == line_user_id,
            Record.type == "income",
            Record.timestamp >= cutoff,
        )
        .scalar()
    ) or 0.0
    expense = (
        db.query(sqlfunc.sum(Record.amount))
        .filter(
            Record.line_user_id == line_user_id,
            Record.type == "expense",
            Record.timestamp >= cutoff,
        )
        .scalar()
    ) or 0.0
    return float(income) - float(expense)


def predict_risk_for_user(line_user_id: str, db) -> dict | None:
    """
    從 DB 取消費紀錄 → 計算 aligned features → BiGRU ensemble 預測 → 回傳風險評估結果。
    若消費紀錄不足 30 天則回傳 None（silent skip）。

    Risk Score = 0.6 × Spending Pressure + 0.4 × Cash Flow Risk
      Spending Pressure = 預測未來7天消費 / 未來7天可動用收入
      Cash Flow Risk    = 1 - clip(Net Cash Flow 30d / 月均收入, -2, 2)
    """
    _load_assets()

    daily_df = get_user_daily_expense(line_user_id, db)
    if len(daily_df) < _INPUT_DAYS:
        print(f"[bigru_service] {line_user_id} 資料不足（{len(daily_df)} 天），跳過預測")
        return None

    window_df = daily_df.tail(_INPUT_DAYS).reset_index(drop=True)
    feat_df = compute_aligned_features(window_df["daily_expense"], window_df["date"])
    X = feat_df[ALIGNED_FEATURE_COLS].values  # (30, 10)
    X_tensor = torch.tensor(X[np.newaxis, :, :], dtype=torch.float32).to(_get_device())

    seed_preds = []
    with torch.no_grad():
        for model in _models:
            pred = model(X_tensor).cpu().numpy().flatten()[0]
            seed_preds.append(pred)
    scaled_pred = float(np.mean(seed_preds))
    predicted_7d = float(_scaler.inverse_transform([[scaled_pred]])[0][0])
    predicted_7d = max(0.0, predicted_7d)

    monthly_income_avg = get_monthly_income_avg(line_user_id, db)

    if monthly_income_avg <= 0:
        spending_pressure = 99.0
        net_cash_flow = None
        cf_risk = 3.0  # 無收入紀錄視為高風險
    else:
        future_available_7d = (monthly_income_avg / 30.0) * 7.0
        spending_pressure = min(predicted_7d / future_available_7d, 99.0)

        net_cash_flow = get_net_cash_flow_30d(line_user_id, db)
        saving_rate = np.clip(net_cash_flow / monthly_income_avg, -2.0, 2.0)
        cf_risk = 1.0 - saving_rate

    risk_score = 0.6 * spending_pressure + 0.4 * cf_risk
    level = _risk_level(risk_score)
    alarm = "high_risk" if risk_score > 1.0 else "low_risk"

    return {
        "predicted_expense_7d": predicted_7d,
        "monthly_income_avg": monthly_income_avg,
        "spending_pressure": spending_pressure,
        "net_cash_flow_30d": net_cash_flow,
        "cf_risk": cf_risk,
        "risk_score": risk_score,
        "risk_level": level,
        "alarm": alarm,
        "data_days": len(daily_df),
    }
