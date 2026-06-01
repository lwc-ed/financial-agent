"""財務狀況指標計算（不接 LLM）。

記帳後在背景 thread 呼叫，把餘額 / 本月收支 / 7 日趨勢 / 最大類別等
指標連同 BiGRU 風險結果算好，upsert 進 financial_status 表。
白話文字的生成在讀取 API（routes/financial_status.py）才做。
"""
import json
import hashlib
from datetime import datetime, timedelta

import pytz
from sqlalchemy import func

from backend.models.record import Record
from backend.models.financial_status import FinancialStatus

taipei = pytz.timezone("Asia/Taipei")

# 資料不足門檻：記帳天數少於此值就不顯示風險分級，改走引導文案
MIN_DATA_DAYS = 30

# Level → (emoji, 親和標題, 前端色票 key)
LEVEL_DISPLAY = {
    1: {"emoji": "🟢", "title": "財務狀況很穩！", "color": "emerald"},
    2: {"emoji": "🔵", "title": "整體還不錯喔", "color": "blue"},
    3: {"emoji": "🟡", "title": "最近要稍微注意一下", "color": "amber"},
    4: {"emoji": "🔴", "title": "要當心囉，支出有點吃緊", "color": "red"},
}


def _now_taipei() -> datetime:
    return datetime.now(taipei).replace(tzinfo=None)


def _sum_expense_between(db, line_user_id: str, start: datetime, end: datetime) -> int:
    total = (
        db.query(func.coalesce(func.sum(Record.amount), 0))
        .filter(
            Record.line_user_id == line_user_id,
            Record.type == "expense",
            Record.timestamp >= start,
            Record.timestamp < end,
        )
        .scalar()
    )
    return int(total or 0)


def _compute_data_metrics(db, line_user_id: str) -> dict:
    """純記帳資料算出的指標（餘額、最近30天收支、7 日趨勢、最大類別）。

    收支採「滾動 30 天」而非 calendar 月，避免月初卡片空白。
    """
    now = _now_taipei()
    window_start = now - timedelta(days=30)

    # 最近 30 天收支
    rows = (
        db.query(Record.type, func.coalesce(func.sum(Record.amount), 0))
        .filter(Record.line_user_id == line_user_id, Record.timestamp >= window_start)
        .group_by(Record.type)
        .all()
    )
    win_totals = {t: int(v or 0) for t, v in rows}
    income_30d = win_totals.get("income", 0)
    expense_30d = win_totals.get("expense", 0)

    # 餘額（全期間 收入 - 支出，與 /api/expense_history/summary 一致）
    all_rows = (
        db.query(Record.type, func.coalesce(func.sum(Record.amount), 0))
        .filter(Record.line_user_id == line_user_id)
        .group_by(Record.type)
        .all()
    )
    all_totals = {t: int(v or 0) for t, v in all_rows}
    balance = all_totals.get("income", 0) - all_totals.get("expense", 0)

    # 7 日支出趨勢：最近 7 天 vs 前 7 天
    recent_7 = _sum_expense_between(db, line_user_id, now - timedelta(days=7), now)
    prev_7 = _sum_expense_between(db, line_user_id, now - timedelta(days=14), now - timedelta(days=7))
    # 上週基數太小時，百分比會爆掉且失去意義 → 改用文字描述方向
    TREND_FLOOR = 500    # 上週支出低於此金額視為基數不足
    TREND_CAP = 200      # 增幅超過此值，百分比已失去意義，改用文字
    if prev_7 >= TREND_FLOOR:
        raw_pct = round((recent_7 - prev_7) / prev_7 * 100)
        if raw_pct > TREND_CAP:
            trend_pct = None
            trend_label = "明顯增加"   # 暴增，講百分比反而嚇人
        else:
            trend_pct = raw_pct       # 下降最多 -100%，不會爆值
            trend_label = None
    elif recent_7 >= TREND_FLOOR:
        trend_pct = None
        trend_label = "明顯增加"   # 上週幾乎沒花、這週有花
    else:
        trend_pct = None
        trend_label = "持平"       # 兩週都幾乎沒花

    # 最近 30 天最大消費類別
    top_row = (
        db.query(Record.category, func.coalesce(func.sum(Record.amount), 0).label("amt"))
        .filter(
            Record.line_user_id == line_user_id,
            Record.type == "expense",
            Record.timestamp >= window_start,
        )
        .group_by(Record.category)
        .order_by(func.sum(Record.amount).desc())
        .first()
    )
    top_category = None
    if top_row and expense_30d > 0:
        top_category = {
            "name": top_row[0],
            "amount": int(top_row[1] or 0),
            "pct": round(int(top_row[1] or 0) / expense_30d * 100),
        }

    return {
        "balance": balance,
        "income_30d": income_30d,
        "expense_30d": expense_30d,
        "recent_7d_expense": recent_7,
        "trend_pct": trend_pct,      # 百分比；基數不足時為 None
        "trend_label": trend_label,  # 基數不足時的文字描述，否則 None
        "top_category": top_category,
    }


def _metrics_hash(metrics: dict) -> str:
    """對影響「文字內容」的欄位做穩定 hash，指標沒變就不需重生成文字。"""
    keys = ["level", "balance", "income_30d", "expense_30d",
            "trend_pct", "trend_label", "top_category", "insufficient"]
    payload = {k: metrics.get(k) for k in keys}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_metrics(db, line_user_id: str, risk_result: dict | None) -> dict:
    """組出完整指標包。risk_result 為 None 時（沒有風險預測）視為資料不足。"""
    data_metrics = _compute_data_metrics(db, line_user_id)

    data_days = int(risk_result.get("data_days", 0)) if risk_result else 0
    monthly_income_avg = float(risk_result.get("monthly_income_avg", 0)) if risk_result else 0.0
    risk_level = int(risk_result.get("risk_level", 0)) if risk_result else 0
    risk_score = float(risk_result.get("risk_score", 0)) if risk_result else 0.0
    predicted_7d = int(risk_result.get("predicted_expense_7d", 0)) if risk_result else 0

    # 資料不足 gate：排在分級之前
    insufficient = (risk_result is None) or (data_days < MIN_DATA_DAYS) or (monthly_income_avg <= 0)

    metrics = {
        **data_metrics,
        "data_days": data_days,
        "monthly_income_avg": int(monthly_income_avg),
        "predicted_expense_7d": predicted_7d,
        "risk_score": round(risk_score, 3),
        "level": 0 if insufficient else risk_level,
        "insufficient": insufficient,
    }
    return metrics


def store_metrics(db, user_id: int, metrics: dict) -> None:
    """upsert 指標到 financial_status，不動 summary（讓讀取端判斷是否重生成）。"""
    row = db.get(FinancialStatus, user_id)
    if row is None:
        row = FinancialStatus(user_id=user_id)
        db.add(row)
    row.metrics_json = json.dumps(metrics, ensure_ascii=False)
    row.metrics_hash = _metrics_hash(metrics)
    row.level = metrics["level"]
    row.insufficient = 1 if metrics["insufficient"] else 0


def compute_and_store(db, user_id: int, line_user_id: str, risk_result: dict | None) -> None:
    """記帳背景 thread 用：算指標 + 存 DB（caller 負責 commit）。"""
    metrics = build_metrics(db, line_user_id, risk_result)
    store_metrics(db, user_id, metrics)


def ensure_metrics(db, user_id: int, line_user_id: str):
    """讀取端用：若快取列不存在，拿既有的 risk_predictions 即時補算（不重跑推論）。

    回傳 FinancialStatus 列；若連風險預測都沒有則回 None（真正資料不足）。
    caller 負責 commit。
    """
    from backend.models.risk_prediction import RiskPrediction

    row = db.get(FinancialStatus, user_id)
    if row is not None:
        return row

    rp = db.get(RiskPrediction, user_id)
    risk_result = None
    if rp is not None:
        risk_result = {
            "data_days": rp.data_days or 0,
            "monthly_income_avg": rp.monthly_income_avg or 0,
            "risk_level": rp.risk_level or 0,
            "risk_score": rp.risk_ratio or 0,
            "predicted_expense_7d": rp.predicted_expense_7d or 0,
        }

    metrics = build_metrics(db, line_user_id, risk_result)
    store_metrics(db, user_id, metrics)
    return db.get(FinancialStatus, user_id)


def refresh_for_user(line_user_id: str) -> None:
    """web 端記帳後用：自開 session，跑風險預測 + 算指標 + 存。不發 LINE 通知。"""
    from backend.database import SessionLocal
    from backend.models.user import User
    from backend.models.risk_prediction import RiskPrediction
    from backend.ml_inference.bigru_service import predict_risk_for_user

    try:
        with SessionLocal() as db:
            user = db.query(User).filter(User.line_user_id == line_user_id).first()
            if user is None:
                return

            result = predict_risk_for_user(line_user_id, db)
            if result is not None:
                row = db.get(RiskPrediction, user.id)
                if row is None:
                    row = RiskPrediction(user_id=user.id)
                    db.add(row)
                row.predicted_expense_7d = result["predicted_expense_7d"]
                row.monthly_income_avg = result["monthly_income_avg"]
                row.risk_ratio = result["risk_score"]
                row.risk_level = result["risk_level"]
                row.alarm = result["alarm"]
                row.data_days = result["data_days"]

            compute_and_store(db, user.id, line_user_id, result)
            db.commit()
    except Exception as e:
        print(f"[financial_status] refresh_for_user 失敗：{repr(e)}")
