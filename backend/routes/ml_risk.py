from flask import Blueprint, request, jsonify
from sqlalchemy import desc
from backend.database import SessionLocal
from backend.models.risk_prediction import RiskPrediction

ml_risk_bp = Blueprint("ml_risk", __name__)


@ml_risk_bp.get("/history")
def get_risk_history():
    """
    GET /api/ml/history?line_user_id=Uxxxx&limit=10
    回傳使用者最近幾筆風險預測紀錄。
    """
    line_user_id = (request.args.get("line_user_id") or "").strip()
    if not line_user_id:
        return jsonify({"status": "error", "message": "line_user_id 必填"}), 400

    try:
        limit = min(int(request.args.get("limit", 10)), 50)
    except ValueError:
        limit = 10

    with SessionLocal() as db:
        rows = (
            db.query(RiskPrediction)
            .filter(RiskPrediction.line_user_id == line_user_id)
            .order_by(desc(RiskPrediction.created_at))
            .limit(limit)
            .all()
        )
        data = [
            {
                "id": r.id,
                "predicted_expense_7d": r.predicted_expense_7d,
                "monthly_income_avg": r.monthly_income_avg,
                "risk_ratio": r.risk_ratio,
                "risk_level": r.risk_level,
                "alarm": r.alarm,
                "data_days": r.data_days,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

    return jsonify({"status": "ok", "data": data})
