# backend/routes/expense_history.py
from flask import Blueprint, request, jsonify
from sqlalchemy import desc
from backend.database import SessionLocal
from backend.models.record import Record

# 給 app.py 用的 Blueprint 名稱
expense_history_bp = Blueprint("expense_history", __name__)

@expense_history_bp.get("/recent")
def get_recent_expenses():
    """
    GET /api/expense_history/recent

    Query Params:
      - line_user_id: 必填（使用者的 LINE ID）
      - limit: 選填，要抓幾筆，預設 10，上限 100
      - offset: 選填，預設 0（做分頁用）
      - type: 選填，"支出" 或 "收入"（不填就兩種都查）
      - category: 選填，類別名稱（例如 午餐、交通）
    """
    line_user_id = (request.args.get("line_user_id") or "").strip()
    if not line_user_id:
        return jsonify({"status": "error", "message": "line_user_id 必填"}), 400

    # limit / offset 做分頁
    try:
        limit = int(request.args.get("limit", 10))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"status": "error", "message": "limit / offset 必須是整數"}), 400

    # 限制最大值，避免一次撈太多
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    tx_type = request.args.get("type")      # "支出" / "收入" / None
    category = request.args.get("category") # 例如 "午餐"

    db = SessionLocal()
    try:
        # 基本條件：只查這個使用者的紀錄
        query = db.query(Record).filter(Record.line_user_id == line_user_id)

        # 如果有指定 type，就再加條件
        if tx_type in ("支出", "收入"):
            query = query.filter(Record.type == tx_type)

        # 如果有指定 category，就再加條件
        if category:
            query = query.filter(Record.category == category)

        # 先算總筆數（用於前端做分頁）
        total = query.count()

        # 依時間新到舊排序
        rows = (
            query.order_by(desc(Record.timestamp), desc(Record.id))
                 .offset(offset)
                 .limit(limit)
                 .all()
        )

        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "line_user_id": r.line_user_id,
                "type": r.type,
                "category": r.category,
                "amount": r.amount,
                "note": r.note,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            })

        return jsonify({
            "status": "ok",
            "message": "查詢成功",
            "meta": {
                "total": total,
                "limit": limit,
                "offset": offset,
                "returned": len(data),
            },
            "data": data
        }), 200

    except Exception as e:
        # 開發中先把錯誤丟回去看，之後可以改成寫 log
        return jsonify({"status": "error", "message": f"資料庫錯誤：{e}"}), 500
    finally:
        db.close()
