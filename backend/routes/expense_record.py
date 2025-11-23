
'''# 紀錄消費 API
from flask import Blueprint, request, jsonify

# ✅ 名字一定要對
expense_record_bp = Blueprint("expense_record", __name__)

@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
    data = request.json
    # TODO: 之後存進 DB
    return jsonify({"status": "ok", "message": "已新增消費", "data": data}), 200'''



# backend/routes/expense_record.py
# 紀錄消費 API（SQLAlchemy 版）
from flask import Blueprint, request, jsonify
import re

from database import SessionLocal
from models.record import Record

expense_record_bp = Blueprint("expense_record", __name__)

def normalize_amount(val):
    """把 '$1,200'、'120元'、' 300 ' 轉為 int；不合法回 None"""
    if val is None:
        return None
    s = str(val)
    s = re.sub(r"[,\s\$＄元圓]", "", s)
    return int(s) if re.fullmatch(r"\d+", s) else None

@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
    """
    POST /api/expense_record/save
    JSON body：
    {
      "user_id": "Uxxxxxxxx",
      "type": "支出",              # '支出' / '收入'（預設 '支出'）
      "category": "午餐",         # 必填
      "amount": "$120元",         # 必填（可含 $、逗號、元/圓）
      "note": "公司附近便當"       # 可選
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "請以 JSON 傳送資料"}), 400

    line_user_id = (data.get("line_user_id") or "anonymous").strip()
    tx_type = (data.get("type") or "支出").strip()
    category = (data.get("category") or "").strip()
    amount = normalize_amount(data.get("amount"))
    note = (data.get("note") or "").strip()

    # 基本驗證
    if not category:
        return jsonify({"status": "error", "message": "category 必填"}), 400
    if amount is None or amount <= 0:
        return jsonify({"status": "error", "message": "amount 必須是正整數（可接受 120、$1,200、120元）"}), 400
    if tx_type not in ("支出", "收入"):
        return jsonify({"status": "error", "message": "type 只能是 '支出' 或 '收入'"}), 400

    db = SessionLocal()
    try:
        rec = Record(
            line_user_id=line_user_id,
            type=tx_type,
            category=category,
            amount=amount,
            note=note
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)

        return jsonify({
            "status": "ok",
            "message": "已新增消費",
            "data": {
                "id": rec.id,
                "line_user_id": rec.line_user_id,
                "type": rec.type,
                "category": rec.category,
                "amount": rec.amount,
                "note": rec.note,
                "timestamp": rec.timestamp.isoformat() if rec.timestamp else None
            }
        }), 200
    except Exception as e:
        db.rollback()
        return jsonify({"status": "error", "message": f"資料庫錯誤：{e}"}), 500
    finally:
        db.close()

