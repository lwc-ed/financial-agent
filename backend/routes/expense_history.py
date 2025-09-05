# 消費紀錄 API(查詢)
from flask import Blueprint, jsonify

expense_history_bp = Blueprint("expense_history", __name__)

# 假資料（暫時用，未來接 DB）
expenses = [
    {"userId": "U12345", "amount": 100, "category": "餐飲", "note": "早餐", "date": "2025-09-06"}
]

@expense_history_bp.route("/list", methods=["GET"])
def list_expense():
    return jsonify({"status": "ok", "data": expenses}), 200