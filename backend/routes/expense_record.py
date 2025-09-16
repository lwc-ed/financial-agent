# routes/expense_record.py
from flask import Blueprint, request, jsonify

expense_record_bp = Blueprint("expense_record", __name__)

# ✅ 儲存消費紀錄
@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
    data = request.get_json()  # 例如: {"user_id": "xxx", "amount": 100, "category": "food"}
    user_id = data.get("user_id")
    amount = data.get("amount")
    category = data.get("category")

    # 這裡可以呼叫 DB 來存資料
    # db.save_expense(user_id, amount, category)

    return jsonify({"msg": f"已記錄 {user_id} 的消費：{amount} ({category})"})