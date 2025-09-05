# 紀錄消費 API
from flask import Blueprint, request, jsonify

# ✅ 名字一定要對
expense_record_bp = Blueprint("expense_record", __name__)

@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
    data = request.json
    # TODO: 之後存進 DB
    return jsonify({"status": "ok", "message": "已新增消費", "data": data}), 200