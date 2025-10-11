'''# 紀錄消費 API
# routes/expense_record.py
>>>>>>> 309f73e (hi)
from flask import Blueprint, request, jsonify

expense_record_bp = Blueprint("expense_record", __name__)

# ✅ 儲存消費紀錄
@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
'''# 紀錄消費 API
from flask import Blueprint, request, jsonify

# ✅ 名字一定要對
expense_record_bp = Blueprint("expense_record", __name__)

@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
    data = request.json
    # TODO: 之後存進 DB
    return jsonify({"status": "ok", "message": "已新增消費", "data": data}), 200
