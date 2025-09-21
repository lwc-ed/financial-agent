'''# 紀錄消費 API
from flask import Blueprint, request, jsonify

# ✅ 名字一定要對
expense_record_bp = Blueprint("expense_record", __name__)

@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
    data = request.json
    # TODO: 之後存進 DB
    return jsonify({"status": "ok", "message": "已新增消費", "data": data}), 200'''



# routes/expense_record.py
# 紀錄消費 API
from flask import Blueprint, request, jsonify
from app import db  # 從 app.py 匯入既有的 MySQL 連線物件
import re

# ✅ Blueprint 名稱要和 app.register_blueprint 時一致
expense_record_bp = Blueprint("expense_record", __name__)

def normalize_amount(val):
    """
    把 '120', '120元', '$1,200', '  300 ' 轉成 int；不合法回傳 None
    """
    if val is None:
        return None
    s = str(val)
    s = re.sub(r"[,\s\$＄元圓]", "", s)
    return int(s) if re.fullmatch(r"\d+", s) else None

@expense_record_bp.route("/save", methods=["POST"])
def save_expense():
    """
    POST /api/expense_record/save
    Content-Type: application/json

    範例 body：
    {
      "user_id": "Uxxxxxxxx",     # 可選，預設 'anonymous'
      "type": "支出",              # 可選，'支出' 或 '收入'，預設 '支出'
      "category": "午餐",         # 必填
      "amount": "$120元",         # 必填（可含 $、逗號、元/圓）
      "note": "公司附近便當"       # 可選
      // timestamp 交給 MySQL 用 NOW() 產生
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "請以 JSON 傳送資料"}), 400

    user_id = (data.get("user_id") or "anonymous").strip()
    tx_type = (data.get("type") or "支出").strip()
    category = (data.get("category") or "").strip()
    amount = normalize_amount(data.get("amount"))
    note = (data.get("note") or "").strip()

    # --- 基本驗證 ---
    if not category:
        return jsonify({"status": "error", "message": "category 必填"}), 400
    if amount is None or amount <= 0:
        return jsonify({"status": "error", "message": "amount 必須是正整數（可接受 120、$1,200、120元）"}), 400
    if tx_type not in ("支出", "收入"):
        return jsonify({"status": "error", "message": "type 只能是 '支出' 或 '收入'"}), 400

    # --- 寫入 MySQL ---
    try:
        with db.cursor() as cursor:
            sql = """
                INSERT INTO records (user_id, type, category, amount, note, timestamp)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(sql, (user_id, tx_type, category, amount, note))
            db.commit()
            new_id = cursor.lastrowid
    except Exception as e:
        # 若要更乾淨可以改成統一訊息，並把 e 寫入 log
        return jsonify({"status": "error", "message": f"資料庫錯誤：{e}"}), 500

    # --- 成功回傳 ---
    return jsonify({
        "status": "ok",
        "message": "已新增消費",
        "data": {
            "id": new_id,
            "user_id": user_id,
            "type": tx_type,
            "category": category,
            "amount": amount,
            "note": note
        }
    }), 200
