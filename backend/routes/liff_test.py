from flask import Blueprint, request, jsonify
from backend.database import SessionLocal
from backend.models.user import User

liff_test_bp = Blueprint("liff_test", __name__)

@liff_test_bp.route("/api/check_user", methods=["POST"])
def check_user():
    data = request.get_json(silent=True) or {}
    print(f"DEBUG: 收到 data = {data}")  # ✅ 先印出來看前端傳什麼


    line_user_id = data.get("line_user_id")

    if not line_user_id:
        print(f"DEBUG: line_userid 是 None，data={data}")  # ✅ 印錯誤原因
        return jsonify({
            "exists": False,
            "error": "missing line_user_id"
        }), 400

    db = SessionLocal()
    try:
        user = (
            db.query(User)
            .filter_by(line_user_id=line_user_id)
            .first()
        )

        if not user:
            return jsonify({"exists": False}), 200

        return jsonify({
            "exists": True,
            "user": {
                "id": user.id,
                "name": user.name,
                "line_user_id": user.line_user_id,
            }
        }), 200
    finally:
        db.close()