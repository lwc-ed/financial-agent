#個人資料API
# 個人資料 API
from flask import Blueprint, request, jsonify

# ✅ 名字要和 app.py 一致
profile_bp = Blueprint("profile", __name__)

@profile_bp.route("/set", methods=["POST"])
def set_profile():
    """
    TODO: 儲存使用者個人資料
    body 範例：
    {
      "userId": "U12345",
      "name": "小明",
      "age": 20,
      "goal": "存錢去日本"
    }
    """
    data = request.json
    # TODO: 存進 DB
    return jsonify({"status": "ok", "message": "已設定個人資料", "data": data}), 200

@profile_bp.route("/get", methods=["GET"])
def get_profile():
    """
    TODO: 從 DB 查詢使用者個人資料
    """
    return jsonify({"status": "ok", "data": {}}), 200

from flask import Blueprint, request, jsonify
from backend.database import SessionLocal
from backend.models.user import User   # 依你實際路徑調整

profile_bp = Blueprint("profile", __name__)

@profile_bp.route("/user/profile", methods=["POST"])
def get_user_profile():
    data = request.get_json()
    line_user_id = data.get("line_user_id")

    if not line_user_id:
        return jsonify({"error": "missing line_user_id"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        if not user:
            return jsonify({"error": "user not found"}), 404

        return jsonify({
            "name": user.name
        }), 200
    finally:
        db.close()