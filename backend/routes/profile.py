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