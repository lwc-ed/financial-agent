#儲蓄挑戰API# 儲蓄挑戰 API
from flask import Blueprint, jsonify

# ✅ 名字一定要對：challenge_bp
challenge_bp = Blueprint("challenge", __name__)

@challenge_bp.route("/progress", methods=["GET"])
def get_challenge_progress():
    """
    TODO: 查詢使用者目前儲蓄挑戰進度
    """
    return jsonify({"status": "ok", "progress": 0}), 200