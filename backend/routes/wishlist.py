# 慾望清單 API
from flask import Blueprint, request, jsonify

# ✅ 名字一定要對：wishlist_bp
wishlist_bp = Blueprint("wishlist", __name__)

@wishlist_bp.route("/add", methods=["POST"])
def add_wishlist():
    """
    TODO: 新增一個慾望清單項目
    """
    data = request.json
    return jsonify({"status": "ok", "message": "Add wishlist (待實作)", "data": data}), 200

@wishlist_bp.route("/list", methods=["GET"])
def list_wishlist():
    """
    TODO: 查詢所有慾望清單項目
    """
    return jsonify({"status": "ok", "data": []}), 200