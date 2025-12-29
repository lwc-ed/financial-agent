from flask import Blueprint, request

liff_test_bp = Blueprint("liff_test", __name__)

@liff_test_bp.route("/api/liff_open_test", methods=["POST"])
def liff_open_test():
    data = request.get_json()
    print("=== LIFF OPEN TEST ===")
    print("User ID:", data.get("user_id"))
    print("======================")
    return {"ok": True}