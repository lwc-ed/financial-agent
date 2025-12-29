from flask import Blueprint, request, jsonify
from backend.database import SessionLocal
from backend.models.user import User

liff_test_bp = Blueprint("liff_test", __name__)

@liff_test_bp.route("/api/check_user", methods=["POST"])
def check_user():
    data = request.get_json()
    line_user_id = data.get("line_user_id")

    if not line_user_id:
        return jsonify({"exists": False, "error": "missing line_user_id"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        return jsonify({
            "exists": user is not None
        })
    finally:
        db.close()