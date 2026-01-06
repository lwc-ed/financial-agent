from flask import Blueprint, request, jsonify, session
from backend.database import SessionLocal
from backend.models.user import User

liff_test_bp = Blueprint("liff_test", __name__)

@liff_test_bp.route("/api/check_user", methods=["POST"])
def check_user():
    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"exists": False}), 200

    db = SessionLocal()
    try:
        user = db.query(User).get(user_id)
        if not user:
            session.clear()
            return jsonify({"exists": False}), 200

        return jsonify({
            "exists": True,
            "user": {
                "id": user.id,
                "name": user.name,
                "line_user_id": user.line_user_id
            }
        }), 200
    finally:
        db.close()