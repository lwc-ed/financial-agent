import os
from flask import Blueprint, request, jsonify
from linebot.v3.messaging import MessagingApi, ApiClient, Configuration
from backend.database import SessionLocal
from backend.models.user import User

liff_test_bp = Blueprint("liff_test", __name__)

_line_api = MessagingApi(ApiClient(Configuration(
    access_token=os.getenv("CHANNEL_ACCESS_TOKEN", "")
)))

@liff_test_bp.route("/api/check_user", methods=["POST"])
def check_user():
    try:
        data = request.get_json(silent=True) or {}
        print(f"🔍 RAW data: {data}")

        line_userid = data.get('lineuserid') or data.get('lineUserId') or data.get('line_id')
        print(f"🔍 line_userid: '{line_userid}'")

        if not line_userid:
            return jsonify({'exists': False, 'error': 'no line_user_id'}), 400

        db = SessionLocal()
        try:
            user = db.query(User).filter_by(line_user_id=line_userid).first()
            print(f"🔍 DB user found: {user is not None}")

            if not user:
                display_name = data.get('displayName') or data.get('display_name')
                if not display_name:
                    try:
                        profile = _line_api.get_profile(line_userid)
                        display_name = profile.display_name
                    except Exception:
                        display_name = "LINE User"
                user = User(
                    provider="line",
                    provider_id=line_userid,
                    name=display_name,
                    email=None,
                    line_user_id=line_userid,
                )
                db.add(user)
                try:
                    db.commit()
                    db.refresh(user)
                except Exception:
                    db.rollback()
                    user = db.query(User).filter_by(line_user_id=line_userid).first()

            print("✅ User 存在，直接 dashboard！")
            return jsonify({
                'exists': True,
                'dashboard_url': '/dashboard',
                'user': {
                    'id': user.id,
                    'name': getattr(user, 'name', 'Unknown'),
                    'line_user_id': line_userid
                }
            }), 200

        finally:
            db.close()

    except Exception as e:
        print(f"💥 Error: {e}")
        return jsonify({'exists': False, 'error': str(e)}), 500
