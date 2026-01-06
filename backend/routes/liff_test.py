from flask import Blueprint, request, jsonify
from backend.database import SessionLocal
from backend.models.user import User

liff_test_bp = Blueprint("liff_test", __name__)

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
            # 🔥 正確欄位名：line_user_id（有底線）
            user = db.query(User).filter_by(line_user_id=line_userid).first()
            print(f"🔍 DB user found: {user is not None}")
            
            if not user:
                print("❌ 沒找到 user → 跳 Google 登入")
                return jsonify({
                    'exists': False, 
                    'login_url': '/login_page'
                }), 200
            
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
