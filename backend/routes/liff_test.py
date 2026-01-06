from flask import Blueprint, request, jsonify
from backend.database import SessionLocal
from backend.models.user import User

liff_test_bp = Blueprint("liff_test", __name__)

@liff_test_bp.route("/api/check_user", methods=["POST"])
def check_user():
    try:
        data = request.get_json(silent=True) or {}
        print(f"🔍 RAW data: {data}")  # 原始收到的
        
        # 試兩種 key 名稱（防大小寫問題）
        line_userid = data.get('lineuserid') or data.get('lineUserId') or data.get('line_id')
        print(f"🔍 line_userid: '{line_userid}' (type: {type(line_userid)})")
        
        if not line_userid:
            print("❌ 沒有找到 line_userid")
            return jsonify({'exists': False, 'error': 'no line_userid in request'}), 400
        
        db = SessionLocal()
        try:
            user = db.query(User).filter_by(line_userid=line_userid).first()  # 注意這裡用 line_userid
            print(f"🔍 DB user found: {user is not None}")
            
            if not user:
                print("❌ DB 中沒找到 user")
                return jsonify({'exists': False, 'login_url': '/login_page'}), 200
            
            print("✅ User 存在，直接進 dashboard")
            return jsonify({
                'exists': True, 
                'dashboard_url': '/dashboard',
                'user': {
                    'id': user.id, 
                    'name': getattr(user, 'name', 'Unknown'),
                    'line_userid': line_userid
                }
            }), 200
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"💥 Error: {e}")
        return jsonify({'exists': False, 'error': str(e)}), 500
