from flask import Blueprint, request, jsonify
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models.wishlist import Wishlist
from backend.models.user import User

wishlist_bp = Blueprint("wishlist", __name__)

@wishlist_bp.route("/add", methods=["POST"])
def add_wishlist():
    data = request.json
    
    # 1. 先取得原始資料，不要急著轉 int
    line_user_id = data.get("line_user_id")
    item_name = data.get("item_name")
    price_raw = data.get("price")

    # 2. 檢查必要欄位
    if not line_user_id or not item_name or not price_raw:
        return jsonify({"status": "error", "message": "缺少必要欄位"}), 400

    db: Session = SessionLocal()
    try:
        # 3. 檢查使用者是否存在
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        if not user:
            return jsonify({"status": "error", "message": "找不到使用者"}), 404

        # 4. 新增慾望清單 (這裡再轉 int)
        wishlist_item = Wishlist(
            user_id=user.id, 
            item_name=item_name, 
            price=int(price_raw)
        )
        db.add(wishlist_item)
        db.commit()

        # 5. 查詢目前清單回傳
        items = db.query(Wishlist).filter_by(user_id=user.id).all()
        result = [
            {
                "id": w.id, 
                "item_name": w.item_name, 
                "price": int(w.price), # 確保回傳也是數字
                "achieved": w.achieved
            } 
            for w in items
        ]
        
        return jsonify({"status": "ok", "data": result}), 200

    except Exception as e:
        db.rollback() # 發生錯誤時回滾
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        db.close() # 無論成功失敗，最後一定關閉連線