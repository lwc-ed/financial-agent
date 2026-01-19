from flask import Blueprint, request, jsonify
from backend.database import SessionLocal
from backend.models.saving_challenge import SavingChallenge
from backend.models.user import User
from backend.models.wishlist import Wishlist  # 🔥 照 linebot.py 一樣


saving_challenge_bp = Blueprint(
    "saving_challenge",
    __name__,
    url_prefix="/api/saving-challenge"
)


@saving_challenge_bp.route("/create", methods=["POST"])
def create_challenge():
    data = request.get_json()

    line_user_id = data.get("line_user_id")
    item_name = data.get("item_name")
    target_amount = data.get("target_amount")
    pettype = data.get("pettype","chicken") #預設貓
    print(f"收到pettype:{pettype}") #debug


    if not all([line_user_id, item_name, target_amount]):
        return jsonify({"error": "missing fields"}), 400

    db = SessionLocal()

    try:
        # 1️⃣ 找 user
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        if not user:
            return jsonify({"error": "user not found"}), 404

        # 2️⃣ 檢查是否已存在
        exists = db.query(SavingChallenge).filter_by(
            user_id=user.id,
            item_name=item_name
        ).first()

        if exists:
            return jsonify({"error": "challenge already exists"}), 409

        # 3️⃣ 建立挑戰
        challenge = SavingChallenge(
            user_id=user.id,
            item_name=item_name,
            target_amount=target_amount,
            current_amount=0,
            stage=1,
            pettype = pettype
        )

        db.add(challenge)
        db.commit()

        return jsonify({
            "success": True,
            "challenge": {
                "item_name": challenge.item_name,
                "target_amount": float(challenge.target_amount),
                "current_amount": float(challenge.current_amount),
                "stage": challenge.stage
            }
        })

    finally:
        db.close()


# 只改 list 改 GET，其餘完美
@saving_challenge_bp.route("/list", methods=["GET"])
def list_challenges():
    line_user_id = request.args.get("line_user_id")
    if not line_user_id:
        return jsonify({"challenges": []}), 200

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        if not user:
            return jsonify({"challenges": []}), 200

        challenges = db.query(SavingChallenge).filter_by(user_id=user.id).all()
        
        result = []
        for challenge in challenges:  # 🔥 正確變數名
            pettype = getattr(challenge, 'pettype', 'chicken')
            print(f"🔥 /list 回傳: {challenge.item_name} pettype={pettype}")
            
            result.append({
                "item_name": challenge.item_name,      # 🔥 challenge，不是 c
                "target_amount": float(challenge.target_amount),
                "current_amount": float(challenge.current_amount),
                "stage": challenge.stage,
                "pettype": pettype
            })
        
        return jsonify({"challenges": result})
    finally:
        db.close()


# 🔥 新增願望清單 API
@saving_challenge_bp.route("/wishlist", methods=["GET"])
def get_wishlist():
    line_user_id = request.args.get("line_user_id")

    print(f"🔍 查詢 line_user_id: {line_user_id}")
    db = SessionLocal()
    
    try:
        #1️⃣line_user_id -> user.id
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        print(f"🔍 找到 user.id: {user.id if user else 'None'}")
        if not user:
            return jsonify({"wishlist": []}), 200
        

        # 2️⃣ user.id → wishlists.userid
        print(f"🔍 正在查找 user_id={user.id} 的願望清單資料庫")
        wishlists = db.query(Wishlist).filter_by(user_id=user.id).all()
        print(f"🔍 找到 {len(wishlists)} 筆願望")


        # 🔥 用已建立但未完成的挑戰當願望清單
        challenges = db.query(SavingChallenge)\
            .filter_by(user_id=user.id)\
            .filter(SavingChallenge.current_amount < SavingChallenge.target_amount)\
            .all()
        
        wishlist_data = [
            {
                "itemname": c.item_name,
                "price": float(c.price)
            }
            for c in wishlists
        ]
        print(f"🔍 回傳願望清單: {wishlist_data}")
        return jsonify({"wishlist": wishlist_data})
    finally:
        db.close()

    



@saving_challenge_bp.route("/feed", methods=["POST"])
def feed_challenge():
    data = request.get_json()

    line_user_id = data.get("line_user_id")
    item_name = data.get("item_name")
    amount = data.get("amount")

    if not all([line_user_id, item_name, amount]):
        return jsonify({"error": "missing fields"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        if not user:
            return jsonify({"error": "user not found"}), 404

        challenge = db.query(SavingChallenge).filter_by(
            user_id=user.id,
            item_name=item_name
        ).first()

        if not challenge:
            return jsonify({"error": "challenge not found"}), 404

        challenge.current_amount += amount

        # stage 計算（前端會做轉場，後端只存結果）
        ratio = float(challenge.current_amount) / float(challenge.target_amount)
        if ratio >= 1:
            challenge.stage = 4
        elif ratio >= 0.66:
            challenge.stage = 3
        elif ratio >= 0.33:
            challenge.stage = 2
        else:
            challenge.stage = 1

        db.commit()

        return jsonify({
            "success": True,
            "challenge": {
                "item_name": challenge.item_name,
                "target_amount": float(challenge.target_amount),
                "current_amount": float(challenge.current_amount),
                "stage": challenge.stage,
                "pettype": challenge.pettype
            }
        })
    finally:
        db.close()