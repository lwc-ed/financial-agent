from flask import Blueprint, request, jsonify
from backend.database import SessionLocal
from backend.models.saving_challenge import SavingChallenge
from backend.models.user import User

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
            stage=1
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


@saving_challenge_bp.route("/list", methods=["POST"])
def list_challenges():
    data = request.get_json()
    line_user_id = data.get("line_user_id")

    if not line_user_id:
        return jsonify({"error": "missing line_user_id"}), 400

    db = SessionLocal()
    try:
        user = db.query(User).filter_by(line_user_id=line_user_id).first()
        if not user:
            return jsonify({"error": "user not found"}), 404

        challenges = db.query(SavingChallenge).filter_by(user_id=user.id).all()

        return jsonify({
            "challenges": [
                {
                    "item_name": c.item_name,
                    "target_amount": float(c.target_amount),
                    "current_amount": float(c.current_amount),
                    "stage": c.stage,
                    "created_at": c.created_at.isoformat()
                }
                for c in challenges
            ]
        })
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
                "stage": challenge.stage
            }
        })
    finally:
        db.close()