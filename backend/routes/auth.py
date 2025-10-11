from flask import Blueprint, request, redirect, render_template, jsonify, session
import requests, jwt
from backend.database import SessionLocal
from models.user import User

auth_bp = Blueprint("auth", __name__)

GOOGLE_CLIENT_ID = "143127007053-vb0fqvjalcq31bff87j6cuh66fqd9amb.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
REDIRECT_URI = "https://financial-agent.it.com/callback_google"
SECRET_KEY = "super_secret_random_key"

@auth_bp.route("/login_page")
def login_page():
    return render_template("login.html")

@auth_bp.route("/login_google")
def login_google():
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        "&response_type=code"
        "&scope=openid%20email%20profile"
    )
    return redirect(google_auth_url)

@auth_bp.route("/callback_google")   # ← 這裡要用 auth_bp，不是 app
def callback_google():
    code = request.args.get("code")

    # 1. 用 code 換 access_token
    token_res = requests.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }).json()

    access_token = token_res.get("access_token")

    # 2. 用 access_token 拿 user info
    user_info = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"}
    ).json()

    # 3. 取出需要的欄位
    provider = "google"
    provider_id = user_info.get("id")
    name = user_info.get("name")
    email = user_info.get("email")

    # 4. 存 DB (用 SQLAlchemy ORM)
    db = SessionLocal()
    user = db.query(User).filter_by(provider=provider, provider_id=provider_id).first()

    if not user:
        user = User(provider=provider, provider_id=provider_id, name=name, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)  # 讓 user.id 更新
    else:
        # 如果已存在，更新基本資料（以防名字/信箱有變）
        user.name = name
        user.email = email
        db.commit()

    # 5. 發 JWT token（建議用資料庫 user.id）
    token = jwt.encode({"user_id": user.id}, SECRET_KEY, algorithm="HS256")

    # 6. 導到 success.html，並帶 google_id
    return render_template("success.html", google_id=provider_id)

@auth_bp.route("/bind", methods=["POST"])
def bind_user():
    data = request.json
    print("收到綁定請求:", data) 
    line_user_id = data.get("lineUserId")
    google_id = data.get("googleId")

    db = SessionLocal()
    user = db.query(User).filter_by(provider="google", provider_id=google_id).first()
    if not user:
        return jsonify({"status": "error", "msg": "找不到 Google 使用者"}), 404

    user.line_user_id = line_user_id
    db.commit()
    return jsonify({"status": "ok", "msg": "綁定完成"})