from flask_cors import CORS
from flask import Flask, request, redirect, jsonify
from flask import render_template
import requests
import jwt
import pymysql
from database import engine, Base
from models.record import Record   # 確保 Record 被載入
Base.metadata.create_all(bind=engine)

# 建立 MySQL 連線（改成你的 RDS 資訊）
password = "SUPERidol$" # 自動 escape 特殊字元

db = pymysql.connect(
    host="financial-agent.cpwk2ce8cqyu.us-east-2.rds.amazonaws.com",
    user="nycuiemagent",
    password=password,
    database="financial_agent",
    cursorclass=pymysql.cursors.DictCursor
)

# 初始化 Flask
app = Flask(__name__)
CORS(app)

# --- 載入 LINE Bot Handler ---
from linebot_handler import linebot_bp

# --- 載入各功能路由 (API) ---
from routes.expense_record import expense_record_bp
from routes.expense_history import expense_history_bp
from routes.wishlist import wishlist_bp
from routes.challenge import challenge_bp
from routes.profile import profile_bp

#設定 Google OAuth 參數
GOOGLE_CLIENT_ID = "143127007053-vb0fqvjalcq31bff87j6cuh66fqd9amb.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
REDIRECT_URI = "https://financial-agent.it.com/callback_google"
SECRET_KEY = "super_secret_random_key_123456789"

# 註冊 LINE webhook
app.register_blueprint(linebot_bp, url_prefix="")
#

app.register_blueprint(expense_record_bp, url_prefix="/api/expense_record")
app.register_blueprint(expense_history_bp, url_prefix="/api/expense_history")
app.register_blueprint(wishlist_bp, url_prefix="/api/wishlist")
app.register_blueprint(challenge_bp, url_prefix="/api/challenge")
app.register_blueprint(profile_bp, url_prefix="/api/profile")

# 測試用 API
@app.route("/api/hello")
def hello():
    return {"message": "Hello from Flask!"}

# Webhook 用
@app.route("/callback", methods=["POST"])
def callback():
    body = request.get_data(as_text=True)  # 讀取 LINE 傳來的 raw body
    print("Request body:", body)  # 可以先印出來測試
    return "OK"


#登入介面
@app.route("/login_page")
def login_page():
    return render_template("login.html")

#加登入路由 /login_google
@app.route("/login_google")
def login_google():
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        "&response_type=code"
        "&scope=openid%20email%20profile"
    )
    return redirect(google_auth_url)

@app.route("/callback_google")
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

    # 4. 存 DB
    with db.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE provider=%s AND provider_id=%s", (provider, provider_id))
        existing_user = cursor.fetchone()

        if not existing_user:
            cursor.execute(
                "INSERT INTO users (provider, provider_id, name, email, picture) VALUES (%s, %s, %s, %s, %s)",
                (provider, provider_id, name, email, picture)
            )
            db.commit()
            user_id = cursor.lastrowid
        else:
            user_id = existing_user["id"]

    # 5. 發 JWT token（建議用資料庫 user_id 而不是 Google id）
    token = jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm="HS256")

    return jsonify({
        "access_token": token,
        "user": user_info
    })

if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
