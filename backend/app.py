from flask import Flask
from flask_cors import CORS

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

if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")