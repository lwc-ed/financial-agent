from flask import Flask
from flask_cors import CORS
from backend.database import engine, Base
from backend.routes.auth import auth_bp
from backend.routes.linebot import linebot_bp
# from backend.linebot_handler import linebot_bp  # 舊版 handler 已註解
from backend.routes.expense_record import expense_record_bp
from backend.routes.expense_history import expense_history_bp
from backend.routes.wishlist import wishlist_bp
from backend.routes.saving_challenge import saving_challenge_bp
from backend.routes.profile import profile_bp
from backend.routes.liff_test import liff_test_bp
from backend.routes.dashboard import dashboard_bp
from dotenv import load_dotenv
import subprocess
import os
load_dotenv()

# 🔥 顯示目前 git commit（方便部署確認版本）
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    ).decode("utf-8").strip()
    print(f"🚀 Financial Agent starting... git commit = {git_commit}")
except Exception:
    print("🚀 Financial Agent starting... (git commit unknown)")

# 建立資料表
Base.metadata.create_all(bind=engine)

app = Flask(__name__)
CORS(app)


@app.route("/favicon.ico")
def favicon():
    return "", 204

# ✅ 使用新版 LINE Bot (linebot.v3) Blueprint
# 註冊 Blueprint
app.register_blueprint(auth_bp)
app.register_blueprint(linebot_bp)
app.register_blueprint(expense_record_bp, url_prefix="/api/expense_record")
app.register_blueprint(expense_history_bp, url_prefix="/api/expense_history")
app.register_blueprint(wishlist_bp, url_prefix="/api/wishlist")
#app.register_blueprint(saving_challenge_bp, url_prefix="/api/saving_challenge")
app.register_blueprint(profile_bp, url_prefix="/api/profile")
app.register_blueprint(liff_test_bp)
app.register_blueprint(dashboard_bp)
app.register_blueprint(saving_challenge_bp)


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
