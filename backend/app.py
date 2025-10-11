from flask import Flask
from flask_cors import CORS
from backend.database import engine, Base
from routes.auth import auth_bp
from routes.linebot import linebot_bp
from routes.expense_record import expense_record_bp
from routes.expense_history import expense_history_bp
from routes.wishlist import wishlist_bp
from routes.challenge import challenge_bp
from routes.profile import profile_bp
from dotenv import load_dotenv
load_dotenv()

# 建立資料表
Base.metadata.create_all(bind=engine)

app = Flask(__name__)
CORS(app)

# 註冊 Blueprint
app.register_blueprint(auth_bp)
app.register_blueprint(linebot_bp)
app.register_blueprint(expense_record_bp, url_prefix="/api/expense_record")
app.register_blueprint(expense_history_bp, url_prefix="/api/expense_history")
app.register_blueprint(wishlist_bp, url_prefix="/api/wishlist")
app.register_blueprint(challenge_bp, url_prefix="/api/challenge")
app.register_blueprint(profile_bp, url_prefix="/api/profile")

if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")