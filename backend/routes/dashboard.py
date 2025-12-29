# 放在任一你習慣的 routes 檔案，例如 backend/routes/dashboard.py
from flask import Blueprint

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/dashboard")
def dashboard():
    return "✅ 登入成功，這裡是 Dashboard（暫時頁面）"