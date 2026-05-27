from flask import Blueprint, render_template

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/login_page")
def login_page():
    return render_template("login.html")
