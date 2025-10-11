from flask import Blueprint, request, jsonify
from database import engine  # 你的 SQLAlchemy engine
from sqlalchemy import Table, Column, Integer, String, Boolean, ForeignKey, MetaData, text
from sqlalchemy.orm import sessionmaker

# ✅ 名字一定要對
wishlist_bp = Blueprint("wishlist", __name__)

# SQLAlchemy session
Session = sessionmaker(bind=engine)
metadata = MetaData()

# 定義 wishlist_table
wishlist_table = Table(
    "wishlist", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("item_name", String(255)),
    Column("price", Integer),
    Column("achieved", Boolean, default=False)
)

# 確保資料表已經存在
metadata.create_all(engine)

# --- 新增一個慾望清單項目 ---
@wishlist_bp.route("/add", methods=["POST"])
def add_wishlist():
    data = request.json
    line_user_id = data.get("line_user_id")
    item_name = data.get("item_name")
    price = int(data.get("price"))
    if not line_user_id or not item_name or not price:
        return jsonify({"status": "error", "message": "缺少必要欄位"}), 400

    session = Session()
    # 先抓 user_id，用 text() 包 SQL
    user_id = session.execute(
        text("SELECT id FROM users WHERE line_user_id=:line_id"),
        {"line_id": line_user_id}
    ).scalar()

    if not user_id:
        session.close()
        return jsonify({"status": "error", "message": "找不到使用者"}), 404

    # 新增 wishlist
    session.execute(
        wishlist_table.insert().values(user_id=user_id, item_name=item_name, price=price)
    )
    session.commit()

    # 回傳最新清單
    rows = session.execute(
        wishlist_table.select().where(wishlist_table.c.user_id==user_id)
    ).fetchall()

    result = [
        {"id": r.id, "item_name": r.item_name, "price": r.price, "achieved": r.achieved}
        for r in rows
    ]
    session.close()
    return jsonify({"status": "ok", "data": result}), 200


# --- 查詢所有慾望清單項目 ---
@wishlist_bp.route("/list", methods=["GET"])
def list_wishlist():
    line_user_id = request.args.get("line_user_id")
    if not line_user_id:
        return jsonify({"status": "error", "message": "缺少 line_user_id"}), 400

    session = Session()
    # 統一用 line_user_id
    user_id = session.execute(
        text("SELECT id FROM users WHERE line_user_id=:line_id"),
        {"line_id": line_user_id}
    ).scalar()

    if not user_id:
        session.close()
        return jsonify({"status": "error", "message": "找不到使用者"}), 404

    rows = session.execute(
        wishlist_table.select().where(wishlist_table.c.user_id==user_id)
    ).fetchall()

    result = [
        {"id": r.id, "item_name": r.item_name, "price": r.price, "achieved": r.achieved}
        for r in rows
    ]
    session.close()
    return jsonify({"status": "ok", "data": result}), 200
