from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import MessagingApi, ReplyMessageRequest, TextMessage, Configuration, ApiClient
<<<<<<< HEAD
from backend.database import SessionLocal
from backend.models.user import User
from datetime import datetime, timedelta
=======
from database import SessionLocal
from models.user import User
>>>>>>> 26ce9af (same)

linebot_bp = Blueprint("linebot", __name__)

<<<<<<< HEAD
# 這裡填入你的 LINE Secret 與 Access Token
handler = WebhookHandler("bde6ff24868fe4edeef87393ea9db525")
configuration = Configuration(
    access_token="4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU="
)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)

=======
handler = WebhookHandler("bde6ff24868fe4edeef87393ea9db525")
<<<<<<< HEAD
line_bot_api = MessagingApi("4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU=")
>>>>>>> 49fd979 (same)
=======
configuration = Configuration(access_token="4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU=")
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
>>>>>>> 26ce9af (same)

@linebot_bp.route("/callback", methods=["POST"])
def callback():
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature")
    handler.handle(body, signature)
    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    line_user_id = event.source.user_id
    user_msg = event.message.text
    db = SessionLocal()

    # 查詢使用者
    user = db.query(User).filter_by(line_user_id=line_user_id).first()
    if not user:
        user = User(
            line_user_id=line_user_id,
            current_function=None,
            last_activity_time=datetime.utcnow(),
            provider="line",
            provider_id=line_user_id,
            name="",
            email=""
        )
        db.add(user)
        db.commit()

    # 超過 10 分鐘沒互動 → 重置
    if user.last_activity_time and datetime.utcnow() - user.last_activity_time > timedelta(minutes=10):
        user.current_function = None
        db.commit()

    # 功能別名對應表
    function_alias = {
        "紀錄消費": "功能 A",
        "慾望清單": "功能 B",
        "儲蓄挑戰": "功能 D",
        "消費記錄": "功能 C",
        "預算提醒": "功能 E"
    }

    # 如果 user_msg 在 function_alias，則轉換為對應功能
    if user_msg in function_alias:
        user_msg = function_alias[user_msg]

    # 功能對應表
    function_map = {
        "功能 A": "📊 消費分析（待接後端）",
        "功能 B": "📉 支出統計（待接 DB）",
        "功能 C": "🧾 記帳紀錄（待接 DB）",
        "功能 D": "💰 儲蓄進度（待接挑戰功能）",
        "功能 E": "⚠️ 預算提醒（待接分析功能）",
        "功能 F": "📝 功能說明：A=分析 B=統計 C=紀錄 D=挑戰 E=提醒"
    }

    # 回覆文字
    if not user.current_function and user_msg not in function_map:
        reply_text = "請先點選功能"
    elif user_msg in ["功能 A", "功能 B", "功能 C", "功能 D", "功能 E"]:
        user.current_function = user_msg
        user.last_activity_time = datetime.utcnow()
        db.commit()
        reply_text = f"✅ 你選擇了 {function_map[user_msg]}"
    elif user_msg == "紀錄消費":
        from routes import expense_record
        reply_text = expense_record.get_expense_summary(user_id=line_user_id)
    
    elif user_msg == "功能 B": 
        user.current_function = "wishlist"
        user.last_activity_time = datetime.utcnow()
        db.commit()
        reply_text = "✍️ 請輸入欲望清單項目，格式：品項,價格\n例如：iPhone,35000"
    
    elif user.current_function == "wishlist":
        try:
            item_name, price = user_msg.split(",")
            from routes.wishlist import wishlist_table
            from sqlalchemy import insert
            session = SessionLocal()
            session.execute(
                insert(wishlist_table).values(
                    user_id=user.id,
                    item_name=item_name.strip(),
                    price=int(price.strip())
                )
            )
            session.commit()
            session.close()
            reply_text = f"✅ 已新增「{item_name.strip()}」價格 {price.strip()} 元到清單！"
            # ✅ 新增完成後重置 current_function
            user.current_function = None
            db.commit()

        except Exception as e:
            reply_text = f"格式錯誤，請重新輸入：品項,價格\n錯誤：{e}"
    else:
        reply_text = function_map.get(user_msg, f"你說的是：「{user_msg}」")
    

    # 更新最後互動時間
    user.last_activity_time = datetime.utcnow()
    db.commit()

    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply_text)]
        )
    )

