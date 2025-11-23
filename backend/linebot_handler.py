"""
# linebot_handler.py
from flask import Blueprint, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from datetime import datetime, timedelta
from backend.database import SessionLocal
from models.user import User  # 假設你有 User model


linebot_bp = Blueprint("linebot", __name__)

# 替換成你自己的 LINE Channel Access Token 與 Secret
channel_access_token = "4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU="
channel_secret = "bde6ff24868fe4edeef87393ea9db525"

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

@linebot_bp.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ LINE 簽章驗證失敗")
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    user_msg = event.message.text
    reply_token = event.reply_token

    function_map = {
        "功能 A": "📊 消費分析（待接後端）",
        "功能 B": "📉 支出統計（待接 DB）",
        "功能 C": "🧾 記帳紀錄（待接 DB）",
        "功能 D": "💰 儲蓄進度（待接挑戰功能）",
        "功能 E": "⚠️ 預算提醒（待接分析功能）",
        "功能 F": "📝 功能說明：A=分析 B=統計 C=紀錄 D=挑戰 E=提醒"
    }

    db = SessionLocal()
    user = db.query(User).filter_by(line_user_id=user_id).first()

    # 如果使用者不存在，就建立
    if not user:
        user = User(
            line_user_id=user_id,
            current_function=None,
            last_activity_time=datetime.utcnow(),
            provider="line",
            provider_id=user_id,
            name="",
            email=""
        )
    db.add(user)
    db.commit()

    # 檢查是否超過 10 分鐘沒互動
    if user.last_activity_time and datetime.utcnow() - user.last_activity_time > timedelta(minutes=10):
        user.current_function = None
        db.commit()

    # 如果還沒選功能，阻擋文字輸入
    if not user.current_function and user_msg not in ["功能 A", "功能 B", "功能 C", "功能 D", "功能 E", "功能 F"]:
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text="請先點選功能")
        )
        return

    # 如果使用者選了功能 A~E，就更新 current_function
    if user_msg in ["功能 A", "功能 B", "功能 C", "功能 D", "功能 E"]:
        user.current_function = user_msg
        user.last_activity_time = datetime.utcnow()
        db.commit()
        reply_text = f"✅ 你選擇了 {user_msg}"
        line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text))
        return

    # 更新最後互動時間
    user.last_activity_time = datetime.utcnow()
    db.commit()

    # 其他功能回覆
    if user_msg == "紀錄消費":
        from backend.routes import expense_record
        reply_text = expense_record.get_expense_summary(user_id=user_id)
    else:
        reply_text = function_map.get(user_msg, f"你說的是：「{user_msg}」")

    line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text))
"""