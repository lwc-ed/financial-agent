# linebot_handler.py
from flask import Blueprint, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

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
    user_message = event.message.text
    reply_token = event.reply_token

    function_map = {
        "功能 A": "📊 消費分析（待接後端）",
        "功能 B": "📉 支出統計（待接 DB）",
        "功能 C": "🧾 記帳紀錄（待接 DB）",
        "功能 D": "💰 儲蓄進度（待接挑戰功能）",
        "功能 E": "⚠️ 預算提醒（待接分析功能）",
        "功能 F": "📝 功能說明：A=分析 B=統計 C=紀錄 D=挑戰 E=提醒"
    }

    reply_text = function_map.get(user_message, f"你說的是：「{user_message}」")

    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=reply_text)
    )

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_msg = event.message.text

    if user_msg == "紀錄消費":
        # 呼叫 expense_record 裡的功能
        from routes import expense_record
        reply_text = expense_record.get_expense_summary(user_id=event.source.user_id)

        line_bot_api.reply_message(
            event.reply_token,
            TextMessage(text=reply_text)
        )