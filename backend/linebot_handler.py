from flask import request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from app import app

# 替換成你自己的 LINE Channel Access Token 與 Secret
channel_access_token = "YOUR_CHANNEL_ACCESS_TOKEN"
channel_secret = "YOUR_CHANNEL_SECRET"

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

@app.route("/callback", methods=['POST'])
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

    # TODO: 這裡之後可以改成呼叫 Flask API 查資料
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