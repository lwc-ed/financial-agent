from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# 替換成你自己的 LINE Channel Access Token 與 Secret
channel_access_token = "4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU="
channel_secret = "bde6ff24868fe4edeef87393ea9db525"

app = Flask(__name__)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

@app.route("/callback", methods=['POST'])
def callback():
    # 取得 LINE 發送的簽名頭部
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

    # Rich Menu 六個功能對應文字（你可以改掉）
    function_map = {
        "功能 A": "📊 這是您的消費分析：\n（之後可以接後端查資料）",
        "功能 B": "📉 支出統計：\n食物 40%、交通 30%、娛樂 30%",
        "功能 C": "🧾 記帳紀錄如下：\n1. 早餐 $60\n2. 捷運 $25",
        "功能 D": "💰 儲蓄進度：\n已完成 72%，還差 NT$8,000",
        "功能 E": "⚠️ 預算提醒：\n本月已用 90%，建議減少娛樂支出",
        "功能 F": "📝 功能說明：\nA：消費分析\nB：支出統計\nC：記帳紀錄\n…"
    }

    # 根據功能文字回覆
    if user_message in function_map:
        reply_text = function_map[user_message]
    else:
        reply_text = f"你說的是：「{user_message}」"

    # 回傳訊息
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=reply_text)
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")