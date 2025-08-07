from flask import Flask, request, abort
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient, TextMessage, ReplyMessageRequest
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.exceptions import InvalidSignatureError

# 替換為你自己的 Channel 資訊
channel_access_token = "1jSCc+gN+z8FXOJgEW5Qpn15fhV4fOr922yWMFlEduh/5jB5dvz9ZEeqYxp7oogBtaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFMfS4FGElntwIIaXiHsL3kdgKInZ+c22MHKXyEJivvbsgdB04t89/1O/w1cDnyilFU="
channel_secret = "bde6ff24868fe4edeef87393ea9db525"

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    text = event.message.text
    user_id = event.source.user_id

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=f"你說的是：「{text}」")]
            )
        )

if __name__ == "__main__":
    app.run(debug=True, port=5000)