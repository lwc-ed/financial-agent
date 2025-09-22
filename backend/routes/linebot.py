from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import MessagingApi, ReplyMessageRequest, TextMessage
from database import SessionLocal
from models.user import User

linebot_bp = Blueprint("linebot", __name__)

handler = WebhookHandler("你的 channel secret")
line_bot_api = MessagingApi("你的 channel access token")

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
    user = db.query(User).filter_by(line_user_id=line_user_id).first()

    if user:
        reply_text = f"嗨 {user.name}！你剛剛說：{user_msg}"
    else:
        reply_text = "嗨！你還沒綁定 Google 帳號，請先登入～"

    line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text=reply_text)]
        )
    )