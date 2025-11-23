# linebot_handler.py
from flask import Blueprint, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from database import SessionLocal
from models.record import Record
from sqlalchemy import desc
import re

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

def parse_expense_text(text: str):
    """
    把像「午餐 150」這種訊息，拆成 (category, amount)
    回傳：(category:str, amount:int) 或 (None, None) 代表不是合法記帳格式
    """
    text = text.strip()
    # 用空白拆成兩個
    parts = text.split()
    if len(parts) != 2:
        return None, None

    category = parts[0]
    amount_raw = parts[1]

    # 把 150, 150元, $150 都變成 int
    s = re.sub(r"[,\s\$＄元圓]", "", amount_raw)
    if not re.fullmatch(r"\d+", s):
        return None, None

    return category, int(s)


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = (event.message.text or "").strip()
    reply_token = event.reply_token
    line_user_id = event.source.user_id

    # 功能選單（保留你原本的功能鍵說明）
    function_map = {
        "功能 A": "📊 消費分析（之後會接後端統計功能）",
        "功能 B": "📉 支出統計（之後會接 DB 聚合）",
        "功能 C": "🧾 記帳紀錄：可以輸入「查紀錄」查看最近 5 筆",
        "功能 D": "💰 儲蓄進度（之後會接挑戰 / 目標功能）",
        "功能 E": "⚠️ 預算提醒（之後會接分析功能）",
        "功能 F": "📝 使用說明：\n- 記帳：午餐 150\n- 查紀錄：查紀錄"
    }

    # 1) 功能選單的純文字按鈕（保留你原本邏輯）
    if user_message in function_map:
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=function_map[user_message])
        )
        return

    # 2) 使用者輸入「紀錄消費」時，先給他教學
    if user_message == "紀錄消費":
        help_text = (
            "要記帳的話，可以直接輸入：\n"
            "・午餐 150\n"
            "・飲料 60\n"
            "・捷運 30\n\n"
            "我會自動幫你記成【支出】唷！"
        )
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=help_text)
        )
        return

    # 3) 使用者輸入「查紀錄」→ 撈最近 5 筆
    if user_message == "查紀錄":
        db = SessionLocal()
        try:
            q = (
                db.query(Record)
                  .filter(Record.line_user_id == line_user_id)
                  .order_by(desc(Record.timestamp), desc(Record.id))
                  .limit(5)
            )
            rows = q.all()

            if not rows:
                reply_text = "你目前還沒有任何記帳紀錄喔～\n可以試試輸入：午餐 150"
            else:
                lines = ["你最近的記帳紀錄："]
                for r in rows:
                    line = f"- {r.category} {r.amount} 元"
                    if r.note:
                        line += f"（{r.note}）"
                    lines.append(line)
                reply_text = "\n".join(lines)
        except Exception as e:
            print("查紀錄錯誤：", e)
            reply_text = "查詢紀錄時發生錯誤 QQ，等等再試試。"
        finally:
            db.close()

        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=reply_text)
        )
        return

    # 4) 嘗試把一般訊息當成記帳輸入：「午餐 150」
    category, amount = parse_expense_text(user_message)
    if category is not None and amount is not None:
        db = SessionLocal()
        try:
            rec = Record(
                line_user_id=line_user_id,
                type="支出",          # 目前全部當支出，有需要再做收入指令
                category=category,
                amount=amount,
                note=""
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)

            reply_text = f"已幫你記錄：{category} {amount} 元 ✅"
        except Exception as e:
            print("記帳寫入錯誤：", e)
            db.rollback()
            reply_text = "記帳失敗 QQ，等等再試試看。"
        finally:
            db.close()

        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=reply_text)
        )
        return

    # 5) 其他文字 → 當成一般聊天 + 提醒可以用哪些功能
    default_text = (
        f"你說的是：「{user_message}」\n\n"
        "目前可以這樣跟我互動：\n"
        "・記帳：直接輸入「午餐 150」\n"
        "・查紀錄：輸入「查紀錄」\n"
        "・看功能說明：點「功能 F」"
    )
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=default_text)
    )
