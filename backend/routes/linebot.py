from flask import Blueprint, request
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import MessagingApi, ReplyMessageRequest, TextMessage, Configuration, ApiClient
from backend.database import SessionLocal
from backend.models.user import User
from backend.models.wishlist import Wishlist
from datetime import datetime, timedelta

linebot_bp = Blueprint("linebot", __name__)

# 這裡填入你的 LINE Secret (請確認是否正確)
handler = WebhookHandler("bde6ff24868fe4edeef87393ea9db525")

# 👇👇👇 請務必換成您剛剛重新發行的 Token 👇👇👇
configuration = Configuration(
    access_token="4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU="
)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)

@linebot_bp.route("/callback", methods=["POST"])
def callback():
    body = request.get_data(as_text=True)
    signature = request.headers.get("X-Line-Signature")
    try:
        handler.handle(body, signature)
    except Exception as e:
        print(f"Webhook Signature Error: {e}")
        return "Invalid signature", 400
    return "OK"

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    print(f"收到訊息: {event.message.text}")
    
    line_user_id = event.source.user_id
    user_msg = event.message.text
    
    db = SessionLocal()
    
    # 1. 查詢或建立使用者
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

    # 2. 超時重置狀態 (10分鐘)
    if user.last_activity_time and datetime.utcnow() - user.last_activity_time > timedelta(minutes=10):
        user.current_function = None
        db.commit()

    # 3. 功能別名對應
    function_alias = {
        "紀錄消費": "功能 A",
        "慾望清單": "功能 B",
        "儲蓄挑戰": "功能 D",
        "消費記錄": "功能 C",
        "預算提醒": "功能 E"
    }
    if user_msg in function_alias:
        user_msg = function_alias[user_msg]

    # 4. 功能說明對應表
    function_map = {
        "功能 A": " 📊  消費分析（待接後端）",
        "功能 B": " 📉  慾望清單",
        "功能 C": " 🧾  記帳紀錄（待接 DB）",
        "功能 D": " 💰  儲蓄進度（待接挑戰功能）",
        "功能 E": " ⚠️  預算提醒（待接分析功能）",
        "功能 F": " 📝  功能說明：A=分析 B=統計 C=紀錄 D=挑戰 E=提醒"
    }

    reply_text = ""

    # 5. 邏輯判斷
    
    # 若無狀態且非指令
    if not user.current_function and user_msg not in function_map:
        reply_text = "請先點選功能"

    # 一般功能 (排除功能 B，因為要獨立處理)
    elif user_msg in ["功能 A", "功能 C", "功能 D", "功能 E"]:
        user.current_function = user_msg
        db.commit()
        reply_text = f" ✅  你選擇了 {function_map[user_msg]}"

    elif user_msg == "紀錄消費":
        reply_text = "紀錄消費功能開發中..."

    # --- 慾望清單入口 (關鍵修正：強制設定狀態為 wishlist) ---
    elif user_msg == "功能 B":
        print("進入慾望清單模式")
        user.current_function = "wishlist"  # 這裡一定要設為 wishlist
        db.commit()
        reply_text = "✍️ 請輸入欲望清單項目，格式：品項,價格\n例如：iPhone,35000"

    # --- 慾望清單輸入處理 ---
    elif user.current_function == "wishlist":
        print(f"處理慾望清單輸入: {user_msg}")
        try:
            # 處理中英文逗號
            separator = "," if "," in user_msg else "，"
            if separator not in user_msg:
                raise ValueError("缺少逗號")

            item_name, price_str = user_msg.split(separator, 1)
            
            # ORM 寫入
            new_item = Wishlist(
                user_id=user.id,
                item_name=item_name.strip(),
                price=int(price_str.strip())
            )
            db.add(new_item)
            db.commit()
            
            reply_text = f"✅ 已新增「{item_name.strip()}」價格 {price_str.strip()} 元到清單！"
            
            # 完成後重置
            user.current_function = None
            db.commit()
            
        except ValueError:
            reply_text = "格式錯誤！請確認使用逗號分隔，且價格為數字。\n範例：Switch, 10000"
        except Exception as e:
            reply_text = f"發生錯誤：{str(e)}"
            print(f"Error: {e}")

    else:
        reply_text = function_map.get(user_msg, f"你說的是：「{user_msg}」")

    # 6. 更新時間並回覆
    user.last_activity_time = datetime.utcnow()
    db.commit()
    db.close()

    print(f"準備回覆: {reply_text}")
    
    try:
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )
    except Exception as e:
        print(f"LINE 回覆失敗: {e}")
