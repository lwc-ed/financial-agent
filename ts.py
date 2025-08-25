#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linebot.v3.messaging import Configuration, MessagingApi, ApiClient
from linebot.v3.messaging.models import (
    RichMenuRequest,
    RichMenuSize,
    RichMenuArea,
    RichMenuBounds,
    MessageAction
)
import os
from mimetypes import guess_type

# ✅ 請換成你自己的 Channel Access Token
channel_access_token = "4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU="


# ✅ 圖片路徑
image_path = "picture/dy.png"

# ✅ 正確初始化方式（包含 host）
configuration = Configuration(
    access_token=channel_access_token,
    host="https://api.line.me"
)

with ApiClient(configuration) as api_client:
    messaging_api = MessagingApi(api_client)

    # 🧩 建立 Rich Menu
    print("📦 Creating rich menu...")
    rich_menu = RichMenuRequest(
        size=RichMenuSize(width=2500, height=843),
        selected=True,
        name="FinanceHelperMenu",
        chat_bar_text="Open Menu",
        areas=[
            RichMenuArea(
                bounds=RichMenuBounds(x=0, y=0, width=833, height=843),
                action=MessageAction(label="A", text="Function A")
            ),
            RichMenuArea(
                bounds=RichMenuBounds(x=833, y=0, width=833, height=843),
                action=MessageAction(label="B", text="Function B")
            ),
            RichMenuArea(
                bounds=RichMenuBounds(x=1666, y=0, width=834, height=843),
                action=MessageAction(label="C", text="Function C")
            ),
        ]
    )
    created_menu = messaging_api.create_rich_menu(rich_menu)
    rich_menu_id = created_menu.rich_menu_id
    print("✅ Rich Menu created! ID:", rich_menu_id)

    # ✅ 上傳圖片
    if os.path.exists(image_path):
        print("🖼️  Uploading image...")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        mime_type = guess_type(image_path)[0] or "image/png"

        api_client.call_api(
            resource_path=f"/v2/bot/richmenu/{rich_menu_id}/content",
            method="POST",
            body=image_bytes,
            headers={"Content-Type": mime_type},
            auth_settings=["Authorization"],
            response_type=None,
            _return_http_data_only=True
        )
        print("✅ Image uploaded successfully!")
    else:
        print("❌ Image file not found:", image_path)
        exit(1)

    # ✅ 設定為預設 Rich Menu
    print("📌 Setting as default rich menu...")
    messaging_api.set_default_rich_menu(rich_menu_id)
    print("🎉 Success! Default rich menu has been set.")