#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linebot.v3.messaging import Configuration, MessagingApi, MessagingApiBlob, ApiClient
from linebot.v3.messaging import ApiException
from linebot.v3.messaging.models import (
    RichMenuRequest,
    RichMenuSize,
    RichMenuArea,
    RichMenuBounds,
    MessageAction,
    URIAction
)
import os
from mimetypes import guess_type
import requests
import time



# ✅ 請換成你自己的 Channel Access Token
channel_access_token = "4CtUYyGR0+ISjVhzcnGLmJmG8Qf/vzH5/gQM98g/jR2ZoMZguJPkvjiLvMXoSb3ctaKkMO7Onhe6Fa1bc3BHw6sF7coKlYy1dozA7/V6ZFOpt9S9wU8PXZhefQoOGtC2J6fj70vQzIqNktiQVx2MdAdB04t89/1O/w1cDnyilFU="


# ✅ 圖片路徑
image_path = "/Users/liweichen/financial-agent/picture/menu1.jpg"


# ✅ 正確初始化方式（包含 host）
configuration = Configuration(
    access_token=channel_access_token,
    host="https://api.line.me"
)

# Blob 專用 host（上傳圖片等二進位內容需要走 api-data）
blob_configuration = Configuration(
    access_token=channel_access_token,
    host="https://api-data.line.me"
)

# --- Helper: robust uploader for v3.18.1 and nearby versions ---
def _upload_rich_menu_image_any(api_client: ApiClient, blob_api: MessagingApiBlob, rich_menu_id: str, image_bytes: bytes, mime_type: str) -> None:
    """Try Blob API with _headers first; if it fails (signature/version diffs), fallback to low-level call_api."""
    # 1) Preferred path: Blob API with _headers (v3.18.1 接受)
    try:
        blob_api.set_rich_menu_image(
            rich_menu_id=rich_menu_id,
            body=image_bytes,
            _headers={"Content-Type": mime_type}
        )
        return
    except Exception as e1:
        print("⚠️ Blob API upload failed, fallback to low-level call_api. Reason:", e1)

    # 2) Fallback: direct HTTP via requests (bypass SDK signature differences)
    url = f"https://api.line.me/v2/bot/richmenu/{rich_menu_id}/content"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {channel_access_token}",
            "Content-Type": mime_type,
        },
        data=image_bytes,
        timeout=30,
    )
    if resp.status_code >= 400:
        raise ApiException(status=resp.status_code, reason=resp.reason, http_resp=resp)

with ApiClient(configuration) as api_client, ApiClient(blob_configuration) as blob_client:
    messaging_api = MessagingApi(api_client)
    blob_api = MessagingApiBlob(blob_client)

    # 🧩 建立 Rich Menu
    print("📦 Creating rich menu...")
    rich_menu = RichMenuRequest(
        size=RichMenuSize(width=2500, height=1686),
        selected=True,
        name="FinanceHelperMenu",
        chat_bar_text="Menu",
        areas = [
            # 上三個區域
            RichMenuArea(
                bounds=RichMenuBounds(x=0, y=0, width=833, height=843),
                action=MessageAction(label="A", text="信用卡回饋查詢")
            ),
            RichMenuArea(
                bounds=RichMenuBounds(x=833, y=0, width=833, height=843),
                action=MessageAction(label="B", text="欲望清單")
            ),
            RichMenuArea(
                bounds=RichMenuBounds(x=1666, y=0, width=834, height=1686),
                action=MessageAction(label="C",text="紀錄消費")
            ),
            
            # 下兩個區域
            RichMenuArea(
                bounds=RichMenuBounds(x=0, y=843, width=833, height=843),
                action=URIAction(label="D", uri="line://app/2008065321-vlAGLNjW")
            ),
            RichMenuArea(
                bounds=RichMenuBounds(x=833, y=843, width=833, height=843),
                action=MessageAction(label="E", text="儲蓄挑戰")
            ),
        ]
    )
    #line://app/2008065321-vlAGLNjW
    #https://liff.line.me/2008795054-LpCQ6Gdh
    created_menu = messaging_api.create_rich_menu(rich_menu)
    rich_menu_id = created_menu.rich_menu_id
    print("✅ Rich Menu created! ID:", rich_menu_id)

    # 額外驗證：列出所有 rich menu，確認剛建立的 ID 是否在清單中
    try:
        rm_list = messaging_api.get_rich_menu_list()
        # v3.18.1 使用 `richmenus`；部分版本可能是 `rich_menus`
        menus = getattr(rm_list, "richmenus", None) or getattr(rm_list, "rich_menus", None) or []
        ids = []
        for rm in menus:
            rid = getattr(rm, "rich_menu_id", None) or getattr(rm, "richMenuId", None)
            if rid:
                ids.append(rid)
        print("🧾 Rich menus on server:", ids)
        if rich_menu_id not in ids:
            print("❌ Newly created rich menu ID not found in list. Abort.")
            exit(1)
    except ApiException as le:
        print("⚠️ Failed to list rich menus:", le)

    # 有些節點需要一點時間才一致，等待片刻再上傳
    time.sleep(1.0)

    # ✅ 上傳圖
    if os.path.exists(image_path):
        print("🖼️  Uploading image...")
        try:
            # 先確認這個 rich_menu_id 在當前 channel 下是否存在
            try:
                _ = messaging_api.get_rich_menu(rich_menu_id)
            except ApiException as ge:
                print("❌ Rich menu not found on server (get_rich_menu 失敗):", ge)
                exit(1)

            # 強制用標準 MIME，避免奇怪副檔名造成的 x-png 類型
            ext = os.path.splitext(image_path)[1].lower()
            if ext in [".png"]:
                mime_type = "image/png"
            elif ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            else:
                print("❌ Unsupported image type. Use PNG or JPG:", image_path)
                exit(1)

            file_size = os.path.getsize(image_path)
            print(f"📏 Image file size: {file_size} bytes, MIME: {mime_type}, PATH: {image_path}")
            if file_size == 0:
                print("❌ Image file is empty.")
                exit(1)
            # 以 bytes 傳入（SDK 要求 bytes 或 str）
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # 先走 Blob API（_headers），失敗則退回底層 call_api
            _upload_rich_menu_image_any(api_client, blob_api, rich_menu_id, image_bytes, mime_type)
            print("✅ Image uploaded successfully!")
        except ApiException as e:
            print("❌ Failed to upload image:", e)
            exit(1)
    else:
        print("❌ Image file not found:", image_path)
        exit(1)

    # ✅ 設定為預設 Rich Menu
    print("📌 Setting as default rich menu...")
    messaging_api.set_default_rich_menu(rich_menu_id)
    print("🎉 Success! Default rich menu has been set.")