"""
notification_service.py
=======================
控制財務風險通知的發送邏輯：冷卻時間 + 30 天滑動視窗上限。

修改通知規則只需調整 NOTIFICATION_CONFIG。
"""

from datetime import datetime, timedelta

# 開發人員白名單：豁免通知冷卻與次數上限，方便測試
NOTIFICATION_WHITELIST_USER_IDS = {27, 29, 30}

# ── 通知規則（修改這裡即可調整上限）─────────────────────────────────
# cooldown_days : 同等級多少天內不重複通知
# max_per_30d   : 30 天滑動視窗內同等級最多幾次
NOTIFICATION_CONFIG: dict[int, dict] = {
    2: {"cooldown_days": 3, "max_per_30d": 4},
    3: {"cooldown_days": 2, "max_per_30d": 6},
    4: {"cooldown_days": 1, "max_per_30d": 10},
}

_LEVEL_LABELS = {
    1: "安全",
    2: "注意",
    3: "警戒",
    4: "危險",
}

_UPGRADE_MESSAGES = {
    2: "您的消費風險已上升至注意等級，建議留意近期支出。",
    3: "⚠️ 您的消費風險已上升至警戒等級，請減少非必要支出。",
    4: "🔴 您的消費風險已上升至危險等級，強烈建議立即控制消費！",
}

_DOWNGRADE_MESSAGES = {
    1: "🎉 您的財務風險已回到安全等級，繼續保持良好習慣！",
    2: "✅ 您的財務風險已降至注意等級，財務狀況有所改善，繼續加油！",
    3: "📉 您的財務風險已降至警戒等級，情況好轉，請持續控制支出。",
}

_SAME_MESSAGES = {
    2: "提醒您留意近期消費，下週預估花費偏高。",
    3: "⚠️ 財務仍處於警戒狀態，請注意控制支出。",
    4: "🔴 財務仍處於危險狀態，請盡快減少非必要消費。",
}


def check_and_notify(user_id: int, result: dict, prediction_row, db) -> bool:
    """
    判斷是否發送通知，若發送則寫入 risk_notifications 並更新 prediction_row。
    回傳 (should_send, message, direction) 或 None。
    """
    from backend.features.risk.risk_notification import RiskNotification

    new_level = result["risk_level"]

    if new_level == 1:
        return None

    last_level = prediction_row.last_notified_level
    last_at = prediction_row.last_notified_at
    now = datetime.now()

    # 判斷升降級
    if last_level is None:
        direction = "same"
    elif new_level > last_level:
        direction = "upgrade"
    elif new_level < last_level:
        direction = "downgrade"
    else:
        direction = "same"

    # 等級上升或白名單使用者：無視所有限制，強制通知
    is_whitelisted = user_id in NOTIFICATION_WHITELIST_USER_IDS
    if direction != "upgrade" and not is_whitelisted:
        config = NOTIFICATION_CONFIG[new_level]

        # 冷卻檢查
        if last_at is not None:
            cooldown = timedelta(days=config["cooldown_days"])
            if now - last_at < cooldown:
                return None

        # 30 天滑動視窗檢查
        window_start = now - timedelta(days=30)
        count = (
            db.query(RiskNotification)
            .filter(
                RiskNotification.user_id == user_id,
                RiskNotification.risk_level == new_level,
                RiskNotification.notified_at >= window_start,
            )
            .count()
        )
        if count >= config["max_per_30d"]:
            return None

    # 組通知訊息
    predicted = result["predicted_expense_7d"]
    if direction == "upgrade":
        msg = (
            f"📊 財務風險預測\n"
            f"預計下週花費：${predicted:,.0f}\n"
            f"{_UPGRADE_MESSAGES[new_level]}"
        )
    elif direction == "downgrade":
        msg = (
            f"📊 財務風險預測\n"
            f"預計下週花費：${predicted:,.0f}\n"
            f"{_DOWNGRADE_MESSAGES[new_level]}"
        )
    else:
        msg = (
            f"📊 財務風險預測\n"
            f"預計下週花費：${predicted:,.0f}\n"
            f"{_SAME_MESSAGES[new_level]}"
        )

    # 寫入通知歷史
    db.add(RiskNotification(
        user_id=user_id,
        risk_level=new_level,
        direction=direction,
        notified_at=now,
    ))

    # 更新 prediction_row 的通知狀態
    prediction_row.last_notified_at = now
    prediction_row.last_notified_level = new_level

    return msg
