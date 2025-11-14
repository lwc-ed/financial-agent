def build_summary(parsed, results):
    """
    將查詢結果格式化成清楚可讀的回饋摘要。

    parsed 結構：
    {
        "brand_name": "...",
        "category": "...",
        "candidates": [...]
    }

    results 結構（來自 benefit_query）：
    {
        "display_name": "...",
        "group_name": "...",
        "brands": [...],
        "reward_rate": "...",
        "bank": "...",
        "card_name": "...",
        "source_table": "...",
        "score": ...
    }
    """

    # ========== 無查詢結果 ==========
    if not results:
        keyword = parsed.get("brand_name") or parsed.get("category") or "此通路"
        return f"找不到「{keyword}」的信用卡回饋資訊。"

    # ========== 排序：依 reward_rate ==========
    def extract_reward_value(r):
        raw = r.get("reward_rate")
        if not raw:
            return -1
        cleaned = str(raw).replace("%", "").strip()
        try:
            return float(cleaned)
        except:
            return -1

    results = sorted(results, key=extract_reward_value, reverse=True)

    # ========== 製作 summary ==========
    summary_lines = ["回饋分析："]

    for r in results:
        bank = r.get("bank", "未知銀行")
        card = r.get("card_name", "未知卡片")
        reward = r.get("reward_rate") or "（尚未提供）"
        display = r.get("display_name") or "（無項目名稱）"

        summary_lines.append(
            f"- {bank} {card}：{display}（{reward}）"
        )

    return "\n".join(summary_lines)