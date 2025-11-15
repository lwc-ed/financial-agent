def build_summary(parsed, results):
    """
    將查詢結果格式化成清楚可讀的回饋摘要。
    僅顯示：
    - display_name
    - reward_rate
    - bank / card
    （不顯示 group_name, category, brands）
    """

    # ========== 無查詢結果 ==========
    if not results:
        keyword = parsed.get("brand_name") or parsed.get("category") or "此通路"
        return f"找不到「{keyword}」的信用卡回饋資訊。"

    # ========== 排序：依 reward_rate（降序）==========
    def extract_rate(r):
        raw = r.get("reward_rate")
        if not raw:
            return -1
        cleaned = str(raw).replace("%", "").strip()
        try:
            return float(cleaned)
        except:
            return -1

    results = sorted(results, key=extract_rate, reverse=True)

    # ========== 新樣式：只顯示使用者要看的項目 ==========
    summary_lines = ["回饋分析："]

    for r in results:
        bank = r.get("bank", "未知銀行")
        card = r.get("card_name", "未知卡片")
        display = r.get("display_name") or "（無項目名稱）"
        reward = r.get("reward_rate") or "（尚未提供）"

        # 顯示格式：遠東百貨 — CUBE 卡：6%
        summary_lines.append(f"- {display} — {bank} {card}：{reward}")

    return "\n".join(summary_lines)