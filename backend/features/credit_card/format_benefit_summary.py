def build_summary(parsed, results):
    """
    整合所有搜尋結果後，挑出每張卡的最高回饋，並輸出你指定的格式。
    """

    if not results:
        keyword = parsed.get("brand_name") or parsed.get("category") or "此通路"
        return f"📄 Summary：\n找不到「{keyword}」的信用卡回饋資訊。"

    # ---- 解析回饋率（把 "6%" → 6.0）----
    def extract_rate(raw):
        if not raw:
            return -1
        cleaned = str(raw).replace("%", "").strip()
        try:
            return float(cleaned)
        except:
            return -1

    # ---- 每張卡只留下最高回饋 ----
    best_for_card = {}

    for r in results:
        card_key = (r["bank"], r["card_name"])  # 卡片唯一識別

        rate_value = extract_rate(r["reward_rate"])

        if card_key not in best_for_card:
            best_for_card[card_key] = r
        else:
            old_rate = extract_rate(best_for_card[card_key]["reward_rate"])
            if rate_value > old_rate:
                best_for_card[card_key] = r

    # ---- 轉成列表並依回饋率排序 ----
    final_list = list(best_for_card.values())
    final_list.sort(key=lambda r: extract_rate(r["reward_rate"]), reverse=True)

    # ---- 產生 Summary ----
    lines = ["回饋分析："]

    for r in final_list:
        display = r.get("display_name", "（無名稱）")
        bank = r.get("bank", "未知銀行")
        card = r.get("card_name", "未知卡片")
        rate = r.get("reward_rate", "（尚未提供）")

        lines.append(f"- {display} — {bank} {card}：{rate}")

    return "\n".join(lines)