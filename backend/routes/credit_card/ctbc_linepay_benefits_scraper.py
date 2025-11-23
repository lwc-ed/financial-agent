import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from backend.database import SessionBenefit
from backend.models.credit_card_benefit_model.ctbc_linepay_benefits_model import CtbcLinePayBenefit
from backend.models.credit_card_benefit_model.ctbc_linepay_debit_benefits_model import CtbcLinePayDebitBenefit

NOTICE_URL = "https://www.ctbcbank.com/content/dam/minisite/long/creditcard/LINEPay/notice.html"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_PATH = Path(__file__).with_name("ctbc_linepay_benefits.json")


def fetch_notice_html() -> str:
    resp = requests.get(NOTICE_URL, headers=HEADERS, timeout=30, verify=False)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
    return resp.text


def normalize_text(node) -> str:
    if node is None:
        return ""
    if hasattr(node, "get_text"):
        node = node.get_text(separator=" ", strip=True)
    if not isinstance(node, str):
        node = str(node)
    return re.sub(r"\s+", " ", node).strip()


def extract_order_items(order_list) -> list:
    items = []
    if not order_list:
        return items
    for li in order_list.find_all("li", recursive=False):
        text = normalize_text(li)
        if text:
            items.append(text)
    return items


def parse_rate_table(table) -> list:
    if not table:
        return []

    header_rows = table.find("thead")
    credit_label = "LINE Pay信用卡"
    debit_label = "LINE Pay簽帳金融卡"

    if header_rows:
        rows = header_rows.find_all("tr")
        if len(rows) > 1:
            labels = [
                normalize_text(cell)
                for cell in rows[1].find_all(["td", "th"])
                if normalize_text(cell)
            ]
            if len(labels) >= 1:
                credit_label = labels[0]
            if len(labels) >= 2:
                debit_label = labels[1]

    rate_rows = []
    tbody = table.find("tbody")
    if not tbody:
        return rate_rows

    for tr in tbody.find_all("tr"):
        cells = [normalize_text(td) for td in tr.find_all("td")]
        if len(cells) < 4:
            continue
        rate_rows.append(
            {
                "category": cells[0],
                "definition": cells[1],
                "rates": {
                    credit_label: cells[2],
                    debit_label: cells[3],
                },
            }
        )
    return rate_rows


def parse_marketing_bonus(section) -> dict:
    if not section:
        return {}
    order_list = section.find("ol", class_="order-list")
    table = section.find("table", class_="notice-content-table")
    return {
        "title": "行銷加碼回饋",
        "items": extract_order_items(order_list),
        "rate_table": parse_rate_table(table),
    }


def parse_notice_page(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    raise_section = soup.select_one("#raise")
    return {
        "source": NOTICE_URL,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "marketing_bonus": parse_marketing_bonus(raise_section)
    }


# ------------------------------------------------------------
# ⭐ 改良版：轉換 rate_table 並拆成信用卡/金融卡兩種
# ------------------------------------------------------------
def convert_rate_table(rate_table):
    results = []
    for row in rate_table:
        category = row["category"]
        rates = row["rates"]

        for card_type, rate in rates.items():
            # 判斷信用卡 or 簽帳金融卡
            if "信用卡" in card_type:
                table_type = "credit"
                display_name = "LINE Pay信用卡"
            else:
                table_type = "debit"
                display_name = "LINE Pay簽帳金融卡"

            # 只取開頭的「2.8%」
            clean_rate = re.match(r"^[0-9.]+%", rate)
            reward_rate = clean_rate.group(0) if clean_rate else rate

            results.append({
                "table_type": table_type,
                "display_name": display_name,
                "group_name": category,
                "brands": "[]",
                "reward_rate": reward_rate,
            })
    return results


# ------------------------------------------------------------
# ⭐ 依照 credit / debit 分開寫入兩張 table
# ------------------------------------------------------------
def save_to_db(rate_table):
    db = SessionBenefit()
    from sqlalchemy import text


    # --- 完全清空兩個 Table（含 AUTO_INCREMENT）---
    db.execute(text("TRUNCATE TABLE credit_card_benefits.ctbc_linepay_benefits;"))
    db.execute(text("TRUNCATE TABLE credit_card_benefits.ctbc_linepay_debit_benefits;"))
    db.commit()
    
    rows = convert_rate_table(rate_table)

    # 清空舊資料（兩張表分別清除）
    db.query(CtbcLinePayBenefit).filter(CtbcLinePayBenefit.display_name.like("LINE Pay信用卡%")).delete()
    db.query(CtbcLinePayDebitBenefit).filter(CtbcLinePayDebitBenefit.display_name.like("LINE Pay簽帳金融卡%")).delete()

    for row in rows:
        if row["table_type"] == "credit":
            record = CtbcLinePayBenefit(
                display_name=row["display_name"],
                group_name=row["group_name"],
                brands=row["group_name"],
                reward_rate=row["reward_rate"],
            )
        else:  # debit
            record = CtbcLinePayDebitBenefit(
                display_name=row["display_name"],
                group_name=row["group_name"],
                brands=row["group_name"],
                reward_rate=row["reward_rate"],
            )

        db.add(record)

    db.commit()
    db.close()
    print("✔ 已寫入信用卡與簽帳金融卡兩張資料表")


def write_output(data: dict) -> None:
    OUTPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 已寫入 {OUTPUT_PATH}")


def main():
    html = fetch_notice_html()
    parsed = parse_notice_page(html)
    rate_table = parsed["marketing_bonus"]["rate_table"]
    save_to_db(rate_table)
    write_output(parsed)


if __name__ == "__main__":
    main()