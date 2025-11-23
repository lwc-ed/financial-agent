import json
import re
from pathlib import Path

import copy
import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

from backend.database import SessionBenefit
from backend.models.credit_card_benefit_model.dbs_eco_benefits_model import (
    DbsEcoBenefit,
)

URL = "https://www.dbs.com.tw/personal-zh/cards/dbs-credit-cards/eco-ez?pid=tw-pweb-personal-zh_cards_dbs-credit-cards_default_page-hyperlink"
HEADERS = {"User-Agent": "Mozilla/5.0"}
DISPLAY_NAME = "星展eco永續極簡卡"
OUTPUT_PATH = Path(__file__).with_name("dbs_eco_benefits.json")
DEBUG_OUTPUT_PATH = Path(__file__).with_name("dbs_eco_benefits_debug.json")
RAW_OUTPUT_PATH = Path(__file__).with_name("dbs_eco_raw_benefits.json")
LEVEL_PREFIX = {
    1: f"level1-{DISPLAY_NAME}",
    2: f"level2-{DISPLAY_NAME}",
    3: f"level3-{DISPLAY_NAME}",
}


def fetch_html() -> str:
    resp = requests.get(URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
    return resp.text


def clean_group_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw)
    name = re.sub(r"^[●\d\.\s]+", "", name)
    name = (
        name.replace("最高享回饋", "")
        .replace("最高回饋", "")
        .replace("最高享", "")
        .replace("最高", "")
    )
    return name.strip(" ：:")


def parse_benefit_blocks(soup: BeautifulSoup) -> list[dict]:
    records = []
    seen = set()

    for div in soup.find_all("div", class_="d-flex"):
        text = div.get_text("\n", strip=True)
        if "回饋" not in text or "%" not in text:
            continue

        lines = [ln for ln in text.split("\n") if ln.strip()]
        title = clean_group_name(lines[0] if lines else text)
        if len(title) > 30:
            continue  # 跳過描述型長句，避免把計算示意視為群組名稱
        rate_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", text)
        if not rate_match:
            continue

        reward_rate = f"{rate_match.group(1)}%"

        if "4.8%" in text:
            title = f"{title} (刷卡金加碼4.8%)"
        elif "1.8%" in text:
            title = f"{title} (刷卡金加碼1.8%)"

        key = (title, reward_rate)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "display_name": DISPLAY_NAME,
                "group_name": title or "未命名回饋",
                "brands": [],
                "reward_rate": reward_rate,
                "source": "block",
                "raw_text": text,
            }
        )
    return records


def extract_level_number(text: str, fallback: int) -> int:
    """從文字推斷 Level，若無則用 fallback 順序。"""
    circled = {"➊": 1, "➋": 2, "➌": 3, "➍": 4, "➎": 5}
    for sym, num in circled.items():
        if sym in text:
            return num
    m = re.search(r"Level\\s*([0-9])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return fallback


def parse_level_blocks(soup: BeautifulSoup) -> list[dict]:
    """
    依官網版面遇到「Level」區塊自動進位，擷取其內的回饋敘述。
    """
    records = []
    seen = set()
    level_counter = 0

    level_headers = soup.find_all("p", class_=lambda c: c and "subtitle" in c)
    for header in level_headers:
        header_text = header.get_text(" ", strip=True)
        if "Level" not in header_text:
            continue

        level_counter += 1
        level = extract_level_number(header_text, level_counter)
        display_name = LEVEL_PREFIX.get(level, f"level{level}-{DISPLAY_NAME}")

        box_topic = header.find_parent("div", class_=lambda c: c and "box-topic" in c)
        target_ul = None
        if box_topic:
            sib = box_topic.find_next_sibling()
            while sib and sib.name != "ul":
                sib = sib.find_next_sibling()
            target_ul = sib if sib and sib.name == "ul" else None
            if not target_ul:
                target_ul = box_topic.find("ul")
        if not target_ul:
            continue

        for tag in target_ul.find_all(["p", "h3"]):
            # 跳過 <small> 內容，避免抓到附註的小字
            tag_copy = copy.deepcopy(tag)
            for sm in tag_copy.find_all("small"):
                sm.decompose()

            p_text = re.sub(r"\s+", " ", tag_copy.get_text(" ", strip=True))
            if not p_text:
                continue
            if "回饋" not in p_text and "%" not in p_text and "NT$" not in p_text:
                continue

            reward_match = re.search(r"([0-9]+(?:\.[0-9]+)?%)", p_text)
            money_match = re.search(r"(NT\$[0-9,]+)", p_text)
            reward_rate = ""
            if reward_match:
                reward_rate = reward_match.group(1)
            elif money_match:
                reward_rate = money_match.group(1)

            group_name = p_text
            group_name = group_name.replace("●", "").replace("回饋", "")
            group_name = re.sub(r"\s+", " ", group_name).strip()

            key = (display_name, group_name, reward_rate)
            if key in seen:
                continue
            seen.add(key)

            records.append(
                {
                    "display_name": display_name,
                    "group_name": group_name or "未命名回饋",
                    "brands": [],
                    "reward_rate": reward_rate or "NA",
                    "source": "level-block",
                    "raw_text": p_text,
                    "level": level,
                }
            )

    return records


def parse_highlight_rates(soup: BeautifulSoup) -> list[dict]:
    text_content = soup.get_text(" ", strip=True)
    results = []

    pattern_map = [
        ("國內／國外一般消費", r"國內／?國外一般消費回饋?([0-9]+(?:\.[0-9]+)?%)"),
        ("國外指定地區一般消費", r"國外指定地區[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?%)"),
        ("eco永續消費", r"eco消費[^0-9]{0,10}([0-9]+(?:\.[0-9]+)?%)"),
    ]

    for label, pattern in pattern_map:
        match = re.search(pattern, text_content)
        if not match:
            continue

        reward_rate = match.group(1)
        if label == "eco永續消費" and "10%現金紅利回饋" in text_content:
            reward_rate = "10%"
        elif label == "國外指定地區一般消費" and "最高5％回饋" in text_content:
            reward_rate = "5%"

        results.append(
            {
                "display_name": DISPLAY_NAME,
                "group_name": label,
                "brands": [],
                "reward_rate": reward_rate,
                "source": "highlight",
                "raw_text": match.group(0),
            }
        )

    return results


def merge_records(*groups: list[dict]) -> list[dict]:
    """
    依照傳入順序決定優先權去重。
    優先權：level-block > block > highlight。
    去重 key 以 (group_name, reward_rate) 為準，避免同一方案跨來源重覆。
    """
    merged = []
    seen = {}

    for group in groups:
        for record in group:
            key = (record.get("group_name"), record.get("reward_rate"))

            if key in seen:
                # 若先前是 NA，但現在有實際回饋率則替換
                prev_idx = seen[key]
                if merged[prev_idx].get("reward_rate") == "NA" and record.get("reward_rate") != "NA":
                    merged[prev_idx] = record
                continue

            seen[key] = len(merged)
            merged.append(record)

    return merged


def assign_level(records: list[dict]) -> list[dict]:
    """依 reward_rate 判斷 level，並更新 display_name。"""
    def to_rate_float(rate: str) -> float:
        try:
            return float(rate.strip().replace("%", ""))
        except ValueError:
            return 0.0

    for record in records:
        # highlight 來源不套用 level，保持原卡名
        if record.get("source") == "highlight":
            record["display_name"] = DISPLAY_NAME
            record["level"] = None
            continue

        rate = to_rate_float(record["reward_rate"])
        if record.get("level"):
            level = int(record["level"])
        elif "線上購物" in record.get("group_name", ""):
            level = 1  # 官網規則：線上購物屬於 Level 1 基礎
        elif rate >= 6:
            level = 3
        elif rate >= 3:
            level = 2
        else:
            level = 1
        record["display_name"] = LEVEL_PREFIX.get(level, f"level{level}-{DISPLAY_NAME}")
        record["level"] = level
    return records


def write_json(records: list[dict]) -> None:
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 已輸出 {OUTPUT_PATH.name}")


def collect_raw_segments(soup: BeautifulSoup) -> list[dict]:
    """蒐集含回饋/% 關鍵字的原始文字片段（不做分類）。"""
    raw = []
    seen = set()
    for node in soup.find_all(string=True):
        text = node.strip()
        if not text:
            continue
        if "回饋" not in text and not re.search(r"[0-9]+(?:\.[0-9]+)?%", text):
            continue
        snippet = node.parent.get_text(" ", strip=True)
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) < 4 or len(snippet) > 600:
            continue
        if snippet in seen:
            continue
        seen.add(snippet)
        raw.append(
            {
                "tag": node.parent.name,
                "text": snippet,
            }
        )
    return raw


def write_raw_json(raw_segments: list[dict]) -> None:
    RAW_OUTPUT_PATH.write_text(
        json.dumps(raw_segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 已輸出 {RAW_OUTPUT_PATH.name}（原始回饋片段，不分類）")


def write_debug_json(records: list[dict], raw_segments: list[dict]) -> None:
    payload = {
        "classified_records": [
            {
                "display_name": rec.get("display_name"),
                "group_name": rec.get("group_name"),
                "reward_rate": rec.get("reward_rate"),
                "level": rec.get("level"),
                "source": rec.get("source"),
                "raw_text": rec.get("raw_text"),
            }
            for rec in records
        ],
        "raw_segments": raw_segments,
    }
    DEBUG_OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ 已輸出 {DEBUG_OUTPUT_PATH.name}（含分類結果與原始片段）")


def save_to_db(records: list[dict]) -> None:
    db = SessionBenefit()
    db.execute(text("TRUNCATE TABLE credit_card_benefits.dbs_eco_benefits;"))
    db.commit()

    objs = [
        DbsEcoBenefit(
            display_name=record["display_name"],
            group_name=record["group_name"],
            brands=record["group_name"],
            reward_rate=record["reward_rate"],
        )
        for record in records
    ]

    db.bulk_save_objects(objs)
    db.commit()
    db.close()
    print(f"✅ 已寫入 {len(objs)} 筆資料到 dbs_eco_benefits")


def main():
    html = fetch_html()
    soup = BeautifulSoup(html, "html.parser")

    # block_records = parse_benefit_blocks(soup)  # 不再使用 block 來源
    highlight_records = parse_highlight_rates(soup)
    level_records = parse_level_blocks(soup)
    raw_segments = collect_raw_segments(soup)

    # 只使用 Level 區塊與 highlight，忽略 block 來源
    records = merge_records(level_records, highlight_records)
    records = assign_level(records)
    if not records:
        raise SystemExit("❌ 未擷取到任何權益資料，請檢查目標頁面結構。")

    write_json(records)
    write_raw_json(raw_segments)
    write_debug_json(records, raw_segments)
    save_to_db(records)


if __name__ == "__main__":
    main()
