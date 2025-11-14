import os
import json
import pymysql
import re
from dotenv import load_dotenv

load_dotenv()

print("MAIN_DB_URL =", os.getenv("MAIN_DB_URL"))
print("BENEFIT_DB_URL =", os.getenv("BENEFIT_DB_URL"))

BANK_CARD_MAP = {
    "cube_benefits": ("國泰世華", "CUBE 卡"),
    # 之後如果有其他表，例如 "esun_ubear_benefits": ("玉山銀行", "Ubear 卡")
}

# ===========================
# 共用：解析 MySQL URL
# ===========================
def parse_mysql_url(url):
    pattern = r"mysql\+?pymysql?:\/\/(.*?):(.*?)@(.*?):(\d+)\/(.*)"
    match = re.match(pattern, url)
    if not match:
        print("❌ 解析失敗：URL 格式錯誤 →", url)
        return None
    return match.groups()

# ===========================
# 連線 MAIN_DB（目前可能暫時沒用到，先保留）
# ===========================
def get_main_db_connection():
    url = os.getenv("MAIN_DB_URL")
    if not url:
        print("❌ MAIN_DB_URL 未設定")
        return None

    parsed = parse_mysql_url(url)
    if not parsed:
        return None

    user, password, host, port, db = parsed
    try:
        return pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db,
            port=int(port),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        print("❌ 無法連線 MAIN DB:", e)
        return None

# ===========================
# 連線 credit_card_benefits DB
# ===========================
def get_benefit_db_connection():
    url = os.getenv("BENEFIT_DB_URL")
    if not url:
        print("❌ BENEFIT_DB_URL 未設定")
        return None

    parsed = parse_mysql_url(url)
    if not parsed:
        return None

    user, password, host, port, db = parsed
    try:
        return pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db,
            port=int(port),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        print("❌ 無法連線 BENEFIT DB:", e)
        return None

# ===========================
# 建 FTS 查詢字串
# ===========================
def build_fts_query(brand_name, category, candidates):
    keywords = []

    if brand_name:
        keywords.append(brand_name)

    if category:
        keywords.append(category)

    if candidates:
        for c in candidates:
            bn = c.get("brand_name")
            if bn:
                keywords.append(bn)

    # 轉成 FTS boolean 模式："關鍵詞*"
    return " ".join([f'"{k}*"' for k in keywords if k])

# ===========================
# 核心：查詢信用卡回饋（目前先只查 cube_benefits）
# ===========================
def query_benefits(brand_name=None, category=None, candidates=None):
    conn = get_benefit_db_connection()
    if not conn:
        return []

    fts_query = build_fts_query(brand_name, category, candidates)
    if not fts_query:
        return []

    # 目前只有 cube_benefits，一張卡一個 table 的第一版
    # 使用你在 cube_benefits 上建好的 FULLTEXT(display_name, group_name, brands_text, reward_rate)
    sql = """
        SELECT
            display_name,
            group_name,
            brands_text AS brands,
            reward_rate,
            'cube_benefits' AS source_table,
            MATCH(display_name, group_name, brands_text, reward_rate)
                AGAINST (%s IN BOOLEAN MODE) AS score
        FROM cube_benefits
        WHERE MATCH(display_name, group_name, brands_text, reward_rate)
                AGAINST (%s IN BOOLEAN MODE)
        ORDER BY score DESC
        LIMIT 50;
    """

    results = []
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (fts_query, fts_query))
            rows = cursor.fetchall()

            for row in rows:
                raw = row["brands"]
                # brands_text 裡通常長得像 ["國內餐飲(不含餐券)", "連鎖速食－麥當勞"]
                try:
                    if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
                        brands_list = json.loads(raw)
                    else:
                        brands_list = [b.strip() for b in re.split("[,、，]", raw)] if isinstance(raw, str) else []
                except Exception:
                    brands_list = []

                bank, card = BANK_CARD_MAP.get(row.get("source_table"), ("未知銀行", "未知卡片"))

                results.append({
                    "display_name": row["display_name"],
                    "group_name": row["group_name"],
                    "brands": brands_list,
                    "reward_rate": row["reward_rate"],
                    "bank": bank,
                    "card_name": card,
                    "source_table": row.get("source_table"),
                    "score": row["score"],
                })
        return results

    except Exception as e:
        print("❌ FTS 查詢錯誤:", e)
        return []

    finally:
        conn.close()

# ===========================
# 從 DB 反查品牌 → 做 parser fallback 用
# ===========================
def find_brand_in_db(keyword):
    conn = get_benefit_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cursor:
            sql = "SELECT display_name, group_name, brands FROM cube_benefits"
            cursor.execute(sql)
            rows = cursor.fetchall()

            for row in rows:
                try:
                    brands_list = json.loads(row["brands"])
                except Exception:
                    continue

                for b in brands_list:
                    if keyword.lower() in b.lower():
                        return {
                            "brand_name": b,
                            "category": row["group_name"],
                            "intent": "查詢回饋",
                        }
        return None
    finally:
        conn.close()