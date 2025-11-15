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
}

# ---------------------------
# 解析 MySQL URL
# ---------------------------
def parse_mysql_url(url):
    pattern = r"mysql\+?pymysql?:\/\/(.*?):(.*?)@(.*?):(\d+)\/(.*)"
    match = re.match(pattern, url)
    if not match:
        print("❌ URL 格式錯誤 →", url)
        return None
    return match.groups()

# ---------------------------
# 連 Benefit DB
# ---------------------------
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
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )
    except Exception as e:
        print("❌ 無法連線 BENEFIT DB:", e)
        return None

# ---------------------------
# 主搜尋：FTS → fallback LIKE
# ---------------------------
def query_benefits(brand_name=None, category=None, candidates=None):
    conn = get_benefit_db_connection()
    if not conn:
        return []

    # ---- 先組 keyword 搜尋列表（依 AI parser 排序） ----
    search_keys = []

    if candidates:
        for c in candidates:
            bn = c.get("brand_name")
            if bn:
                search_keys.append(bn)

    if brand_name and brand_name not in search_keys:
        search_keys.insert(0, brand_name)

    # ---------------------------
    # FTS 搜尋
    # ---------------------------
    def fts_search(keyword):
        sql = """
            SELECT
                display_name,
                group_name,
                brands_text AS brands,
                reward_rate,
                'cube_benefits' AS source_table,
                MATCH(display_name, group_name, brands_text)
                    AGAINST (%s IN BOOLEAN MODE) AS score
            FROM cube_benefits
            WHERE MATCH(display_name, group_name, brands_text)
                    AGAINST (%s IN BOOLEAN MODE)
            ORDER BY score DESC
            LIMIT 50;
        """

        key = f"\"{keyword}*\""

        with conn.cursor() as cursor:
            cursor.execute(sql, (key, key))
            return cursor.fetchall()

    # ---------------------------
    # LIKE fallback
    # ---------------------------
    def like_search(keyword):
        sql = """
            SELECT
                display_name,
                group_name,
                brands_text AS brands,
                reward_rate,
                'cube_benefits' AS source_table,
                0 AS score
            FROM cube_benefits
            WHERE brands_text LIKE %s
            LIMIT 50;
        """

        pat = f"%{keyword}%"

        with conn.cursor() as cursor:
            cursor.execute(sql, (pat,))
            return cursor.fetchall()

    # ---------------------------
    # 主搜尋流程
    # ---------------------------
    final_results = []

    for kw in search_keys:
        # 1️⃣ 先試 FTS
        fts_rows = fts_search(kw)
        if fts_rows:
            final_results.extend(fts_rows)
            continue

        # 2️⃣ 若找不到 → LIKE fallback
        like_rows = like_search(kw)
        if like_rows:
            final_results.extend(like_rows)

    conn.close()

    # ---------------------------
    # 後處理：轉成統一格式
    # ---------------------------
    results = []
    for row in final_results:
        raw = row["brands"]
        try:
            brands_list = json.loads(raw) if raw.startswith("[") else [raw]
        except:
            brands_list = [raw]

        bank, card = BANK_CARD_MAP.get("cube_benefits", ("未知銀行", "未知卡片"))

        results.append({
            "display_name": row["display_name"],
            "group_name": row["group_name"],
            "brands": brands_list,
            "reward_rate": row["reward_rate"],
            "bank": bank,
            "card_name": card,
            "source_table": "cube_benefits",
            "score": row["score"],
        })

    return results

# ---------------------------
# parser fallback 查詢品牌
# ---------------------------
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
                    bl = json.loads(row["brands"])
                except:
                    continue

                for b in bl:
                    if keyword.lower() in b.lower():
                        return {
                            "brand_name": b,
                            "category": row["group_name"],
                            "intent": "查詢回饋",
                        }
        return None
    finally:
        conn.close()