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
    "ctbc_linepay_benefits": ("中國信託", "LinePay 信用卡"),
    "ctbc_linepay_debit_benefits": ("中國信託", "LinePay 簽帳卡"),
    # ---【新增資料表時必改：在此加入新 table 對應的銀行/卡名】---
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
        key = f"\"{keyword}*\""
        # ---【新增資料表時必改：在此新增 FTS 查詢 SQL】---
        sql_list = [
            ("cube_benefits",
             """
             SELECT display_name, group_name, brands_text AS brands,
                    reward_rate, 'cube_benefits' AS source_table,
                    MATCH(display_name, group_name, brands_text)
                        AGAINST (%s IN BOOLEAN MODE) AS score
             FROM cube_benefits
             WHERE MATCH(display_name, group_name, brands_text)
                        AGAINST (%s IN BOOLEAN MODE)
             """),
            ("ctbc_linepay_benefits",
             """
             SELECT display_name, group_name, brands AS brands,
                    reward_rate, 'ctbc_linepay_benefits' AS source_table,
                    0 AS score
             FROM ctbc_linepay_benefits
             WHERE group_name LIKE %s
             """),
            ("ctbc_linepay_debit_benefits",
             """
             SELECT display_name, group_name, brands AS brands,
                    reward_rate, 'ctbc_linepay_debit_benefits' AS source_table,
                    0 AS score
             FROM ctbc_linepay_debit_benefits
             WHERE group_name LIKE %s
             """)
        ]
        results = []
        with conn.cursor() as cursor:
            for tbl, sql in sql_list:
                if tbl == "cube_benefits":
                    cursor.execute(sql, (key, key))
                else:
                    cursor.execute(sql, (f"%{keyword}%",))
                results.extend(cursor.fetchall())
        return results

    # ---------------------------
    # LIKE fallback
    # ---------------------------
    def like_search(keyword):
        pat = f"%{keyword}%"
        # ---【新增資料表時必改：在此新增 LIKE fallback 查詢 SQL】---
        sql_list = [
            ("cube_benefits",
             """
             SELECT display_name, group_name, brands_text AS brands,
                    reward_rate, 'cube_benefits' AS source_table,
                    0 AS score
             FROM cube_benefits
             WHERE brands_text LIKE %s
             """),
            ("ctbc_linepay_benefits",
             """
             SELECT display_name, group_name, brands AS brands,
                    reward_rate, 'ctbc_linepay_benefits' AS source_table,
                    0 AS score
             FROM ctbc_linepay_benefits
             WHERE group_name LIKE %s
             """),
            ("ctbc_linepay_debit_benefits",
             """
             SELECT display_name, group_name, brands AS brands,
                    reward_rate, 'ctbc_linepay_debit_benefits' AS source_table,
                    0 AS score
             FROM ctbc_linepay_debit_benefits
             WHERE group_name LIKE %s
             """)
        ]
        results = []
        with conn.cursor() as cursor:
            for tbl, sql in sql_list:
                cursor.execute(sql, (pat,))
                results.extend(cursor.fetchall())
        return results

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
            brands_list = json.loads(raw) if raw and isinstance(raw, str) and raw.startswith("[") else [raw]
        except:
            brands_list = [raw]

        # ---【新增資料表時必改：必須在 BANK_CARD_MAP 補上新表】---
        bank, card = BANK_CARD_MAP.get(row["source_table"], ("未知銀行", "未知卡片"))

        results.append({
            "display_name": row["display_name"],
            "group_name": row["group_name"],
            "brands": brands_list,
            "reward_rate": row["reward_rate"],
            "bank": bank,
            "card_name": card,
            "source_table": row["source_table"],
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
            # ---【如需支援新資料表搜尋，需在此加入新表 SELECT】---
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