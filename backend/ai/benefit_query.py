import os
import json
import pymysql

def get_db_connection():
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("⚠️ DATABASE_URL 環境變數未設定")
            return None
        # 假設 DATABASE_URL 格式為 mysql://user:password@host:port/dbname
        import re
        pattern = r"mysql\+?pymysql?:\/\/(.*?):(.*?)@(.*?):(\d+)\/(.*)"
        match = re.match(pattern, database_url)
        if not match:
            print("⚠️ DATABASE_URL 格式錯誤")
            return None
        user, password, host, port, db = match.groups()
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db,
            port=int(port),
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Exception as e:
        print(f"⚠️ 無法連接資料庫: {e}")
        return None

def query_benefits(brand_name: str = None, category: str = None):
    results = []
    conn = get_db_connection()
    if not conn:
        return results
    try:
        with conn.cursor() as cursor:
            sql = "SELECT display_name, group_name, brands FROM cube_benefits"
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                try:
                    brands_list = json.loads(row['brands'])
                except Exception as e:
                    print(f"⚠️ JSON parse error for brands in {row['display_name']}: {e}")
                    continue

                matched_brand = False
                matched_category = False

                if brand_name:
                    # 忽略大小寫比對brands陣列中是否包含brand_name關鍵字
                    for b in brands_list:
                        if brand_name.lower() in b.lower():
                            matched_brand = True
                            break
                else:
                    matched_brand = True

                if category:
                    if category.lower() in row['group_name'].lower():
                        matched_category = True
                else:
                    matched_category = True

                if matched_brand and matched_category:
                    results.append({
                        "display_name": row['display_name'],
                        "group_name": row['group_name'],
                        "brands": brands_list
                    })
        return results
    except Exception as e:
        print(f"⚠️ 查詢資料錯誤: {e}")
        return []
    finally:
        conn.close()

def find_brand_in_db(keyword):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cursor:
            sql = "SELECT display_name, group_name, brands FROM cube_benefits"
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                try:
                    brands_list = json.loads(row['brands'])
                except Exception as e:
                    print(f"⚠️ JSON parse error for brands in {row['display_name']}: {e}")
                    continue

                for b in brands_list:
                    if keyword.lower() in b.lower():
                        print(f"🔍 找到匹配品牌：{b} → {row['group_name']}")
                        return {
                            "brand_name": b,
                            "category": row['group_name'],
                            "intent": "查詢回饋"
                        }
        return None
    except Exception as e:
        print(f"⚠️ 查詢資料錯誤: {e}")
        return None
    finally:
        conn.close()
