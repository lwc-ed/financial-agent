import requests            # 發 HTTP 請求抓取遠端資料
import urllib3             # requests 內部用到的連線套件，這裡用來關閉警告
import json                # 讀寫 JSON
import re                  # 正規表達式，做字串清理

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 取消「未驗證 SSL 憑證」的警告（因為下面 verify=False）

HEADERS = {"User-Agent": "Mozilla/5.0"}  # 簡單偽裝瀏覽器 UA，避免被擋爬
MODEL_URL = "https://www.cathay-cube.com.tw/cathaybk/personal/product/credit-card/cards/cube-list.model.json"
# 目標 JSON 模型的網址（國泰 CUBE 卡權益頁面的結構資料）

def fetch_json(url: str) -> dict:
    resp = requests.get(url, headers=HEADERS, verify=False, timeout=30)
    # 發 GET 請求；verify=False 不驗證 SSL 憑證；timeout 避免卡住
    resp.raise_for_status()
    # 非 2xx 會丟錯，讓程式中止（利於除錯）
    return resp.json()
    # 直接把回應轉成 Python dict

def find_block(node, anchor):
    # 在任意型態的 node（可能是 dict 或 list）裡遞迴搜尋，找到 anchorKey == anchor 的那個區塊
    if isinstance(node, dict):
        if node.get("anchorKey") == anchor:
            return node
        # 沒命中就繼續往所有 value 遞迴
        for v in node.values():
            result = find_block(v, anchor)
            if result:
                return result
    elif isinstance(node, list):
        # 如果是 list，對每個元素遞迴
        for v in node:
            result = find_block(v, anchor)
            if result:
                return result
    return None
    # 找不到就回傳 None

def extract_block(data: dict, anchor):
    # 先找到指定 anchor 的整個區塊
    block = find_block(data, anchor)
    if block:
        # 依照該 JSON 結構，真正的內容在鍵 ':items' 下
        return block.get(":items", {})
    return {}
    # 找不到就給空 dict，方便後續判斷

def parse_category(cat: dict):
    """解析一個 category，回傳 (群組名稱, 品牌清單)"""
    name = cat.get("categoryName", "未命名類別")
    brands = []
    # 每個 category 底下通常會有多個 section（用 contentTrees 包起來）
    for section in cat.get("contentTrees", []):
        inner = section.get("contentTrees", {})
        # 真正的品牌清單通常在第二層的 contentTrees（多半是個 dict）
        for key, val in inner.items():
            if isinstance(val, dict) and "itemText" in val:
                brands.append(val["itemText"].strip())
                # 取出品牌名稱（itemText），順手去除前後空白
    return name, brands

def list_benefits(node):
    """遞迴掃描 JSON，找出所有回饋方案"""
    benefits = {}
    def dfs(n):
        # 深度優先搜尋整個 JSON
        if isinstance(n, dict):
            # 有些區塊的錨點鍵名可能是 anchorKey 或 anchorTarget（保險起見都抓）
            anchor = n.get("anchorKey") or n.get("anchorTarget")
            # 顯示給使用者看的標題可能在 title / mainTitle / text
            title = n.get("title") or n.get("mainTitle") or n.get("text")
            if anchor and title:
                # mainTitle 可能含 HTML 標籤，先用 regex 去掉 <...>
                clean_title = re.sub(r"<.*?>", "", title).strip()
                # 再把 &nbsp;、換行等雜訊清乾淨
                clean_title = re.sub(r"&nbsp;|\r|\n", "", clean_title).strip()
                # 如果字串裡有「適用期間」，通常後面是日期說明，僅保留前半段做為方案名
                if "適用期間" in clean_title:
                    clean_title = clean_title.split("適用期間")[0].strip()
                benefits[clean_title] = anchor
                # 以「顯示名稱」當 key、「anchor」當 value 建 map（之後要靠 anchor 找內容）
            # 持續往下遞迴
            for v in n.values():
                dfs(v)
        elif isinstance(n, list):
            for v in n:
                dfs(v)
    dfs(node)
    return benefits
    # 回傳 { 方案名稱: 對應 anchor } 的字典

def main():
    data = fetch_json(MODEL_URL)
    # 抓下整份 JSON（頁面模型）
    benefits_map = list_benefits(data)
    # 全文掃描，把所有「像是權益方案」的節點（含標題 + anchor）收集起來

    print("=== 可用的回饋方案 ===")
    for title, anchor in benefits_map.items():
        print(f"{title} -> {anchor}")
    print()
    # 先列出目前抓到的方案名稱與 anchor 對應，幫助確認抓到哪些

    all_outputs = {}
    exclude_keys = {"注意事項", "權益切換", "如何切換權益"} #黑名單
    # 你之前問「不小心爬到注意事項能不能略過」→ 這裡就是用黑名單跳過的機制

    for display_name, anchor in benefits_map.items():
        if display_name in exclude_keys:
            continue #跳過這些不處理

        items = extract_block(data, anchor)
        # 依 anchor 把該方案真正的內容（:items）抽出來
        output = {}
        print(f"=== {display_name} ===")

        if not items:
            # 找不到內容就提示一下（可能結構變了或這個 anchor 沒內容）
            print(f"找不到 {anchor} 權益，可能結構不同。")
            all_outputs[display_name] = {}
            continue

        # items 的每個 value 通常是一個「category 區塊」
        for cat in items.values():
            group, brands = parse_category(cat)
            # 取群組名稱與底下的品牌清單
            output[group] = brands

            print(f"【{group}】")
            for b in brands:
                print("  -", b)
            print()

        all_outputs[display_name] = output
    # === 寫入本地JSON檔 ===
    with open("cube_benefits_list.json", "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=2)
    print("✅ 已將結果寫入 cube_benefits_list.json")

    # === 寫入 MySQL 資料庫 ===
    from backend.database import SessionLocal
    from backend.models.cube_benefits_model import CubeBenefit
    from sqlalchemy import text

    db = SessionLocal() # 先建立資料庫 Session 連線

    # 先清空資料表
    db.execute(text("TRUNCATE TABLE credit_card_benefits.cube_benefits;"))
    db.commit()
    print("⚠️ 已清空舊資料（開始重新寫入）")

    records = []

    for display_name, group_data in all_outputs.items():
        for group_name, brands in group_data.items():
            record = CubeBenefit(
                display_name=display_name,
                group_name=group_name,
                brands=brands,
                reward_rate=None  # 目前還沒爬回饋％數，先留空
            )
            records.append(record)

    db.bulk_save_objects(records)
    db.commit()
    db.close()
    print(f"✅ 已成功寫入 {len(records)} 筆資料到 MySQL！")


if __name__ == "__main__":
    main()
# 直接執行此檔案時，跑 main()