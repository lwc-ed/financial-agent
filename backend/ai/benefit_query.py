from backend.database import SessionLocal
from backend.models.cube_benefits_model import CubeBenefit

def query_benefits(brand_name=None, category=None):
    """
    查詢信用卡回饋資料。
    若指定 brand_name，則在 brands JSON 內搜尋；
    若指定 category，則查 group_name。
    """
    db = SessionLocal()
    results = []

    try:
        if brand_name:
            rows = db.query(CubeBenefit).all()
            for r in rows:
                for b in r.brands:
                    if brand_name in b:
                        results.append({
                            "display_name": r.display_name,
                            "group_name": r.group_name,
                            "brand": b,
                            "reward_rate": r.reward_rate or "尚未爬取"
                        })
        elif category:
            rows = db.query(CubeBenefit).filter(CubeBenefit.group_name.like(f"%{category}%")).all()
            for r in rows:
                results.append({
                    "display_name": r.display_name,
                    "group_name": r.group_name,
                    "brand": "（通路查詢）",
                    "reward_rate": r.reward_rate or "尚未爬取"
                })
    finally:
        db.close()

    return results