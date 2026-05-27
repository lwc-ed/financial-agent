"""
一次性 migration：把 record table 裡的舊中文 type 值更新成英文
  支出 → expense
  收入 → income
執行方式：python3 -m backend.migrate_record_types
"""
from backend.database import SessionLocal, engine
from sqlalchemy import text

def migrate():
    db = SessionLocal()
    try:
        result_expense = db.execute(
            text("UPDATE records SET type = 'expense' WHERE type = '支出'")
        )
        result_income = db.execute(
            text("UPDATE records SET type = 'income' WHERE type = '收入'")
        )
        db.commit()
        print(f"✅ 支出 → expense：{result_expense.rowcount} 筆")
        print(f"✅ 收入 → income：{result_income.rowcount} 筆")
    except Exception as e:
        db.rollback()
        print(f"❌ 失敗：{e}")
    finally:
        db.close()

if __name__ == "__main__":
    migrate()
