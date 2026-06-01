from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from backend.database import Base


class FinancialStatus(Base):
    """財務狀況卡片的快取。

    指標於記帳時（背景 thread）算好寫入，不接 LLM。
    白話文字（summary_text）於使用者開 dashboard 時才生成，
    並以 summary_hash 對應「生成當下的指標」；
    若 summary_hash == metrics_hash 代表指標沒變，可直接回快取、不重叫 LLM。
    """
    __tablename__ = "financial_status"

    user_id = Column(Integer, primary_key=True)
    metrics_json = Column(Text)              # 算好的指標包（JSON 字串）
    metrics_hash = Column(String(64))        # 指標的 hash，用來判斷是否需重生成
    level = Column(Integer)                  # 1~4；0 代表資料不足
    insufficient = Column(Integer, default=0)  # 1=資料不足，走引導文案
    summary_text = Column(Text, nullable=True)   # LLM 生成的白話內文（不含標題）
    summary_hash = Column(String(64), nullable=True)  # summary_text 對應的指標 hash
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
