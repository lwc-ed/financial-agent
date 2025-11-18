

from sqlalchemy import Column, Integer, String, JSON
from backend.database import Base

class CtbcLinePayBenefit(Base):
    __tablename__ = "ctbc_linepay_benefits"

    id = Column(Integer, primary_key=True, index=True)
    display_name = Column(String(100), nullable=False)     # 信用卡 or 簽帳金融卡
    group_name = Column(String(200), nullable=False)       # 回饋分類（如：國內外一般消費）
    brands = Column(JSON, nullable=False, default=list)    # 品牌（本頁通常為空陣列）
    reward_rate = Column(String(200), nullable=False)      # 回饋率文字，例如 "1%"、"2.8%"