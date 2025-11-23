from sqlalchemy import Column, Integer, String, JSON
from backend.database import Base

class CubeBenefit(Base):
    __tablename__ = "cube_benefits"

    id = Column(Integer, primary_key=True, index=True)
    display_name = Column(String(100), nullable=False)  # 權益名稱
    group_name = Column(String(100), nullable=False)    # 群組名稱
    brands = Column(JSON)                               # 品牌清單（以 JSON 格式儲存）
    reward_rate = Column(String(20))  # 例如 "3%", "5.5%", "最高10%" 等文字格式
    #reward_rate = Column(JSON)  # 用 JSON 格式存多筆回饋資料