from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DECIMAL, TIMESTAMP # 確保有 DECIMAL, TIMESTAMP
from sqlalchemy.sql import func
from backend.database import Base
from backend.models.user import User

class Wishlist(Base):
    __tablename__ = "wishlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey(User.id)) 
    item_name = Column(String(255))
    price = Column(Integer)
    achieved = Column(Boolean, default=False)
