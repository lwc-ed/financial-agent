# backend/models/wishlist.py
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from database import Base

class Wishlist(Base):
    __tablename__ = "wishlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    item_name = Column(String(255))
    price = Column(Integer)
    achieved = Column(Boolean, default=False)
