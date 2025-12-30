from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime # 改用 DateTime 比較單純
from sqlalchemy.sql import func
from backend.database import Base
from backend.models.user import User
from datetime import datetime
import pytz

def get_taiwan_time():
    tw = pytz.timezone("Asia/Taipei")
    return datetime.now(tw)

class Wishlist(Base):
    __tablename__ = "wishlist"
    no = Column(Integer, primary_key=True, autoincrement=True) 
    
    user_id = Column(Integer, ForeignKey(User.id))
    item_name = Column(String(255))
    price = Column(Integer)
    achieved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=get_taiwan_time)