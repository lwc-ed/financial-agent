from sqlalchemy import Column, Integer, String, Numeric, DateTime, UniqueConstraint
from backend.database import Base
from datetime import datetime
import pytz

tz = pytz.timezone("Asia/Taipei")


class SavingChallenge(Base):
    __tablename__ = "saving_challenges"

    no = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    item_name = Column(String(255), nullable=False)
    target_amount = Column(Numeric(12, 2), nullable=False)
    current_amount = Column(Numeric(12, 2), default=0)
    stage = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(tz))

    __table_args__ = (
        UniqueConstraint("user_id", "item_name", name="uq_user_item"),
    )