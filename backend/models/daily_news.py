from sqlalchemy import Column, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from backend.database import Base
from backend.models.user import User


class DailyNews(Base):
    __tablename__ = "daily_news"

    no = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey(User.id), nullable=False, index=True)

    perplexity_scraper = Column(JSON, nullable=False)
    gpt_response = Column(JSON, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
