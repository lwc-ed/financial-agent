from sqlalchemy import Column, Integer, String, Date, DateTime, UniqueConstraint
from backend.core.database import Base
from datetime import datetime


class UserTokenLog(Base):
    __tablename__ = "user_token_logs"
    __table_args__ = (
        UniqueConstraint("user_id", "date", "source", name="uq_user_date_source"),
    )

    id                        = Column(Integer, primary_key=True, autoincrement=True)
    user_id                   = Column(Integer, nullable=False, index=True)
    date                      = Column(Date, nullable=False, index=True)
    source                    = Column(String(50), nullable=False)  # pipeline 名稱：credit_card / daily_news / expense / ...
    model_openai              = Column(String(50), nullable=True)
    model_perplexity          = Column(String(50), nullable=True)
    openai_prompt_tokens      = Column(Integer, default=0)
    openai_completion_tokens  = Column(Integer, default=0)
    openai_total_tokens       = Column(Integer, default=0)
    perplexity_prompt_tokens      = Column(Integer, default=0)
    perplexity_completion_tokens  = Column(Integer, default=0)
    perplexity_total_tokens       = Column(Integer, default=0)
    updated_at                = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
