"""
每月 token 用量彙總。
手動執行：python3 -m backend.utils.monthly_stats
排程建議：每月 1 號 00:05 跑上個月的彙總
"""
import sys
from datetime import datetime, date
from sqlalchemy import text
from backend.database import SessionLocal


def aggregate_monthly_stats(year_month: str | None = None):
    """
    彙總指定月份（格式 '2026-05'）的 token 用量到 monthly_token_stats。
    未指定則彙總當月。
    """
    if year_month is None:
        year_month = date.today().strftime("%Y-%m")

    print(f"[monthly_stats] 開始彙總 {year_month} ...")
    db = SessionLocal()
    try:
        db.execute(
            text("""
                INSERT INTO monthly_token_stats (
                    year_month, source, request_count, unique_users,
                    openai_prompt_tokens, openai_completion_tokens, openai_total_tokens,
                    perplexity_prompt_tokens, perplexity_completion_tokens, perplexity_total_tokens,
                    updated_at
                )
                SELECT
                    DATE_FORMAT(date, '%Y-%m')        AS `year_month`,
                    source,
                    COUNT(*)                          AS request_count,
                    COUNT(DISTINCT user_id)           AS unique_users,
                    SUM(openai_prompt_tokens)         AS openai_prompt_tokens,
                    SUM(openai_completion_tokens)     AS openai_completion_tokens,
                    SUM(openai_total_tokens)          AS openai_total_tokens,
                    SUM(perplexity_prompt_tokens)     AS perplexity_prompt_tokens,
                    SUM(perplexity_completion_tokens) AS perplexity_completion_tokens,
                    SUM(perplexity_total_tokens)      AS perplexity_total_tokens,
                    NOW()
                FROM user_token_logs
                WHERE DATE_FORMAT(date, '%Y-%m') = :ym
                GROUP BY `year_month`, source
                ON DUPLICATE KEY UPDATE
                    request_count                 = VALUES(request_count),
                    unique_users                  = VALUES(unique_users),
                    openai_prompt_tokens          = VALUES(openai_prompt_tokens),
                    openai_completion_tokens      = VALUES(openai_completion_tokens),
                    openai_total_tokens           = VALUES(openai_total_tokens),
                    perplexity_prompt_tokens      = VALUES(perplexity_prompt_tokens),
                    perplexity_completion_tokens  = VALUES(perplexity_completion_tokens),
                    perplexity_total_tokens       = VALUES(perplexity_total_tokens),
                    updated_at                    = VALUES(updated_at)
            """),
            {"ym": year_month},
        )
        db.commit()
        print(f"[monthly_stats] {year_month} 彙總完成")
    except Exception as e:
        db.rollback()
        print(f"[monthly_stats] 錯誤：{e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # 支援傳入指定月份，例如：python3 -m backend.utils.monthly_stats 2026-04
    ym = sys.argv[1] if len(sys.argv) > 1 else None
    aggregate_monthly_stats(ym)
