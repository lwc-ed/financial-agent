import os
from datetime import date, datetime
from sqlalchemy import func, text
from backend.database import SessionLocal

DAILY_TOKEN_LIMIT = int(os.getenv("DAILY_TOKEN_LIMIT", "50000"))


def upsert_pipeline_tokens(
    line_user_id: str,
    source: str,
    model_openai: str | None = None,
    openai_prompt: int = 0,
    openai_completion: int = 0,
    model_perplexity: str | None = None,
    perplexity_prompt: int = 0,
    perplexity_completion: int = 0,
):
    """
    Pipeline 完成後呼叫一次。
    同一 user + date + source 已存在就累加，否則新增。
    """
    if not line_user_id:
        return
    today = date.today()
    oai_total = openai_prompt + openai_completion
    ppl_total = perplexity_prompt + perplexity_completion
    try:
        db = SessionLocal()
        db.execute(
            text("""
                INSERT INTO user_token_logs (
                    line_user_id, date, source,
                    model_openai, model_perplexity,
                    openai_prompt_tokens, openai_completion_tokens, openai_total_tokens,
                    perplexity_prompt_tokens, perplexity_completion_tokens, perplexity_total_tokens,
                    updated_at
                ) VALUES (
                    :uid, :dt, :src,
                    :m_oai, :m_ppl,
                    :oai_p, :oai_c, :oai_t,
                    :ppl_p, :ppl_c, :ppl_t,
                    :now
                )
                ON DUPLICATE KEY UPDATE
                    model_openai                 = COALESCE(model_openai, VALUES(model_openai)),
                    model_perplexity             = COALESCE(model_perplexity, VALUES(model_perplexity)),
                    openai_prompt_tokens         = openai_prompt_tokens     + VALUES(openai_prompt_tokens),
                    openai_completion_tokens     = openai_completion_tokens + VALUES(openai_completion_tokens),
                    openai_total_tokens          = openai_total_tokens      + VALUES(openai_total_tokens),
                    perplexity_prompt_tokens     = perplexity_prompt_tokens     + VALUES(perplexity_prompt_tokens),
                    perplexity_completion_tokens = perplexity_completion_tokens + VALUES(perplexity_completion_tokens),
                    perplexity_total_tokens      = perplexity_total_tokens      + VALUES(perplexity_total_tokens),
                    updated_at                   = VALUES(updated_at)
            """),
            {
                "uid":   line_user_id,
                "dt":    today,
                "src":   source,
                "m_oai": model_openai,
                "m_ppl": model_perplexity,
                "oai_p": openai_prompt,
                "oai_c": openai_completion,
                "oai_t": oai_total,
                "ppl_p": perplexity_prompt,
                "ppl_c": perplexity_completion,
                "ppl_t": ppl_total,
                "now":   datetime.utcnow(),
            },
        )
        db.commit()
        print(f"[token_tracker] {source} | {line_user_id} "
              f"| openai={oai_total} | perplexity={ppl_total}")
    except Exception as e:
        import traceback
        print(f"[token_tracker] upsert error: {e}")
        traceback.print_exc()
    finally:
        try:
            db.close()
        except Exception:
            pass


def get_daily_total(line_user_id: str) -> int:
    """今日 openai + perplexity 合計 token。"""
    from backend.models.token_log import UserTokenLog
    try:
        db = SessionLocal()
        row = db.query(
            func.sum(UserTokenLog.openai_total_tokens)
        ).filter(
            UserTokenLog.line_user_id == line_user_id,
            UserTokenLog.date == date.today(),
        ).scalar()
        return row or 0
    except Exception as e:
        print(f"[token_tracker] get_daily_total error: {e}")
        return 0
    finally:
        try:
            db.close()
        except Exception:
            pass


def is_over_daily_limit(line_user_id: str) -> bool:
    used = get_daily_total(line_user_id)
    if used >= DAILY_TOKEN_LIMIT:
        print(f"[token_tracker] {line_user_id} 已達日限 {used}/{DAILY_TOKEN_LIMIT}")
        return True
    return False
