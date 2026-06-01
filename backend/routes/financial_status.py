"""財務狀況讀取 API。

開 dashboard 時呼叫：讀 financial_status 快取 → 指標沒變回上次文字、
變了才叫 LLM 翻成白話。資料不足 / token 超限 / LLM 失敗都有 fallback，
保證這個欄位永遠有東西、不會空白或一直轉圈。
"""
import os
import json

from flask import Blueprint, request, jsonify
from openai import OpenAI

from backend.database import SessionLocal
from backend.models.user import User
from backend.models.financial_status import FinancialStatus
from backend.ml_inference.financial_status_service import LEVEL_DISPLAY, ensure_metrics
from backend.utils.token_tracker import upsert_pipeline_tokens, is_over_daily_limit

financial_status_bp = Blueprint("financial_status", __name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INSUFFICIENT_TEXT = "再記幾天我就能幫你看囉"

_SYSTEM_PROMPT = """你是親切的理財小幫手，幫使用者把財務數字翻成白話。
規則：
- 只能根據我給的指標說話，絕對不要自己編造或計算新數字。
- 輸出固定格式：兩條「・」開頭的重點，加一行「👉」開頭的建議。
- 建議用柔性、有親和力的口吻（例如「建議…會比較好喔！」「可以試試看…」），不要命令式。
- 全部用繁體中文、口語、簡短。整體控制在 60 字內。
- 不要加標題、不要加開場白、不要 emoji（開頭的「・」和「👉」除外）。
指標說明：income_30d / expense_30d 是「最近 30 天」的收入 / 支出；
trend_pct 是最近 7 天支出相對前一週的增減百分比（為 null 時請改用 trend_label 描述方向，不要編造百分比）。
範例：
・餘額 $12,400，最近花費比收入多了一些
・最近 30 天外食花最多（佔 38%）
👉 建議這週外食控制在 $1,500 內會比較好喔！"""


def _fallback_body(metrics: dict) -> str:
    """LLM 不可用時的樣板句，純填指標、不需 LLM。"""
    lines = [f"・目前餘額 ${metrics.get('balance', 0):,}"]
    top = metrics.get("top_category")
    if top:
        lines.append(f"・最近 30 天「{top['name']}」花最多（佔 {top['pct']}%）")
    else:
        lines.append(f"・最近 30 天支出 ${metrics.get('expense_30d', 0):,}")
    trend = metrics.get("trend_pct")
    if trend is not None and trend > 0:
        lines.append(f"👉 最近 7 天支出比前一週多 {trend}%，建議留意一下會比較好喔！")
    elif metrics.get("trend_label") == "明顯增加":
        lines.append("👉 最近支出明顯增加，建議稍微留意一下會比較好喔！")
    else:
        lines.append("👉 支出控制得不錯，繼續保持喔！")
    return "\n".join(lines)


def _generate_body(metrics: dict, user_id: int) -> str | None:
    """呼叫 LLM 生成內文；失敗回 None 由 caller fallback。"""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(metrics, ensure_ascii=False)},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        body = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        if usage:
            upsert_pipeline_tokens(
                user_id=user_id,
                source="financial_status",
                model_openai="gpt-4o-mini",
                openai_prompt=usage.prompt_tokens,
                openai_completion=usage.completion_tokens,
            )
        return body or None
    except Exception as e:
        print(f"[financial_status] LLM 生成失敗：{repr(e)}")
        return None


@financial_status_bp.get("/")
def get_financial_status():
    """GET /api/financial-status?line_user_id=Uxxxx"""
    line_user_id = (request.args.get("line_user_id") or "").strip()
    if not line_user_id:
        return jsonify({"status": "error", "message": "line_user_id 必填"}), 400

    with SessionLocal() as db:
        user = db.query(User).filter(User.line_user_id == line_user_id).first()
        if user is None:
            return jsonify({"status": "insufficient", "body": INSUFFICIENT_TEXT})

        # 快取不存在就用既有 risk_predictions 即時補算（不重跑推論）
        row = ensure_metrics(db, user.id, line_user_id)
        db.commit()

        # 資料不足 → 引導文案，不叫 LLM
        if row is None or row.insufficient:
            return jsonify({"status": "insufficient", "body": INSUFFICIENT_TEXT})

        metrics = json.loads(row.metrics_json) if row.metrics_json else {}
        display = LEVEL_DISPLAY.get(row.level, LEVEL_DISPLAY[1])

        # 指標沒變 → 直接回快取文字（0 token）
        if row.summary_text and row.summary_hash == row.metrics_hash:
            body = row.summary_text
        else:
            # 指標變了：token 沒超限才叫 LLM，否則用樣板句
            body = None
            if not is_over_daily_limit(user.id):
                body = _generate_body(metrics, user.id)
            if body is None:
                # token 超限或 LLM 失敗 → 留用舊文字，沒有舊文字才用樣板
                body = row.summary_text or _fallback_body(metrics)
            else:
                row.summary_text = body
                row.summary_hash = row.metrics_hash
                db.commit()

        return jsonify({
            "status": "ok",
            "level": row.level,
            "emoji": display["emoji"],
            "title": display["title"],
            "color": display["color"],
            "body": body,
        })
