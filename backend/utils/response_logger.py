import json
import os
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "logs" / "response_time.jsonl"


def log_response(user_id: int, user_input: str, pipeline: str, elapsed_seconds: float):
    """每次 pipeline 完整回覆後呼叫，append 一筆紀錄到 jsonl 檔。"""
    record = {
        "timestamp":       datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "user_id":         user_id,
        "user_input":      user_input,
        "pipeline":        pipeline,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[response_logger] 寫入失敗：{e}")
