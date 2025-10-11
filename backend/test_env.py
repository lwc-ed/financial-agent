from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")  # ✅ 明確指定路徑
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY")[:10], "...OK")
print("DATABASE_URL:", os.getenv("DATABASE_URL")[:40], "...OK")