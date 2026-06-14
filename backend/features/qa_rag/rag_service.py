import os
import re
from typing import List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from backend.core.embedder import encode_query, encode_documents

load_dotenv()

_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "knowledge", "chroma_db")

_vector_store = None
_openai_client = None


class _SharedEmbeddings(Embeddings):
    """langchain Embeddings wrapper，共用 embedder.py 的 singleton，避免重複載入模型。"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return encode_documents(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return encode_query(text).tolist()


def _init():
    global _vector_store, _openai_client
    if _vector_store is not None:
        return
    print("[rag_service] 初始化向量資料庫（共用 embedder singleton）...")
    _vector_store = Chroma(persist_directory=_CHROMA_DIR, embedding_function=_SharedEmbeddings())
    _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("[rag_service] 初始化完成")


def answer_financial_question(query: str) -> str:
    _init()
    docs = _vector_store.similarity_search(query, k=3)
    print(f"[rag_service] query={query!r}, docs_found={len(docs)}")
    for i, d in enumerate(docs):
        print(f"[rag_service] doc[{i}]: {d.page_content[:80]!r}")
    context = "\n\n".join(d.page_content for d in docs)
    prompt = (
        f"請根據以下資訊回答問題：\n{context}\n\n"
        "注意：若資訊不足以回答，請直接回答「抱歉，文件中未提及此資訊」，請勿編造答案。"
        "請使用繁體中文回答，回覆為純文字，不可使用 ** 或 * 等 markdown 格式。"
        f"\n\n問題：{query}"
    )
    response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    content = (response.choices[0].message.content or "").strip()
    content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
    content = re.sub(r'\*(.+?)\*', r'\1', content)
    return content
