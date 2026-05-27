import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

load_dotenv()

_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "knowledge", "chroma_db")
_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_vector_store = None
_openai_client = None


def _init():
    global _vector_store, _openai_client
    if _vector_store is not None:
        return
    print("[rag_service] 初始化 embedding 模型與向量資料庫...")
    embedding = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
    _vector_store = Chroma(persist_directory=_CHROMA_DIR, embedding_function=embedding)
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
        f"請使用繁體中文回答。\n\n問題：{query}"
    )
    response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
