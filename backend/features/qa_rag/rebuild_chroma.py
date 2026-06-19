"""
從 PDF 重建 chroma_db。
執行：venv/bin/python -m backend.features.qa_rag.rebuild_chroma
"""
import shutil
from pathlib import Path

import pypdf
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH   = Path(__file__).parent / "knowledge" / "Financial_knowledge_final.pdf"
CHROMA_DIR = Path(__file__).parent / "knowledge" / "chroma_db"
MODEL_NAME = "google/embeddinggemma-300m"
CHUNK_SIZE = 500
OVERLAP    = 50


def _load_pdf(path: Path) -> list[Document]:
    reader = pypdf.PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": i + 1}))
    return docs


def _split(docs: list[Document]) -> list[Document]:
    chunks = []
    for doc in docs:
        text = doc.page_content
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(Document(page_content=chunk, metadata=doc.metadata))
            start += CHUNK_SIZE - OVERLAP
    return chunks


def main():
    print(f"[rebuild] 載入 PDF：{PDF_PATH}")
    pages = _load_pdf(PDF_PATH)
    print(f"[rebuild] 共 {len(pages)} 頁有文字")

    chunks = _split(pages)
    print(f"[rebuild] 切成 {len(chunks)} 個 chunk")

    print(f"[rebuild] 載入 embedding 模型：{MODEL_NAME}")
    embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    if CHROMA_DIR.exists():
        print("[rebuild] 刪除舊 chroma_db")
        shutil.rmtree(CHROMA_DIR)

    print("[rebuild] 建立新 chroma_db")
    Chroma.from_documents(chunks, embedding, persist_directory=str(CHROMA_DIR))
    print("[rebuild] 完成！")


if __name__ == "__main__":
    main()
