"""
Sentence-Transformer Singleton wrapper，用於新聞向量搜尋。

程式啟動後第一次呼叫時載入模型，之後所有執行緒共用同一個實例。
多執行緒並行呼叫 encode_documents() 是安全的（CPU inference read-only）。
"""
from __future__ import annotations

import threading

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model: SentenceTransformer | None = None
_load_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                print(f"[embedder] 載入模型 {_MODEL_NAME!r} ...")
                _model = SentenceTransformer(_MODEL_NAME)
                print("[embedder] 模型載入完成")
    return _model


def encode_query(text: str) -> np.ndarray:
    """把使用者輸入的 query 轉成向量（1-D, 384 維）。"""
    emb = _get_model().encode(text, normalize_embeddings=True)
    return np.array(emb).flatten()


def encode_documents(texts: list[str]) -> np.ndarray:
    """把多篇文章標題批次轉成向量（2-D, shape: [N, 384]）。"""
    embs = _get_model().encode(texts, normalize_embeddings=True)
    return np.array(embs)


def cosine_similarities(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """
    計算 query 向量與每篇文章向量的 cosine similarity。

    Returns:
        1-D ndarray，長度 = len(doc_embs)，值域約 [-1, 1]
    """
    # 已在 encode 時做 normalize，dot product 即等於 cosine similarity
    sims = doc_embs @ query_emb
    return np.array(sims).flatten()
