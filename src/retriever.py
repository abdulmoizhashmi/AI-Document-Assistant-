from __future__ import annotations

import os
from functools import lru_cache

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_chroma import Chroma


@lru_cache(maxsize=2)
def _load_reranker(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name, device="cpu")


def retrieve_and_rerank(
    store: Chroma,
    question: str,
    candidate_count: int = 6,
    final_count: int = 4,
    use_reranker: bool = True,
) -> list[Document]:
    """Retrieve semantic candidates from ChromaDB, optionally rerank by question relevance."""
    # Fast path: if reranking disabled, fetch final_count directly
    fetch_k = candidate_count if use_reranker else final_count
    candidates = store.similarity_search(question, k=fetch_k)
    if not candidates:
        return []

    if not use_reranker or len(candidates) <= 1:
        return candidates[:final_count]

    model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = _load_reranker(model_name)
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda pair: float(pair[0]), reverse=True)
    return [doc for _, doc in ranked[:final_count]]
