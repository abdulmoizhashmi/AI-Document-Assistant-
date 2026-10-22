from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_PATH = str(Path(__file__).resolve().parents[1] / "data" / "chroma_db")


@lru_cache(maxsize=1)
def get_embeddings(model_name: str | None = None) -> HuggingFaceEmbeddings:
    if not model_name:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_or_get_vector_store(chunks: list[Document], collection_name: str) -> Chroma:
    os.makedirs(DB_PATH, exist_ok=True)
    embeddings = get_embeddings()
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Index the provided chunks only if the collection is empty.
    # Collection name is based on uploaded document content hash.
    if store._collection.count() == 0 and chunks:
        ids = [chunk.metadata["id"] for chunk in chunks]
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            store.add_documents(documents=chunks[i:i + batch_size], ids=ids[i:i + batch_size])

    return store
