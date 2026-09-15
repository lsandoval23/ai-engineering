"""Retrieval layer: similarity search over the persisted collection with top_k in 3-5."""

from __future__ import annotations

from typing import Any

from langchain_core.vectorstores import VectorStoreRetriever

from local_rag.config import get_settings
from local_rag.schemas import SOURCE_METADATA_KEY
from local_rag.store import load_vector_store


def build_retriever(source_filter: str | None = None) -> VectorStoreRetriever:
    """Similarity retriever over the collection, using the same embeddings that indexed it.

    ``source_filter`` restricts the search to one data file through Chroma's metadata
    ``where`` filter (the course's "filter before the vector search" idea).
    """
    settings = get_settings()
    search_kwargs: dict[str, Any] = {"k": settings.top_k}
    if source_filter is not None:
        search_kwargs["filter"] = {SOURCE_METADATA_KEY: source_filter}
    return load_vector_store().as_retriever(search_type="similarity", search_kwargs=search_kwargs)
