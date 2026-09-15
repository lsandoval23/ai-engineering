"""Vector store factories: the single embedding model and the persisted ChromaDB collection."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from local_rag.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def build_embeddings() -> HuggingFaceEmbeddings:
    """The ONE embedding model, used both to index and to query.

    Cached so that ingest and retrieval share the same object within a process. The
    model name comes from ``config.yaml``; nothing else in the code names a model.
    e5 models were trained with a ``"passage: "`` prefix on documents and a ``"query: "``
    prefix on queries, so each side gets its own encode kwargs.

    The model is loaded from the local Hugging Face cache first: once downloaded, runs
    make no Hub requests (faster, offline-capable, and no unauthenticated-request
    warning). Only a cache miss — the first run — goes to the network.
    """
    settings = get_settings()
    embedding_kwargs: dict[str, Any] = {
        "model_name": settings.embedding_model_name,
        "encode_kwargs": {
            "prompt": settings.embedding_document_prefix,
            "normalize_embeddings": True,
        },
        "query_encode_kwargs": {
            "prompt": settings.embedding_query_prefix,
            "normalize_embeddings": True,
        },
    }
    try:
        return HuggingFaceEmbeddings(model_kwargs={"local_files_only": True}, **embedding_kwargs)
    except OSError:
        logger.info("Embedding model %s not cached; downloading it", settings.embedding_model_name)
        return HuggingFaceEmbeddings(**embedding_kwargs)


def load_vector_store() -> Chroma:
    """Open (or create) the persisted collection with the shared embedding model."""
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=build_embeddings(),
        persist_directory=str(settings.persist_path),
    )


def count_chunks(store: Chroma) -> int:
    """Number of chunks in the collection, through the public API."""
    return len(store.get(include=[])["ids"])
