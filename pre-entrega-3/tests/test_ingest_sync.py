"""ingest.sync_collection against a throwaway Chroma collection with fake embeddings. Runs offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from local_rag.ingest import CHUNK_INDEX_KEY, CONTENT_HASH_KEY, sync_collection
from local_rag.schemas import SOURCE_METADATA_KEY

TEXTS: tuple[str, ...] = ("uno", "dos", "tres")


def _chunks() -> list[Document]:
    return [
        Document(
            page_content=text,
            metadata={SOURCE_METADATA_KEY: "a.txt", CHUNK_INDEX_KEY: index, CONTENT_HASH_KEY: f"hash-{text}"},
        )
        for index, text in enumerate(TEXTS)
    ]


def _store(tmp_path: Path, dimensions: int) -> Chroma:
    return Chroma(
        collection_name="sync_test",
        embedding_function=DeterministicFakeEmbedding(size=dimensions),
        persist_directory=str(tmp_path / "vectorstore"),
    )


def test_rerun_with_the_same_embedding_config_embeds_nothing(tmp_path, custom_config_file):
    custom_config_file()
    sync_collection(_chunks(), _store(tmp_path, 8))

    report = sync_collection(_chunks(), _store(tmp_path, 8))

    assert (report.added, report.updated, report.unchanged, report.rebuilt) == (0, 0, len(TEXTS), False)


@pytest.mark.parametrize(
    ("override", "dimensions"),
    [
        ({"embedding_model_name": "another/model"}, 16),  # new model with different dimensions
        ({"embedding_document_prefix": "document: "}, 8),  # same dimensions, different vectors
    ],
)
def test_changing_the_embedding_config_rebuilds_the_collection(
    tmp_path, custom_config_file, override: dict[str, Any], dimensions: int
):
    custom_config_file()
    sync_collection(_chunks(), _store(tmp_path, 8))

    custom_config_file(**override)
    store = _store(tmp_path, dimensions)
    report = sync_collection(_chunks(), store)

    assert report.rebuilt
    assert (report.added, report.unchanged, report.total) == (len(TEXTS), 0, len(TEXTS))
    assert len(store.similarity_search("uno", k=1)) == 1
