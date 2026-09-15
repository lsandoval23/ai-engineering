"""Ingest pipeline: load data/ -> clean -> chunk by tokens -> deterministic ids -> upsert into ChromaDB."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_rag.config import get_settings
from local_rag.logging_config import configure_logging
from local_rag.schemas import SOURCE_METADATA_KEY
from local_rag.store import count_chunks, embedding_fingerprint, load_vector_store

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".txt", ".md"})
CHUNK_INDEX_KEY: str = "chunk_index"
CONTENT_HASH_KEY: str = "content_hash"
EMBEDDING_FINGERPRINT_KEY: str = "embedding_fingerprint"
# Paragraph -> line -> sentence -> word -> character: the course's recursive hierarchy
# plus the sentence separator the DocumentProcessor exercise adds.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_INLINE_WHITESPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")


def load_documents(data_path: Path | None = None) -> list[Document]:
    """Read every .txt/.md file under the data folder, in sorted order, as one Document each.

    The ``source`` metadata is the bare file name, not the OS path (backslashes on
    Windows), so chunk ids stay platform-independent. Files are read with pathlib
    rather than langchain-community's TextLoader, a package that is being sunset.
    """
    folder = data_path or get_settings().data_path
    if not folder.is_dir():
        raise FileNotFoundError(f"data directory not found: {folder}")
    documents = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={SOURCE_METADATA_KEY: path.name})
        for path in sorted(folder.iterdir())
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    logger.info("Loaded %d documents from %s", len(documents), folder)
    return documents


def clean_text(text: str) -> str:
    """Normalise noise without destroying paragraph boundaries.

    Collapses runs of spaces/tabs, trailing spaces and 3+ blank lines; keeps ``\\n\\n``
    because it is the splitter's first separator.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WHITESPACE.sub(" ", text)
    text = _SPACE_BEFORE_NEWLINE.sub("\n", text)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    """Recursive splitter whose chunk_size/overlap are measured in tiktoken tokens."""
    settings = get_settings()
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=settings.tiktoken_encoding,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
        separators=list(SEPARATORS),
    )


def split_documents(documents: Sequence[Document]) -> list[Document]:
    """Clean, then split keeping metadata; add a per-source chunk index and a content hash."""
    cleaned = [
        Document(page_content=clean_text(doc.page_content), metadata=dict(doc.metadata))
        for doc in documents
    ]
    chunks = build_text_splitter().split_documents(cleaned)
    next_index: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata[SOURCE_METADATA_KEY]
        index = next_index.get(source, 0)
        next_index[source] = index + 1
        chunk.metadata[CHUNK_INDEX_KEY] = index
        chunk.metadata[CONTENT_HASH_KEY] = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()
    logger.info("Split %d documents into %d chunks", len(documents), len(chunks))
    return chunks


def build_chunk_id(source: str, chunk_index: int) -> str:
    """Deterministic id: the same file and position always map to the same record."""
    return f"{source}:{chunk_index}"


@dataclass(frozen=True)
class IngestReport:
    """What one ingest run did to the collection."""

    added: int
    updated: int
    unchanged: int
    pruned: int
    total: int
    rebuilt: bool


def sync_collection(chunks: Sequence[Document], store: Chroma) -> IngestReport:
    """Make the collection mirror ``chunks``: upsert new/changed ids, skip unchanged, prune stale.

    This is the "check for an existing index before re-indexing" step done per chunk
    instead of per folder: re-running is a no-op, a new file gets indexed, an edited
    file gets re-embedded, and a deleted file's chunks disappear.

    A chunk is only "unchanged" if both its text and the embedding config that produced
    its vector match. If any stored chunk was embedded with another model or document
    prefix, the whole collection is reset and rebuilt: upserting into it would either
    fail on a dimension mismatch or silently mix vectors from two models.
    """
    fingerprint = embedding_fingerprint()
    existing = store.get(include=["metadatas"])
    existing_metadatas = [metadata or {} for metadata in existing["metadatas"]]
    rebuilt = any(
        metadata.get(EMBEDDING_FINGERPRINT_KEY) != fingerprint for metadata in existing_metadatas
    )
    existing_hashes: dict[str, str | None] = {}
    if rebuilt:
        logger.warning(
            "Collection was embedded with a different config than %r; rebuilding all %d chunks",
            fingerprint, len(existing_metadatas),
        )
        store.reset_collection()
    else:
        existing_hashes = {
            chunk_id: metadata.get(CONTENT_HASH_KEY)
            for chunk_id, metadata in zip(existing["ids"], existing_metadatas)
        }

    to_upsert: list[Document] = []
    upsert_ids: list[str] = []
    added = updated = unchanged = 0
    for chunk in chunks:
        chunk_id = build_chunk_id(chunk.metadata[SOURCE_METADATA_KEY], chunk.metadata[CHUNK_INDEX_KEY])
        if chunk_id not in existing_hashes:
            added += 1
        elif existing_hashes[chunk_id] == chunk.metadata[CONTENT_HASH_KEY]:
            unchanged += 1
            continue
        else:
            updated += 1
        chunk.metadata[EMBEDDING_FINGERPRINT_KEY] = fingerprint
        to_upsert.append(chunk)
        upsert_ids.append(chunk_id)

    if to_upsert:
        store.add_documents(to_upsert, ids=upsert_ids)

    current_ids = {
        build_chunk_id(c.metadata[SOURCE_METADATA_KEY], c.metadata[CHUNK_INDEX_KEY]) for c in chunks
    }
    stale = sorted(set(existing_hashes) - current_ids)
    if stale:
        store.delete(ids=stale)

    report = IngestReport(
        added=added,
        updated=updated,
        unchanged=unchanged,
        pruned=len(stale),
        total=count_chunks(store),
        rebuilt=rebuilt,
    )
    logger.info(
        "Collection synced: added=%d updated=%d unchanged=%d pruned=%d total=%d rebuilt=%s",
        report.added, report.updated, report.unchanged, report.pruned, report.total, report.rebuilt,
    )
    return report


def ingest_documents(data_path: Path | None = None) -> int:
    """Full pipeline: load -> clean -> chunk -> embed -> persist. Returns the collection size."""
    chunks = split_documents(load_documents(data_path))
    report = sync_collection(chunks, load_vector_store())
    return report.total


def main() -> None:
    """Entry point: ``python -m local_rag.ingest``."""
    configure_logging()
    settings = get_settings()
    total = ingest_documents()
    print(
        f"Collection '{settings.collection_name}' at {settings.persist_path} holds {total} chunks "
        f"(chunk_size={settings.chunk_size_tokens} tokens, overlap={settings.chunk_overlap_tokens}, "
        f"embeddings={settings.embedding_model_name})"
    )


if __name__ == "__main__":
    main()
