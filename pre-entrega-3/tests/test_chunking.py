"""ingest.py: loading, cleaning and chunking data/. Runs offline (tiktoken file caches locally)."""

from __future__ import annotations

import tiktoken

from local_rag.config import get_settings
from local_rag.ingest import (
    CHUNK_INDEX_KEY,
    CONTENT_HASH_KEY,
    build_chunk_id,
    clean_text,
    load_documents,
    split_documents,
)
from local_rag.schemas import SOURCE_METADATA_KEY

# Words that would make the trap question ("annual performance bonus policy") answerable.
# None of the four documents may contain them.
COMPENSATION_WORDS = (
    "bono", "bonos", "bonus", "salario", "salarios", "sueldo", "sueldos",
    "remuneración", "remuneraciones", "compensación", "compensaciones",
    "incentivo", "incentivos", "premio", "premios", "aguinaldo",
)


def test_clean_text_collapses_inline_whitespace_but_keeps_paragraph_breaks():
    dirty = "Uno   dos\t\ttres.\n\n\n\nCuatro cinco.  \n"
    cleaned = clean_text(dirty)
    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "\n\n" in cleaned


def test_load_documents_returns_four_docs_with_bare_file_names():
    docs = load_documents()
    assert len(docs) == 4
    sources = {doc.metadata[SOURCE_METADATA_KEY] for doc in docs}
    assert sources == {
        "politica_vacaciones.txt",
        "politica_teletrabajo.txt",
        "politica_seguridad_informatica.txt",
        "onboarding_nuevos_empleados.txt",
    }
    for source in sources:
        assert "/" not in source and "\\" not in source


def test_chunk_count_exceeds_top_k():
    settings = get_settings()
    chunks = split_documents(load_documents())
    assert len(chunks) > settings.top_k


def test_every_chunk_is_within_the_token_ceiling():
    settings = get_settings()
    encoding = tiktoken.get_encoding(settings.tiktoken_encoding)
    for chunk in split_documents(load_documents()):
        assert len(encoding.encode(chunk.page_content)) <= settings.chunk_size_tokens


def test_every_chunk_has_source_index_and_hash_metadata():
    for chunk in split_documents(load_documents()):
        assert SOURCE_METADATA_KEY in chunk.metadata
        assert CHUNK_INDEX_KEY in chunk.metadata
        assert CONTENT_HASH_KEY in chunk.metadata


def test_chunk_ids_are_unique_and_deterministic():
    chunks = split_documents(load_documents())
    first_ids = [build_chunk_id(c.metadata[SOURCE_METADATA_KEY], c.metadata[CHUNK_INDEX_KEY]) for c in chunks]
    assert len(first_ids) == len(set(first_ids))

    chunks_again = split_documents(load_documents())
    second_ids = [
        build_chunk_id(c.metadata[SOURCE_METADATA_KEY], c.metadata[CHUNK_INDEX_KEY]) for c in chunks_again
    ]
    assert first_ids == second_ids


def test_corpus_does_not_mention_compensation():
    """The trap question must stay genuinely uncovered even if the documents are edited."""
    for doc in load_documents():
        lowered = doc.page_content.lower()
        for word in COMPENSATION_WORDS:
            assert word not in lowered, f"{doc.metadata[SOURCE_METADATA_KEY]} mentions {word!r}"


def test_chunks_fit_the_embedding_model_token_limit():
    """The e5 tokenizer differs from tiktoken; verify chunks (with the document prefix) still fit."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        import pytest as _pytest

        _pytest.skip("transformers not available")

    settings = get_settings()
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model_name)
    except OSError:
        import pytest as _pytest

        _pytest.skip("embedding tokenizer not cached locally")

    max_length = getattr(tokenizer, "model_max_length", 512)
    for chunk in split_documents(load_documents()):
        text = settings.embedding_document_prefix + chunk.page_content
        token_count = len(tokenizer.encode(text))
        assert token_count <= max_length
