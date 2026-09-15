"""schemas.py: LLMAnswer, RAGResponse, and the AppSettings validators. No API key needed."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from local_rag.chain import OUTPUT_PARSER
from local_rag.config import get_settings
from local_rag.schemas import LLMAnswer, RAGResponse


def test_llm_answer_has_exactly_one_field():
    assert set(LLMAnswer.model_fields) == {"answer"}


def test_rag_response_accepts_valid_input():
    response = RAGResponse(answer="21 días", sources=["a.txt", "b.txt"], retrieved_chunks=2)
    assert response.answer == "21 días"
    assert response.retrieved_chunks == 2


def test_rag_response_rejects_negative_chunk_count():
    with pytest.raises(ValidationError):
        RAGResponse(answer="x", sources=[], retrieved_chunks=-1)


def test_rag_response_sources_are_deduplicated_and_sorted():
    response = RAGResponse(
        answer="x", sources=["b.txt", "a.txt", "b.txt", "c.txt"], retrieved_chunks=3
    )
    assert response.sources == ["a.txt", "b.txt", "c.txt"]


def test_from_documents_builds_sources_from_metadata():
    docs = [
        Document(page_content="p1", metadata={"source": "b.txt"}),
        Document(page_content="p2", metadata={"source": "a.txt"}),
        Document(page_content="p3", metadata={"source": "a.txt"}),
    ]
    response = RAGResponse.from_documents(answer="21 días", docs=docs)
    assert response.sources == ["a.txt", "b.txt"]
    assert response.retrieved_chunks == 3


def test_from_documents_handles_missing_source_metadata():
    docs = [Document(page_content="p1", metadata={})]
    response = RAGResponse.from_documents(answer="x", docs=docs)
    assert response.sources == ["unknown"]


def test_format_instructions_contain_the_spanish_description():
    instructions = OUTPUT_PARSER.get_format_instructions()
    assert "CONTEXTO" in instructions


@pytest.mark.parametrize("top_k", [2, 6])
def test_app_settings_rejects_top_k_outside_3_to_5(custom_config_file, top_k):
    custom_config_file(top_k=top_k)
    with pytest.raises(ValidationError):
        get_settings()


def test_app_settings_rejects_overlap_not_below_chunk_size(custom_config_file):
    custom_config_file(chunk_size_tokens=500, chunk_overlap_tokens=500)
    with pytest.raises(ValidationError):
        get_settings()


def test_real_config_yaml_loads():
    settings = get_settings()
    assert settings.top_k in (3, 4, 5)
    assert settings.chunk_overlap_tokens < settings.chunk_size_tokens
    assert settings.embedding_model_name == "intfloat/multilingual-e5-small"
