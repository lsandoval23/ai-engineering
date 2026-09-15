"""chain.py: prompt shape, format_documents, and get_rag_response with fakes. No API key needed."""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import local_rag.chain as chain_module
from local_rag.chain import RAGPipeline, build_chain, build_prompt, format_documents, get_rag_response
from local_rag.config import get_settings


def test_format_documents_labels_each_chunk_with_its_source():
    docs = [
        Document(page_content="Texto uno", metadata={"source": "a.txt"}),
        Document(page_content="Texto dos", metadata={"source": "b.txt"}),
    ]
    formatted = format_documents(docs)
    assert "[Fuente: a.txt]" in formatted
    assert "[Fuente: b.txt]" in formatted
    assert "Texto uno" in formatted and "Texto dos" in formatted
    assert "---" in formatted


def test_prompt_input_variables_are_context_and_question():
    prompt = build_prompt()
    assert set(prompt.input_variables) == {"context", "question"}


def test_prompt_system_message_contains_refusal_sentence_and_format_instructions():
    prompt = build_prompt()
    rendered = prompt.format_messages(context="ctx", question="q")
    system_text = rendered[0].content
    assert get_settings().refusal_sentence in system_text
    assert "CONTEXTO" in system_text  # part of the format instructions or human template


def _retriever_returning(docs: list[Document]) -> RunnableLambda:
    return RunnableLambda(lambda _query: docs)


@pytest.mark.asyncio
async def test_get_rag_response_assembles_sources_from_retrieved_metadata(scripted, monkeypatch):
    docs = [
        Document(page_content="p1", metadata={"source": "b.txt"}),
        Document(page_content="p2", metadata={"source": "a.txt"}),
        Document(page_content="p3", metadata={"source": "a.txt"}),
    ]
    model = scripted(AIMessage(content=json.dumps({"answer": "21 días"}, ensure_ascii=False)))
    pipeline = RAGPipeline(retriever=_retriever_returning(docs), chain=build_chain(model))
    monkeypatch.setattr(chain_module, "get_pipeline", lambda: pipeline)

    response = await get_rag_response("¿Cuántos días de vacaciones tengo?")

    assert response.answer == "21 días"
    assert response.sources == ["a.txt", "b.txt"]
    assert response.retrieved_chunks == 3


@pytest.mark.asyncio
async def test_get_rag_response_retries_after_a_malformed_reply(scripted, monkeypatch):
    docs = [Document(page_content="p1", metadata={"source": "a.txt"})]
    model = scripted(
        AIMessage(content="not json at all"),
        AIMessage(content=json.dumps({"answer": "ok"})),
    )
    pipeline = RAGPipeline(retriever=_retriever_returning(docs), chain=build_chain(model))
    monkeypatch.setattr(chain_module, "get_pipeline", lambda: pipeline)

    response = await get_rag_response("pregunta")

    assert response.answer == "ok"
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_get_rag_response_raises_after_exhausting_retries(scripted, monkeypatch):
    settings = get_settings()
    docs = [Document(page_content="p1", metadata={"source": "a.txt"})]
    model = scripted(*(AIMessage(content="still not json") for _ in range(settings.llm.max_attempts)))
    pipeline = RAGPipeline(retriever=_retriever_returning(docs), chain=build_chain(model))
    monkeypatch.setattr(chain_module, "get_pipeline", lambda: pipeline)

    with pytest.raises(OutputParserException):
        await get_rag_response("pregunta")


@pytest.mark.asyncio
async def test_get_rag_response_normalises_a_decorated_refusal(scripted, monkeypatch):
    refusal = get_settings().refusal_sentence
    docs = [Document(page_content="p1", metadata={"source": "a.txt"})]
    model = scripted(AIMessage(content=json.dumps({"answer": f"Lo siento, {refusal}"})))
    pipeline = RAGPipeline(retriever=_retriever_returning(docs), chain=build_chain(model))
    monkeypatch.setattr(chain_module, "get_pipeline", lambda: pipeline)

    response = await get_rag_response("¿Cuál es la política de bonos?")

    assert response.answer == refusal
