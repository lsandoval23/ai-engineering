"""Output contracts: what the LLM generates (LLMAnswer) and what the system returns (RAGResponse)."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

SOURCE_METADATA_KEY: str = "source"
UNKNOWN_SOURCE: str = "unknown"


class LLMAnswer(BaseModel):
    """Parser target: the only thing the LLM generates. It has no field for sources on purpose."""

    # This description is injected into the prompt by PydanticOutputParser, so it is
    # user-facing text and stays in Spanish like the rest of the prompt.
    answer: str = Field(
        description=(
            "Respuesta a la pregunta del usuario, basada EXCLUSIVAMENTE en el CONTEXTO. "
            "Si la información no está en el CONTEXTO, respondé exactamente con la frase de "
            "rechazo indicada en las instrucciones, sin agregar nada más."
        )
    )


class RAGResponse(BaseModel):
    """Public return type of ``get_rag_response``: the answer plus verifiable references."""

    answer: str
    sources: list[str] = Field(description="Data files the retrieved fragments came from")
    retrieved_chunks: int = Field(ge=0)

    @field_validator("sources")
    @classmethod
    def _unique_sorted(cls, value: list[str]) -> list[str]:
        """Sources are a set in spirit: deduplicated and sorted for a stable output."""
        return sorted(set(value))

    @classmethod
    def from_documents(cls, answer: str, docs: Sequence[Document]) -> "RAGResponse":
        """Assemble the response from the LLM answer and the retrieved documents' metadata.

        The sources are read from the real ``source`` metadata of what the retriever
        returned; the LLM is never asked for them (that is where references get invented).
        """
        sources = [doc.metadata.get(SOURCE_METADATA_KEY, UNKNOWN_SOURCE) for doc in docs]
        return cls(answer=answer, sources=sources, retrieved_chunks=len(docs))
