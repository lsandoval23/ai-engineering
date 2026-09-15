"""Entry point: runs the answerable question and the trap question."""

from __future__ import annotations

import asyncio
import sys

from local_rag.chain import get_rag_response
from local_rag.config import get_settings, load_environment
from local_rag.logging_config import configure_logging
from local_rag.schemas import RAGResponse
from local_rag.store import count_chunks, load_vector_store

# User-facing test questions (Spanish, the documents' language).
ANSWERABLE_QUESTION: str = (
    "¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?"
)
TRAP_QUESTION: str = "¿Cuál es la política de bonos por rendimiento anual en TechCorp?"


def print_response(title: str, question: str, response: RAGResponse) -> None:
    """Print every field of the response: the course notebook's loop forgot the answer."""
    print(f"\n=== {title} ===")
    print(f"Question: {question}")
    print(f"Answer:   {response.answer}")
    print(f"Sources:  {', '.join(response.sources)}")
    print(f"Retrieved chunks: {response.retrieved_chunks}")


async def run_demo() -> None:
    """The two tests the brief requires: a grounded answer and a refusal."""
    answerable = await get_rag_response(ANSWERABLE_QUESTION)
    print_response("Answerable question", ANSWERABLE_QUESTION, answerable)

    trap = await get_rag_response(TRAP_QUESTION)
    print_response("Trap question", TRAP_QUESTION, trap)


def main() -> None:
    """``python -m local_rag.main``."""
    configure_logging()
    load_environment()

    if count_chunks(load_vector_store()) == 0:
        settings = get_settings()
        sys.exit(
            f"The collection '{settings.collection_name}' is empty. "
            "Run 'python -m local_rag.ingest' first."
        )

    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
