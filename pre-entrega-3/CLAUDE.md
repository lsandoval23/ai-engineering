# Project conventions

## Language
- All code is in English: identifiers, comments, docstrings, log messages, tests, commit messages, README.
- Spanish is used only for text the end user reads: the system prompt, the refusal sentence,
  Pydantic field descriptions injected into the prompt, the sample documents in `data/`,
  and the test questions. Do not translate those.
- Do not copy Spanish identifiers from the course notebook (e.g. `respuesta`, `fuentes`,
  `formatear_documentos`); use the English names defined in the spec.

## Layout
- Source code lives in `src/local_rag/`; tests live in `tests/`; sample data lives in `data/`.
- No code at the repository root. Configuration files only.
- Run: `python -m local_rag.ingest`, `python -m local_rag.main`, `pytest`.

## Style
- Python 3.12. `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
- Type hints on every public function. One-line module docstring per file.
- No wildcard imports, no bare `except`.

## RAG rules (graded)
- `get_rag_response(query: str)` is async and awaits both the retriever and the LLM call. Keep the name.
- Chunk by tokens: 500 tokens, 50 overlap. `TOP_K` between 3 and 5.
- One embedding model for indexing and querying, defined once in `config.yaml` and exposed through `config.py`.
- Ingest checks for an existing index and is idempotent (deterministic chunk ids + upsert).
- The LLM returns only the answer text; sources come from retrieved-document metadata.

## Secrets and artefacts
- API keys only through `.env` (loaded with python-dotenv). Never commit `.env` or `vectorstore/`.
- Keep `.env.example` up to date with empty values.
- Model names and tunables live in `config.yaml` (validated by `config.py`), not scattered in code.
