# Pre-Entrega 3 — Local semantic retrieval system (RAG)

A **local RAG system** over four Spanish "TechCorp" HR policies: it ingests the
documents into a persistent **ChromaDB** collection, retrieves the fragments most
relevant to a question, and generates an answer **grounded exclusively in those
fragments**, refusing with a fixed sentence when the information is not in the
documents.

```
data/*.txt ──▶ token-based chunking (500/50) ──▶ embeddings ──▶ ChromaDB (vectorstore/)
                                                                      │
question ──▶ embeddings ──▶ similarity search (top_k=4) ─────────────┘
                                    │
                                    ▼
                    prompt | Gemini | PydanticOutputParser
                                    │
                                    ▼
                RAGResponse(answer, sources, retrieved_chunks)
```

All code (identifiers, logs, docstrings) is in English. Spanish appears only where the
end user reads it: the system prompt, the refusal sentence, the Pydantic
`description=` strings injected into the prompt, the documents in `data/`, and the two
test questions.

## Installation

Requires **Python 3.12**.

```bash
# from this folder
py -3.12 -m venv .venv          # Windows (Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate          # (source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                # registers the local_rag package for "python -m local_rag.*"

cp .env.example .env            # then fill in your API key
```

Dependencies are **pinned to exact versions** in `requirements.txt` (pin date
included): LangChain and ChromaDB move fast, and the brief does not pin versions. The
first run downloads the embedding model (~470 MB) from Hugging Face and the tiktoken
encoding file; later runs use the local cache.

## Configuration

- **`.env`** — secrets only, gitignored. Only **one** key is needed: `GOOGLE_API_KEY`,
  with a free tier at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
  Embeddings run locally and need no key.
- **[`config.yaml`](config.yaml)** — everything else: data and persistence paths, the
  collection name, **the embedding model (one place)**, the prefixes that model
  expects, the tokenizer and chunking parameters, `top_k`, the LLM model with its
  temperature/timeout/retries, and the refusal sentence.

Nothing tunable is hardcoded. The YAML is parsed at startup into a validated Pydantic
model (`AppSettings`, in `src/local_rag/config.py`), so a malformed file — `top_k`
outside 3–5, overlap greater than or equal to the chunk size — **fails fast with a
clear error**. The `LOCAL_RAG_CONFIG` environment variable points at an alternative
YAML file (the tests use it to isolate their own runs).

| Parameter | Value | Source rule |
|---|---|---|
| `chunk_size_tokens` / `chunk_overlap_tokens` | 500 / 50 | brief: "minimum 500 tokens with 50 of overlap" |
| `top_k` | 4 | brief: between 3 and 5 ("infinite context") |
| `embedding_model_name` | `intfloat/multilingual-e5-small` | local, keyless, multilingual (the corpus is Spanish), 512-token limit |
| `llm.model_name` | `gemini-3.5-flash` | verified against the API on 2026-09-14 |
| `tiktoken_encoding` | `cl100k_base` | course rule: measure in tokens, not characters |

**One embedding model, for indexing and querying.** `build_embeddings()` in
`src/local_rag/store.py` is the only function that instantiates
`HuggingFaceEmbeddings`; both `ingest.py` and `retriever.py` use it through this
module — this is how the brief's "error #1" (never mix embedding models) is avoided.
The chosen model, `multilingual-e5-small`, was trained with the prefixes
`"passage: "` (documents) and `"query: "` (questions); both live in `config.yaml` and
are applied inside the same factory, one for `embed_documents` and one for
`embed_query`.

## How to run

```bash
python -m local_rag.ingest              # loads data/, chunks it, indexes into ./vectorstore
python -m local_rag.main                # runs the answerable question and the trap question
pytest                                  # see the Tests section
```

`python -m local_rag.ingest` is **safe to re-run**: every chunk has a deterministic id
(`"<file>:<index>"`) and a content hash. Running it again re-indexes nothing that
hasn't changed (`added=0`); adding a new file to `data/` indexes its chunks; editing a
file re-embeds only its changed chunks; deleting a file removes its chunks from the
collection. This replaces the course notebook's check ("does the `vectorstore/` folder
already exist?"), which ignores new files once the index exists.

Every chunk also records the embedding model and document prefix that produced its
vector. If `embedding_model_name` or `embedding_document_prefix` changes in
`config.yaml`, the next ingest resets the collection and re-embeds everything
(`rebuilt=True`) instead of mixing vectors from two models.

With the `data/` corpus (four policies, ~700–800 words each), chunking produced
**12 chunks** — well above `top_k=4`, so retrieval is genuinely selective:

```
INFO local_rag.ingest: Collection synced: added=12 updated=0 unchanged=0 pruned=0 total=12
Collection 'techcorp_policies' at .../vectorstore holds 12 chunks
  (chunk_size=500 tokens, overlap=50, embeddings=intfloat/multilingual-e5-small)
```

## Evidence: the two required questions

Real output of `python -m local_rag.main` (2026-09-14), with `GOOGLE_API_KEY` set:

```
=== Answerable question ===
Question: ¿Cuántos días de vacaciones corresponden a un empleado con 5 años de antigüedad?
Answer:   A las personas con una antigüedad de entre 5 y 10 años cumplidos les
          corresponden 21 días corridos de vacaciones por año calendario.
Sources:  onboarding_nuevos_empleados.txt, politica_vacaciones.txt
Retrieved chunks: 4

=== Trap question ===
Question: ¿Cuál es la política de bonos por rendimiento anual en TechCorp?
Answer:   No tengo acceso a esa información en los documentos disponibles.
Sources:  onboarding_nuevos_empleados.txt, politica_seguridad_informatica.txt,
          politica_teletrabajo.txt, politica_vacaciones.txt
Retrieved chunks: 4
```

The first answer is correct and grounded: the vacation policy states that "5 to 10
years completed" earns 21 days. None of the four documents mentions bonuses, salary,
or compensation of any kind (`tests/test_chunking.py` audits the corpus against that
word list), so the refusal is genuine, not a lucky guess. Note that the first
question's sources are only 2 of the 4 files: with `top_k=4` over 12 chunks, retrieval
actually filters.

`sources` is never asked of the model: `RAGResponse.from_documents()`
(`src/local_rag/schemas.py`) builds it in code from the `source` metadata of the
documents the retriever returned — this is how the model is kept from "remembering" an
invented reference.

## How each requirement is met

| Brief requirement | Where |
|---|---|
| Ingest from `/data`, chunking that preserves semantic context | `src/local_rag/ingest.py`: `load_documents`, `clean_text`, `build_text_splitter` (`RecursiveCharacterTextSplitter.from_tiktoken_encoder`, 500/50), `split_documents` |
| Persistence without re-indexing everything | `sync_collection()`: deterministic ids + upsert + pruning of stale chunks |
| `async def get_rag_response(query: str) -> RAGResponse`, awaiting both the retriever and the LLM | `src/local_rag/chain.py`: `docs = await pipeline.retriever.ainvoke(query)` and `llm_answer = await pipeline.chain.ainvoke(...)` |
| Similarity search, `top_k` between 3 and 5 | `src/local_rag/retriever.py::build_retriever`; `AppSettings.top_k` validates the 3–5 range |
| One embedding model for indexing and querying | `src/local_rag/store.py::build_embeddings`, a single cached factory |
| Output through `PydanticOutputParser` | `src/local_rag/chain.py::OUTPUT_PARSER`, at the end of the chain |
| Pydantic object with text and references | `src/local_rag/schemas.py::RAGResponse` (`answer`, `sources`, `retrieved_chunks`) |
| Prompt as a "truth filter", refusal when the answer isn't in the context | `SYSTEM_PROMPT` in `src/local_rag/chain.py`, with `{refusal_sentence}` |
| Two tests: an answerable question and a trap question | `src/local_rag/main.py::run_demo`; evidence recorded above; `tests/test_rag_integration.py` |
| No secrets in the repo | `.env` + `.gitignore`; `.env.example` committed with empty values |

## Tests

```bash
pytest
```

35 tests. **33 run without an API key**; the other 2 (a real Gemini call) **skip
automatically** when `GOOGLE_API_KEY` is not set.

- `tests/test_schemas.py` (11) — `LLMAnswer` has exactly one field; `RAGResponse`
  rejects a negative `retrieved_chunks`, deduplicates and sorts `sources`, builds
  sources from metadata; `AppSettings` rejects `top_k` outside 3–5 and
  `overlap >= chunk_size`.
- `tests/test_chunking.py` (7) — `clean_text` keeps paragraph breaks; the 4 documents
  load with a clean file name; the corpus produces more chunks than `top_k`; every
  chunk respects the tiktoken ceiling; chunk ids are deterministic and unique; the
  corpus mentions no compensation; every chunk (with the `"passage: "` prefix) fits the
  embedding model's own 512-token limit.
- `tests/test_chain.py` (8) — `format_documents` labels every fragment with its
  source; the prompt's variables are exactly `context`/`question`; with a fake model
  (`ScriptedChatModel`), `get_rag_response` builds sources from metadata, retries on a
  malformed reply, propagates the error once retries are exhausted, and normalises a
  decorated refusal to the exact sentence.


## Decisions and scope

- **Multilingual embedding model.** The brief names no specific model; this project
  uses `intfloat/multilingual-e5-small` (local, keyless, 512-token context, trained for
  Spanish) instead of the course notebook's `all-MiniLM-L6-v2` (English-trained,
  256-token effective limit), because this project's corpus is entirely in Spanish.
- **Overlap 50, not 70.** The course notebook's code uses `chunk_overlap=70` but its
  comment says 50; this project follows the brief (50) and enforces it with a
  `model_validator` on `AppSettings`.
- **`top_k=4` over 12 chunks**, not over 4 like the course notebook: the corpus was
  written longer on purpose so retrieval actually filters.
- **LangChain's `Chroma` wrapper instead of the native `chromadb` client.** The wrapper
  exposes `as_retriever().ainvoke`, which is the async retrieval the brief asks for,
  while still allowing explicit ids, `get()` and `delete()` for deterministic
  upsert/pruning.
- **`config.yaml` + `AppSettings`, not loose constants.** Same pattern as Pre-Entrega 1
  and 2 in this repository: every tunable in a validated YAML, secrets only in `.env`.
- **Not implemented, on purpose** (outside the brief's scope): metadata filtering in
  the main flow (`build_retriever(source_filter=...)` exists and is covered by a test,
  but `main.py` doesn't use it); a fallback to a second LLM provider; caching
  embeddings between test runs.
- **Documents are read with `pathlib`, not `TextLoader`.** The course brief uses
  `TextLoader` from `langchain-community`, which is being sunset and emits a
  deprecation warning on import. For plain `.txt`/`.md` files the loader added nothing
  but that dependency, so `load_documents()` builds the `Document` objects directly.
- **Quiet runs.** The embedding model loads from the local Hugging Face cache when it is
  already there (no Hub requests, so no "unauthenticated requests" warning), automatic
  function calling is disabled explicitly on the Gemini model (the chain passes no
  tools), and per-request `httpx` logs are held at WARNING.

---

Una versión en español de este documento está disponible en [README.md](README.md).
