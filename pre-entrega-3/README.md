# Pre-Entrega 3 — Sistema de recuperación semántica local (RAG)

Un **sistema RAG local** sobre cuatro políticas de RR. HH. de "TechCorp" en español:
ingesta los documentos en una colección persistente de **ChromaDB**, recupera los
fragmentos más relevantes para una pregunta y genera una respuesta **basada
exclusivamente en esos fragmentos**, rechazando con una frase fija cuando la
información no está en los documentos.

```
data/*.txt ──▶ chunking por tokens (500/50) ──▶ embeddings ──▶ ChromaDB (vectorstore/)
                                                                      │
pregunta ──▶ embeddings ──▶ similarity search (top_k=4) ─────────────┘
                                    │
                                    ▼
                    prompt | Gemini | PydanticOutputParser
                                    │
                                    ▼
                RAGResponse(answer, sources, retrieved_chunks)
```

Todo el código (identificadores, logs, docstrings) está en inglés. El español aparece
solo donde lo lee la persona usuaria final: el prompt del sistema, la frase de rechazo,
las `description=` de Pydantic que se inyectan en el prompt, los documentos de `data/`
y las dos preguntas de prueba.

## Instalación

Requiere **Python 3.12**.

```bash
# desde esta carpeta
py -3.12 -m venv .venv          # Windows (Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate          # (source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                # registra el paquete local_rag para "python -m local_rag.*"

cp .env.example .env            # después completá tu API key
```

Las dependencias están **fijadas a versión exacta** en `requirements.txt` (fecha de
fijación incluida): LangChain y ChromaDB cambian rápido y la consigna no fija
versiones. La primera corrida descarga el modelo de embeddings (~470 MB) desde
Hugging Face y el archivo de codificación de tiktoken; corridas siguientes usan el
caché local.

## Configuración

- **`.env`** — solo secretos, está en `.gitignore`. Alcanza con **una** clave:
  `GOOGLE_API_KEY`, con free tier en
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Los embeddings
  corren localmente y no necesitan clave.
- **[`config.yaml`](config.yaml)** — todo lo demás: rutas de datos y persistencia,
  nombre de la colección, **el modelo de embeddings (un solo lugar)**, los prefijos que
  ese modelo espera, la codificación y los parámetros de chunking, `top_k`, el modelo de
  LLM con su temperatura/timeout/reintentos, y la frase de rechazo.

Nada tuneable está hardcodeado. El YAML se parsea al arranque en un modelo Pydantic
validado (`AppSettings`, en `src/local_rag/config.py`), así que un archivo mal
formado —`top_k` fuera de 3–5, overlap mayor o igual al tamaño de chunk, un proveedor
sin modelo— **falla rápido y con un error claro**. La variable de entorno
`LOCAL_RAG_CONFIG` apunta a un YAML alternativo (los tests la usan para aislar sus
propias corridas).

| Parámetro | Valor | Regla de origen |
|---|---|---|
| `chunk_size_tokens` / `chunk_overlap_tokens` | 500 / 50 | consigna: "mínimo 500 tokens con 50 de overlap" |
| `top_k` | 4 | consigna: entre 3 y 5 ("contexto infinito") |
| `embedding_model_name` | `intfloat/multilingual-e5-small` | local, sin clave, multilingüe (el corpus es español), límite de 512 tokens |
| `llm.model_name` | `gemini-3.5-flash` | verificado contra la API el 2026-09-14 |
| `tiktoken_encoding` | `cl100k_base` | regla de la clase: medir en tokens, no en caracteres |

**Un solo modelo de embeddings, para indexar y para consultar.** `build_embeddings()`
en `src/local_rag/store.py` es la única función que instancia `HuggingFaceEmbeddings`;
tanto `ingest.py` como `retriever.py` la usan a través de este módulo — así se evita el
"error #1" de la consigna (nunca mezclar modelos de embeddings). El modelo elegido,
`multilingual-e5-small`, fue entrenado con los prefijos `"passage: "` (documentos) y
`"query: "` (preguntas); ambos están en `config.yaml` y se aplican dentro de la misma
fábrica, uno para `embed_documents` y otro para `embed_query`.

## Cómo correr

```bash
python -m local_rag.ingest              # carga data/, chunkea, indexa en ./vectorstore
python -m local_rag.main                # corre la pregunta respondible y la pregunta trampa
python -m local_rag.main --interactive  # además, abre un prompt para preguntas propias
pytest                                  # ver sección Tests
```

`python -m local_rag.ingest` es **re-ejecutable y seguro**: cada chunk tiene un id
determinístico (`"<archivo>:<índice>"`) y un hash de contenido. Volver a correrlo no
reindexa nada sin cambios (`added=0`); si se agrega un archivo nuevo a `data/`, sus
chunks se indexan; si se edita un archivo, solo sus chunks cambiados se reembeben; si
se borra un archivo, sus chunks se eliminan de la colección. Esto reemplaza el chequeo
del notebook de la clase ("¿existe la carpeta `vectorstore/`?"), que ignora los
archivos nuevos una vez que el índice ya existe.

Con el corpus de `data/` (cuatro políticas, ~700–800 palabras cada una), el chunking
produjo **12 chunks** — bien por encima de `top_k=4`, así que la recuperación es
realmente selectiva:

```
INFO local_rag.ingest: Collection synced: added=12 updated=0 unchanged=0 pruned=0 total=12
Collection 'techcorp_policies' at .../vectorstore holds 12 chunks
  (chunk_size=500 tokens, overlap=50, embeddings=intfloat/multilingual-e5-small)
```

## Evidencia: las dos preguntas obligatorias

Salida real de `python -m local_rag.main` (2026-09-14), con `GOOGLE_API_KEY` configurada:

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

La primera respuesta es correcta y está fundamentada: la política de vacaciones dice
que "de 5 a 10 años cumplidos" corresponden 21 días. Ningún documento del corpus
menciona bonos, sueldos ni compensación de ningún tipo (`tests/test_chunking.py` audita
el corpus contra esa lista de palabras), así que el rechazo es genuino, no un acierto de
casualidad. Nótese que las fuentes de la primera pregunta son solo 2 de los 4 archivos:
con `top_k=4` sobre 12 chunks, la recuperación filtra de verdad.

Las `sources` nunca se le piden al modelo: `RAGResponse.from_documents()`
(`src/local_rag/schemas.py`) las arma en código a partir de los metadatos `source` de
los documentos que devolvió el retriever — así se evita que el modelo "recuerde" una
referencia inventada.

## Cómo se cumple cada requisito

| Requisito de la consigna | Dónde |
|---|---|
| Ingesta desde `/data`, chunking que preserva contexto semántico | `src/local_rag/ingest.py`: `load_documents`, `clean_text`, `build_text_splitter` (`RecursiveCharacterTextSplitter.from_tiktoken_encoder`, 500/50), `split_documents` |
| Persistencia sin reindexar todo | `sync_collection()`: ids determinísticos + upsert + poda de chunks obsoletos |
| `async def get_rag_response(query: str) -> RAGResponse`, con `await` en el retriever y en el LLM | `src/local_rag/chain.py`: `docs = await pipeline.retriever.ainvoke(query)` y `llm_answer = await pipeline.chain.ainvoke(...)` |
| Búsqueda por similitud, `top_k` entre 3 y 5 | `src/local_rag/retriever.py::build_retriever`; `AppSettings.top_k` valida el rango 3–5 |
| Un solo modelo de embeddings para indexar y consultar | `src/local_rag/store.py::build_embeddings`, única fábrica, cacheada |
| Salida a través de `PydanticOutputParser` | `src/local_rag/chain.py::OUTPUT_PARSER`, al final de la cadena |
| Objeto Pydantic con texto y referencias | `src/local_rag/schemas.py::RAGResponse` (`answer`, `sources`, `retrieved_chunks`) |
| Prompt como "filtro de verdad", rechazo si la respuesta no está en el contexto | `SYSTEM_PROMPT` en `src/local_rag/chain.py`, con `{refusal_sentence}` |
| Dos pruebas: pregunta respondible y pregunta trampa | `src/local_rag/main.py::run_demo`; evidencia grabada arriba; `tests/test_rag_integration.py` |
| Sin claves en el repo | `.env` + `.gitignore`; `.env.example` committeado con valores vacíos |

## Tests

```bash
pytest
```

35 tests. **33 corren sin clave de API**; los otros 2 (una respuesta real de Gemini)
se **saltan automáticamente** si `GOOGLE_API_KEY` no está configurada.

- `tests/test_schemas.py` (11) — `LLMAnswer` tiene un solo campo; `RAGResponse` rechaza
  `retrieved_chunks` negativo, deduplica y ordena `sources`, arma las fuentes desde
  metadatos; `AppSettings` rechaza `top_k` fuera de 3–5 y `overlap >= chunk_size`.
- `tests/test_chunking.py` (7) — `clean_text` conserva los párrafos; los 4 documentos
  cargan con nombre de archivo limpio; el corpus produce más chunks que `top_k`; cada
  chunk respeta el techo de tokens de tiktoken; ids determinísticos y únicos; el corpus
  no menciona compensación; cada chunk (con el prefijo `"passage: "`) entra en el límite
  de 512 tokens del tokenizer real del modelo de embeddings.
- `tests/test_chain.py` (8) — `format_documents` etiqueta cada fragmento con su fuente;
  las variables del prompt son exactamente `context`/`question`; con un modelo falso
  (`ScriptedChatModel`), `get_rag_response` arma las fuentes desde los metadatos,
  reintenta ante una respuesta mal formada, propaga el error tras agotar los reintentos,
  y normaliza un rechazo decorado a la frase exacta.
- `tests/test_rag_integration.py` (9) — con el modelo de embeddings real y una colección
  temporal: la ingesta es idempotente, indexa un archivo nuevo, poda uno eliminado, y la
  pregunta de vacaciones recupera la política de vacaciones primero. Las dos últimas
  pruebas (con clave) verifican la respuesta correcta real y la frase de rechazo exacta.

## Decisiones y alcance

- **Modelo de embeddings multilingüe.** La consigna no exige un modelo específico; se
  eligió `intfloat/multilingual-e5-small` (local, sin clave, 512 tokens de contexto,
  entrenado para español) en vez de `all-MiniLM-L6-v2` del notebook de la clase
  (entrenado en inglés, 256 tokens efectivos), porque el corpus de este proyecto es
  íntegramente en español.
- **Overlap 50, no 70.** El notebook de la clase usa `chunk_overlap=70` en el código
  pero dice 50 en el comentario; este proyecto sigue la consigna (50) y lo valida con un
  `model_validator` en `AppSettings`.
- **`top_k=4` sobre 12 chunks**, no sobre 4 como en el notebook de la clase: el corpus
  se escribió más largo a propósito para que la recuperación filtre de verdad.
- **`LangChain` (native `Chroma`) en vez del cliente nativo de `chromadb`.** El wrapper
  expone `as_retriever().ainvoke`, que es la recuperación asíncrona que pide la
  consigna, y aun así permite ids explícitos, `get()` y `delete()` para el
  upsert/poda determinístico.
- **`config.yaml` + `AppSettings`, no constantes sueltas.** Mismo patrón que
  Pre-Entrega 1 y 2 de este repositorio: todo lo tuneable en un YAML validado, secretos
  solo en `.env`.
- **No implementado, a propósito** (fuera del alcance de la consigna): filtrado por
  metadatos en el flujo principal (`build_retriever(source_filter=...)` existe y está
  cubierto por un test, pero `main.py` no lo usa); fallback a un segundo proveedor de
  LLM; caché de embeddings entre corridas de test.
- **Los documentos se leen con `pathlib`, no con `TextLoader`.** La consigna del curso
  usa `TextLoader` de `langchain-community`, que está en vías de discontinuación y emite
  un warning de deprecación al importarse. Para archivos `.txt`/`.md` planos el loader
  no aportaba nada más que esa dependencia, así que `load_documents()` arma los
  `Document` directamente.
- **Ejecuciones sin ruido.** El modelo de embeddings se carga desde la caché local de
  Hugging Face cuando ya está descargado (sin requests al Hub, y por lo tanto sin el
  warning de "unauthenticated requests"), la llamada automática a funciones se
  desactiva explícitamente en el modelo de Gemini (la cadena no pasa herramientas) y los
  logs por request de `httpx` quedan en nivel WARNING.

---

An English version of this document is available in [README.en.md](README.en.md).
