# Pre-Entrega 2 — Validated processing pipeline

A **technical-entity extraction pipeline** built with LangChain and LCEL: it takes a
paragraph of free text (an error log, an architecture description) and **always**
returns a validated Pydantic object — or raises after exhausting retries and the
fallback. It never returns raw text, an unvalidated `dict`, or a silent default.

```
{"text": "..."} ──▶ prompt | model.with_structured_output(TechnicalEntities) | check
                    └─ .with_retry(exponential backoff + jitter)
                       └─ error-aware retry (validation feedback)
                          └─ .with_fallbacks([lighter model])
                                                          ──▶ TechnicalEntities
```

All code (identifiers, logs, docstrings) is in English. Names the assignment spells in
Spanish are translated; the mapping is in [How each requirement is met](#how-each-requirement-is-met).

## Install

Requires **Python 3.12**.

```bash
# from this folder
py -3.12 -m venv .venv          # Windows (Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate          # (source .venv/bin/activate)
pip install -r requirements.txt

cp .env.example .env            # then fill in your API key
```

Dependencies are **pinned to exact versions** in `requirements.txt` (pin date included):
LangChain moves fast and the course pins nothing.

## Configuration

- **`.env`** — secrets only, gitignored. **One** key is enough: the provider you run
  with. `GOOGLE_API_KEY` has a free tier at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and is the default
  provider. `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are optional.
- **[`config.yaml`](config.yaml)** — everything else: the default provider
  (`default_provider`), the **model IDs** per provider (`models`), each one's fallback
  model (`fallback_models`), and under `defaults` the temperature, per-call timeout,
  attempt count and backoff parameters.

Nothing tunable is hardcoded. The YAML is parsed at startup into a validated Pydantic
model (`AppSettings`, in `src/settings.py`), so a malformed file — an unknown provider,
`max_attempts: 0`, a backoff with `max < initial`, a provider without a model — **fails
fast with a clear error** instead of breaking mid-request. Removing a provider's entry
from `fallback_models` runs it without a fallback. The `EXTRACTION_PIPELINE_CONFIG`
environment variable points at an alternative YAML (the tests use it to prove behaviour
comes from the file).

| Provider | Primary model | Fallback (lighter) |
|---|---|---|
| `gemini` (default) | `gemini-3.5-flash` | `gemini-flash-lite-latest` |
| `openai` | `gpt-4o-mini` | `gpt-4.1-mini` |
| `anthropic` | `claude-haiku-4-5` | `claude-sonnet-4-6` |

All three models are built with `temperature=0` (extraction wants determinism, not
creativity), `timeout=30 s` (both from `config.yaml`) and `max_retries=0`: the SDKs' internal retries are switched
off so that `.with_retry()` is **the single retry layer**, and therefore visible in the
logs.

## Run

```bash
python -m src.main                    # default provider: gemini
python -m src.main --provider openai  # or anthropic
python -m src.main --text "..."       # a custom text instead of the two built-in cases
python -m src.main --log-level WARNING
```

`src/main.py` is the assignment's **async test mini-script**: under `asyncio.run()` it runs
two cases through `process_text()` — a clearly technical text and the stress test with
an ambiguous one — and prints each result with `model_dump_json(indent=2)`. Each case is
isolated in its own `try/except` so one failure cannot hide the other; the exit code is
1 if any case failed.

Real output (`python -m src.main`, 2026-09-07, Gemini):

```
INFO    extraction_pipeline: [gemini] Processing text (242 characters)
INFO    httpx: HTTP Request: POST .../models/gemini-3.5-flash:generateContent "HTTP/1.1 200 OK"
INFO    extraction_pipeline: Model responded: finish_reason=STOP input_tokens=119 output_tokens=581
INFO    extraction_pipeline: [gemini] Validated extraction: {'technologies': ['FastAPI', 'Redis', 'PostgreSQL'], 'criticality_level': 'high', ...}

=== Clear technical text ===
Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché en Redis parece
saturarse en picos de tráfico y las conexiones a PostgreSQL se agotan porque el pool está
mal dimensionado. Esto está afectando a usuarios en producción.

{
  "technologies": ["FastAPI", "Redis", "PostgreSQL"],
  "criticality_level": "high",
  "technical_summary": "La API desarrollada con FastAPI presenta timeouts intermitentes en producción debido a la saturación del caché en Redis y al agotamiento de las conexiones en el pool de PostgreSQL."
}
```

Without a key the script fails **with a message, not a traceback**:

```
FAILED: ValueError: OPENAI_API_KEY is not set; add it to .env (see .env.example)
```

## How each requirement is met

Mapping between the assignment's names and the code:

| Assignment (Spanish) | Code (English) | Where |
|---|---|---|
| `EntidadesTecnicas` | `TechnicalEntities` | `src/schemas.py` |
| `NivelCriticidad` · `baja` / `media` / `alta` | `CriticalityLevel` · `low` / `medium` / `high` | `src/schemas.py` |
| `tecnologias` | `technologies` | `src/schemas.py` |
| `nivel_de_criticidad` | `criticality_level` | `src/schemas.py` |
| `resumen_tecnico` | `technical_summary` | `src/schemas.py` |
| prompt variable `{texto}` | `{text}` (invocation key `"text"`) | `src/chain.py` |
| `schemas.py`, `chain.py`, `process_text()` | same names, inside `src/` | `src/` |

### Band 1 — Schema definition and structured output (30 %)

- `src/schemas.py`: `TechnicalEntities` with exactly the three required fields —
  `technologies: List[str]` (`min_length=1`), `criticality_level: CriticalityLevel`
  (closed three-value enum, `class CriticalityLevel(str, Enum)`) and
  `technical_summary: str` (`min_length=10`). Every field carries a `description=`: with
  `with_structured_output` those descriptions are what the provider shows the model,
  i.e. *prompt engineering embedded in the contract*.
- `@field_validator("technologies")`: the **semantic** layer. `min_length=1` would accept
  `["  ", ""]`; the validator strips, drops blanks, deduplicates preserving order and
  raises `ValueError` if nothing survives.
- `src/chain.py`, `_resilient_extraction()`:
  `prompt | model.with_structured_output(TechnicalEntities, include_raw=True) | check`.
  The chain returns an **instance** of `TechnicalEntities`, never an `AIMessage`.
- Evidence: `tests/test_schemas.py` (10 tests) proves what the schema rejects — empty
  list, blank entries, `"HIGH"`/`"alta"` outside the enum, short summary, missing
  field — and `test_happy_path_returns_a_validated_instance` checks the chain's return
  type.

### Band 2 — LCEL chain and prompting (30 %)

- The main flow is composed with the `|` operator (`src/chain.py`,
  `_resilient_extraction()`): `prompt | structured_model | RunnableLambda(check)`.
  There is no imperative "call then parse" sequence anywhere.
- `prompt` is a `ChatPromptTemplate.from_messages` with separate roles: `("system", …)`
  sets the role (technical analyst), the goal and the task breakdown;
  `("human", "{text}")` carries **only** the data. A trailing
  `MessagesPlaceholder("feedback", optional=True)` is the hook for the error-aware
  retry; ordinary calls pass just `{"text": ...}`.
- No f-strings: `tests/test_prompt.py` checks that the input variable is exactly
  `{"text"}`, that a `SystemMessage` and a `HumanMessage` are rendered, and that braces
  inside user text survive (substitution, not interpolation).

### Band 3 — Async execution and resilience (25 %)

- `process_text(text, provider)` is `async` and runs `await chain.ainvoke({"text": text})`
  (`src/chain.py`). It logs at INFO before the call (provider, length), at INFO the validated
  `model_dump()` on success, and at ERROR **re-raising** when everything failed. There is
  no third outcome.
- `.with_retry(stop_after_attempt=3, wait_exponential_jitter=True, ...)` wraps the
  composed chain. Beyond transient errors it covers **incomplete JSON**: the `check` step
  reads the raw response's `finish_reason` (`include_raw=True`) and raises
  `TruncatedResponseError` if the provider cut the output on tokens (`length` /
  `max_tokens` / `MAX_TOKENS`) — the assignment's "error to avoid", detected explicitly
  rather than only bouncing off validation.
- The logs expose the whole flow: every failed attempt (`Model call failed with …`),
  every retry (`Retry attempt 2/3 …`), the model switch (`switching to the fallback
  model`), the `finish_reason` and token usage of each response, and the feedback retry.
- Evidence: `tests/test_chain.py` (12 tests) covers retry on 429/timeout, truncation,
  error-aware retry, fallback, no-retry on 401, and that `process_text` logs and
  re-raises.

### Band 4 — Repository structure and test script (15 %)

```
pre-entrega-2/
├── src/
│   ├── schemas.py      # data contract: CriticalityLevel, TechnicalEntities
│   ├── chain.py        # get_model(), prompt, compose_chain(), build_chain(), process_text()
│   ├── settings.py     # config.yaml -> AppSettings (Pydantic); keys from .env
│   └── main.py         # async mini-script: asyncio.run() over both cases
├── config.yaml         # default provider, models, fallbacks, timeout, retries
├── requirements.txt    # exact versions
├── pytest.ini
├── .env.example        # empty placeholders; .env is gitignored
├── .gitignore
├── README.md · README.en.md
└── tests/              # 35 tests, all offline
    ├── conftest.py     # ScriptedChatModel (scripted fake model) + temporary config.yaml
    ├── test_schemas.py
    ├── test_prompt.py
    ├── test_chain.py
    └── test_settings.py
```

## Resilience: what fails and who handles it

The course's error taxonomy (transient / format / permanent) maps one-to-one onto the
chain's layers. The exception classes are **langchain-core's normalized ones**
(`ModelRateLimitError`, `ModelTimeoutError`, …): all three provider integrations
translate their SDK errors into them, so the policy is provider-agnostic and imports no
SDK.

| Failure | Example | Handled by | What happens |
|---|---|---|---|
| **Transient** | 429, timeout, connection, 5xx, truncated response | `.with_retry()` | up to 3 attempts, wait `min(1·2ⁿ + jitter, 10) s` |
| **Format** | the object fails validation (`ValidationError`, `OutputParserException`) | error-aware retry | **one** re-invocation with Pydantic's error text as an extra message: the model learns *which* field failed |
| **Model down** | the primary exhausted its layers | `.with_fallbacks()` | the same full stack on a lighter model of the same provider |
| **Permanent** | 401, 400, unknown model | none | one attempt, fallback, then the exception reaches the caller with its message |

The error-aware retry is not a blind retry, and it sits *inside* the fallback on
purpose: on a validation error the model first gets a chance to correct itself (class
slide 38); only when a model fails repeatedly do we switch models (slide 22).

This is not theoretical. On the first real run of 2026-09-07 Gemini's primary model was
saturated and the whole stack exercised itself:

```
WARNING extraction_pipeline: Model call failed with GoogleAPIError: 503 UNAVAILABLE. {... 'This model is currently experiencing high demand ...'}
WARNING extraction_pipeline: Retry attempt 2/3 after exponential backoff
WARNING extraction_pipeline: Model call failed with GoogleAPIError: 503 UNAVAILABLE. ...
WARNING extraction_pipeline: Retry attempt 3/3 after exponential backoff
WARNING extraction_pipeline: Model call failed with GoogleAPIError: 503 UNAVAILABLE. ...
WARNING extraction_pipeline: Primary model failed with GoogleAPIError: 503 UNAVAILABLE. ...; switching to the fallback model
INFO    httpx: HTTP Request: POST .../models/gemini-flash-lite-latest:generateContent "HTTP/1.1 200 OK"
INFO    extraction_pipeline: Model responded: finish_reason=STOP input_tokens=119 output_tokens=84
INFO    extraction_pipeline: [gemini] Validated extraction: {'technologies': ['FastAPI', 'Redis', 'PostgreSQL'], 'criticality_level': 'high', ...}
```

Three attempts with backoff, a switch to the fallback, a validated object. The caller
never noticed.

## Stress test: ambiguous text

Input: *"El sistema anda medio raro últimamente, no sé bien qué está pasando."* ("The
system has been acting weird lately, not sure what's going on.") — it names no
technology at all.

**Does the validator raise, or does the model recover?** The model recovers, by making
something up. Across three real runs (two on the `gemini-flash-lite-latest` fallback
during the outage, one on `gemini-3.5-flash`):

| Run | `technologies` | `criticality_level` |
|---|---|---|
| 1 | `["sistema"]` | `low` |
| 2 | `["Sistema"]` | `medium` |
| 3 | `["Sistema"]` | `low` |

Observations:

- The schema demands `min_length=1` on `technologies`, and the model satisfies it with a
  **placeholder** (`"sistema"`, the only technical-looking word in the text). The
  contract is honored *to the letter* and dodged *in spirit*: structural validation
  guarantees **shape, not truth**. This is the course's "a floor, not a ceiling".
- Criticality is not stable across runs (`low` / `medium`) even at `temperature=0`: the
  text gives no basis to decide it, so the model picks.
- The summary is honest in all three runs ("without specifying components, tools or
  concrete errors"): the free-text field reflects the ambiguity better than the closed
  list does.

Making this case *fail* instead of pass would take a cross-field semantic rule (a
`model_validator` rejecting placeholders, or requiring real technologies when
criticality is high). That is outside the assignment's scope and is documented here as a
known limit.

## Tests

```bash
python -m pytest -v
```

35 tests, all **offline**: `tests/conftest.py` defines `ScriptedChatModel`, a fake
`BaseChatModel` that replays a script (an `AIMessage` with a tool call is returned; an
exception is raised) and records every message list it received. This exercises the real
chain, with the real `with_structured_output`, with no network and no keys.

- `test_schemas.py` (10) — what the contract accepts and rejects; the semantic validator.
- `test_prompt.py` (4) — exact `text` variable, roles, feedback placeholder, not an f-string.
- `test_settings.py` (8) — the shipped `config.yaml` is valid; an alternative YAML (via
  `EXTRACTION_PIPELINE_CONFIG`) changes the default provider and models; fallback is
  optional per provider; a malformed file fails at startup with `ValidationError`
  (unknown provider, `max_attempts: 0`, temperature out of range, incoherent backoff,
  provider without a model).
- `test_chain.py` (13) — validated instance on the happy path; 429 + timeout retried
  with backoff and logged; `finish_reason=length` detected and retried; validation
  error → a second call carrying Pydantic's message with the field name; a second
  validation failure propagates; fallback after exhausting retries (primary called
  exactly 3 times); 401 **not** retried (1 call); `process_text` logs INFO and returns /
  logs ERROR and re-raises; unknown provider and missing key fail with a clear message;
  the model ID, timeout and temperature come from `config.yaml` and `max_retries=0`;
  `build_chain()` with no argument uses the file's `default_provider` and fallback.

## Decisions and scope

- **Default provider: Gemini.** The assignment names `ChatOpenAI` or `ChatAnthropic`; all
  three are implemented and interchangeable through `--provider`, but the default is the
  one with a free tier. Gemini IDs were verified against the API on 2026-09-07:
  `gemini-flash-latest` returned sustained 503s, `gemini-3.6-flash` (the course
  notebook's) works but **ignores `temperature`** with a warning, `gemini-2.5-flash` is
  no longer available; `gemini-3.5-flash` honors `temperature=0` and answers in ~4 s.
  The OpenAI/Anthropic IDs could not be exercised without a key.
- **The `langchain` metapackage is not in `requirements.txt`**: nothing imports it.
  Everything comes from `langchain-core` and the three integration packages.
- **Deliberately not implemented** (outside the assignment's scope): a cross-field
  `model_validator`, LangSmith, caching, JSON logging, TTFT/cost metrics.
- **Without OpenAI/Anthropic keys** those paths are covered by the offline tests and by
  the normalized exception-type check, not by a real run.

---

La versión en español de este documento está en [README.md](README.md).
