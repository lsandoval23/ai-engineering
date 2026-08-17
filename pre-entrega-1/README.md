# Pre-Entrega 1 — Robust Async LLM Client

A provider-agnostic, asynchronous and resilient LLM client: the same two calls
(`generate` / `generate_stream`) work against OpenAI, Anthropic or Gemini, and
the manager survives rate limits, timeouts and provider outages by retrying
with exponential backoff and falling back to the next provider automatically.
Built for the Coderhouse AI Engineering course (Clase 1).

## Setup

Requires **Python 3.12**.

```bash
# from this folder
py -3.12 -m venv .venv          # Windows (on Linux/macOS: python3.12 -m venv .venv)
.venv\Scripts\activate          # (source .venv/bin/activate)
pip install -r requirements.txt

cp .env.example .env            # then fill in your API keys
```

`.env` holds **secrets only** (it is gitignored). At least one key is enough —
`GOOGLE_API_KEY` has a free tier at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Everything else — default temperature, token caps, retry policy, timeouts,
semaphore size, **model IDs** and the fallback order — lives in
[`config.yaml`](config.yaml). Nothing tunable is hardcoded: the file is parsed
into a validated Pydantic `AppSettings` model at startup (`src/settings.py`),
so a malformed config fails fast with a clear error, and even the *validation
boundaries* (e.g. the accepted temperature range) come from the config file.
Model IDs being config rather than code is deliberate: catalogs move faster
than course material (the hint notebook shipped `claude-3-5-sonnet-20241022`,
retired in Oct 2025).

### Run the demos

```bash
python examples/demo_streaming.py   # token-by-token streaming + TTFT metrics
python examples/demo_fallback.py    # broken primary key -> automatic failover
python examples/demo_gather.py      # sequential vs asyncio.gather benchmark
```

## Architecture

```
business logic ──▶ AsyncLLMManager ──▶ BaseLLMClient (ABC)
                   factory · semaphore      ├── OpenAIClient
                   timeout · retry          ├── AnthropicClient
                   fallback · metrics       └── GeminiClient
```

- **`BaseLLMClient` (ABC)** — the contract: `generate()` returning a validated
  `ModelResponse`, and `generate_stream()` as an async generator. A subclass
  that forgets a method fails at instantiation, not deep inside a request.
- **Factory** — `AsyncLLMManager._create_client()` is the single place that
  maps `Provider -> concrete class`. Swapping providers is a config change
  (interoperability), and business code only ever holds the abstraction.
- **What the abstraction absorbs** — the three SDKs disagree on almost
  everything: method names, where the text lives, streaming idioms, and how
  the `system` role is passed (OpenAI: in `messages`; Anthropic: a top-level
  `system=` parameter; Gemini: `system_instruction` in the config). Each
  client normalizes its provider's dialect behind the same interface.
- **Result objects, not exceptions** — clients never let provider exceptions
  escape: every failure becomes `ModelResponse(error=..., error_kind=...)`.
  The typed `error_kind` is what makes the resilience layer possible without
  string parsing. Unrecoverable *config* errors (missing key, unsupported
  provider) do the opposite and crash at startup — failing clearly beats
  failing silently. Bugs in our own code propagate: only each SDK's typed
  exception family is caught, never bare `Exception`.

## Resilience

- **Retry policy** — only transient failures are retried: rate limit (429),
  connection errors, timeouts and 5xx. Auth (401) and bad request (400) fail
  identically on every attempt, so they return immediately (retrying a 401
  five times is five identical failures).
- **Backoff formula** — `base_delay_s * 2^attempt * uniform(jitter_min, jitter_max)`,
  all four numbers from `config.yaml`. The **jitter** matters: without it,
  every client that failed at the same moment retries at the same moments
  too, hitting the recovering service in synchronized waves.
- **Fallback** — the manager walks the provider chain in `config.yaml`'s
  `fallback_order`; when a provider exhausts its attempts (or fails with a
  non-retryable error), it logs the switch and tries the next. If all fail,
  the returned `ModelResponse.error` aggregates every attempt.
- **Streaming fallback decision** — the manager falls back **only before the
  first token**. Once tokens have reached the user, restarting on another
  provider would duplicate output, so a mid-stream failure is surfaced
  in-stream instead. Clients signal a pre-first-token failure with a typed
  `StreamError` so the manager can tell the two cases apart.
- **Timeouts** — two layers: `asyncio.timeout(timeout_s)` per call in the
  manager (in streaming it bounds the wait for the *first* token), plus the
  same value passed to each SDK's client-level `timeout`.
- **Semaphore** — `asyncio.Semaphore(max_concurrent)` caps in-flight requests
  (backpressure/throttling: the other requests wait inside the process, where
  waiting is free, instead of turning into a wall of 429s).
- **Structured logging** — Loguru records which provider answered and in how
  long, every retry with its delay, every fallback with its reason, and every
  error with its kind. Keys are `SecretStr` and never logged.
- Not implemented (out of scope, natural next step): a **circuit breaker** —
  stop calling a repeatedly-failing provider for a cooldown period instead of
  re-trying it on every request.

## Measured metrics

Live run of 2026-08-15 (free-tier Gemini, `gemini-flash-lite-latest`):

| Metric | Value |
|---|---|
| TTFT (streaming, primary provider) | **1359 ms** |
| Total latency (streaming, ~40-token answer) | **1476 ms** |
| Throughput | **25.6 chunks/s** (a proxy for tokens/s, since a chunk carries one or a few tokens) |
| 5 prompts sequential | _pending_ |
| 5 prompts `asyncio.gather` | _pending_ |
| Fallback switch overhead (`demo_fallback.py`) | _pending_ |

A finding from that first run, worth more than the numbers: with
`gemini-flash-latest` the stream produced **zero text tokens** — that alias
now points to a "thinking" model that spent 196 of the 200-token budget on
internal reasoning (and the newest Gemini models reject the
`thinking_budget` parameter that used to disable it). The fix was a
config-only change (`models.gemini: gemini-flash-lite-latest`), which is the
whole argument for keeping model IDs in `config.yaml`: TTFT dropped from
6.9 s to 1.4 s without touching a line of code.

## Tests

```bash
python -m pytest tests/ -v
```

38 tests, all **offline** (fakes implementing the ABC — no API credits, no
network): schema validation, config-driven boundaries (changing a limit in a
YAML file changes what is accepted, with zero code changes), factory
dispatch, retry counts (3 attempts on a 429, exactly 1 on a 401), provider
fallback on both the normal and the streaming path, chunk ordering, and the
ABC contract itself.

## Known limitations & bugs inherited from the hint notebook (fixed here)

The course's hint notebook was a *pista*, not a finished solution. Defects
found and fixed during the port:

1. **Retired Anthropic model ID** — `claude-3-5-sonnet-20241022` (404 since
   Oct 2025) replaced by the current `claude-haiku-4-5` alias, and model IDs
   moved to `config.yaml` so the next retirement is a config edit.
2. **Anthropic `system` role mishandled** — the notebook passed system
   messages inside `messages=`, which is not Anthropic's contract (it takes a
   top-level `system=` parameter). Fixed with `_convert_messages()`, the same
   split the notebook's own `GeminiClient` already did. This was invisible in
   the notebook only because its saved run never sent a system message.
3. **`ValidationError` could escape `OpenAIClient`** — OpenAI legitimately
   returns `content=None` (tool calls, content filtering), which crashed the
   Pydantic `ModelResponse(content=...)` construction *inside* the `try` but
   *outside* the `except` ladder. Fixed both ways: `content or ""` plus
   `except ValidationError`.
4. **Gemini caught bare `Exception`** — programming errors were reported as
   "Gemini API error". Now only `google.genai.errors.APIError` subclasses are
   caught and classified by HTTP status.
5. **`getpass` for keys** — replaced with `.env` + `python-dotenv`
   (`.env` gitignored, `.env.example` committed).
6. **Top-level `await`** — works only in notebooks; every entry point here
   uses `asyncio.run(main())`.
7. **`max_tokens` deprecation on OpenAI** — the client sends
   `max_completion_tokens` (verified against the pinned `openai==3.1.0` SDK).
8. **Error swallowing as a blanket policy** — kept the result-object design
   for provider errors (it is what makes fallback trivial: the trigger is
   `error is not None`), but config errors now crash at startup and code bugs
   propagate — see *Architecture* above.

Other limitations: throughput counts chunks/s, not true tokens/s; pure
network-level failures in the Gemini SDK (below its `APIError` family) are
not classified; a mid-stream provider failure is surfaced, not resumed;
"thinking" models need care — their reasoning tokens count against
`max_tokens` and can leave zero visible text on small budgets, so the
configured Gemini model should be a non-thinking one (hence the `-lite`
model in `config.yaml`).
