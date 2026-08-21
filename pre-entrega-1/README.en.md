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

### Run the examples

Everything runnable lives in a single entry point, `src/main.py`, selected by a
positional mode. Run it as a module from this folder:

```bash
python -m src.main normal      # one complete call; answer + total latency
python -m src.main streaming   # token-by-token + TTFT / total / throughput
python -m src.main fallback    # a failing primary -> automatic failover
python -m src.main gather      # 5 prompts sequentially, then with asyncio.gather
python -m src.main all         # the four above, in order
```

Common options: `--prompt TEXT`, `--max-tokens N`, `--quiet` (silences the
structured log, for clean captures), and `--simulate {bad-key,rate-limit}` for
the `fallback` mode. `python -m src.main --help` lists them.

**The `fallback` mode works with a single API key.** `--simulate bad-key` (the
default) breaks the primary's key on purpose and re-inserts the *healthy*
primary as the first fallback, so the failover is demonstrable even when only
one provider is configured. An invalid key is non-retryable (401 on
OpenAI/Anthropic, 400 `API_KEY_INVALID` on Gemini), so the switch is immediate.
`--simulate rate-limit` instead puts a fake always-429 client at the head of
the chain: rate limit *is* retryable, so this exercises the whole resilience
path — `max_attempts` tries with exponential backoff and jitter — before the
manager gives up and falls back. That one costs no credits and needs no
network, and it is exactly step 5 of the brief.

## Architecture

```
business logic ──▶ AsyncLLMManager ──▶ BaseLLMClient (ABC)
                   factory · semaphore      ├── OpenAIClient
                   timeout · retry          ├── AnthropicClient
                   fallback · metrics       └── GeminiClient

src/main.py = presentation only: prints, times, simulates failures.
```

`src/main.py` holds no policy — every decision (retry, fallback, timeout,
semaphore) lives in `AsyncLLMManager`. Loguru is configured there and not
anywhere under the library's import path on purpose: a library should not
install logging side effects when imported.

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

Live run of 2026-08-20, `python -m src.main all` (free-tier Gemini,
`gemini-flash-lite-latest`, `max_tokens=200`):

| Metric | Value |
|---|---|
| Total latency, non-streaming (`normal`) | **1678 ms** |
| TTFT (streaming, primary provider) | **1438 ms** |
| Total latency (streaming, same request) | **1593 ms** |
| Throughput | **19.3 chunks/s** (a proxy for tokens/s, since a chunk carries one or a few tokens) |
| 5 prompts sequential | **4.36 s** |
| 5 prompts `asyncio.gather` | **1.52 s** — a **2.9x** speedup, same 5 calls |
| Failover, `--simulate bad-key` (end to end) | **2606 ms** — one rejected round trip (~1.2 s) plus the successful retry; the *switch itself* is immediate, since an invalid key is non-retryable |
| Failover, `--simulate rate-limit` (end to end) | **6558 ms** — 3 attempts with backoff delays of **1.11 s** and **2.90 s** (`base 1.0 × 2^n × jitter`), then the switch |

The `gather` number is the clearest evidence the architecture is genuinely
async: the same five calls, the same total waiting, but overlapped instead of
queued. The speedup is bounded by `max_concurrent: 5` and by the provider's own
rate limiting, not by the client.

A finding from an earlier run (2026-08-15), worth more than the numbers: with
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

9 tests, all **offline** (fakes implementing the ABC — no API credits, no
network), in `tests/test_resilience.py`:

- `TestWithRetry` — a 429 retried until it succeeds, **no** retry on a 401 or a
  400 (exactly 1 call), the attempt count capped at `max_attempts`, and the
  backoff delay staying inside its jitter bounds.
- `TestFallback` — a rate-limited primary hands over to the secondary only
  after exhausting its retries, a non-retryable primary hands over on the first
  attempt, an all-failing chain aggregates every provider's error, and a
  successful response reports its total latency.

The fallback test is the highest-value one in the suite: it sits squarely in
the 40% resilience band. See *Known limitations* for what is no longer covered.

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

Other limitations:

- **Test coverage is narrower than it was.** `test_schemas.py`, `test_factory.py`
  and `test_streaming.py` were removed from the repo, so schema validation,
  config-driven boundaries, factory dispatch, chunk ordering and the ABC
  contract are no longer verified automatically. What remains is the 40%
  resilience band. Restoring them is the obvious next step.
- Throughput counts chunks/s, not true tokens/s.
- Pure network-level failures in the Gemini SDK (below its `APIError` family)
  are not classified.
- A mid-stream provider failure is surfaced, not resumed.
- "Thinking" models need care — their reasoning tokens count against
  `max_tokens` and can leave zero visible text on small budgets, so the
  configured Gemini model should be a non-thinking one (hence the `-lite` model
  in `config.yaml`).

---

Este documento también está disponible en español: [README.md](README.md).
