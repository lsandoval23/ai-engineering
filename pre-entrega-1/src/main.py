"""Single entry point for every runnable example of the async LLM client.

    python -m src.main normal       # one blocking call, latency reported
    python -m src.main streaming    # token-by-token + TTFT / throughput
    python -m src.main fallback     # a failing primary, failover measured
    python -m src.main gather       # sequential vs asyncio.gather
    python -m src.main all          # the four, in order

This module is the *presentation* layer: it prints, times and simulates
failures, but every decision (retry, fallback, timeout, semaphore) stays in
``AsyncLLMManager``. Loguru is configured here and not in ``src/`` on purpose —
a library should not install logging side effects on import.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from time import perf_counter
from typing import AsyncGenerator, List, Optional, Sequence, Tuple

# The documented invocation is `python -m src.main`, which puts the repo root
# on sys.path. This guard keeps a bare `python src/main.py` working too.
if __package__ in (None, ""):  # pragma: no cover — only hit by direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from src.base import BaseLLMClient  # noqa: E402
from src.manager import AsyncLLMManager  # noqa: E402
from src.metrics import elapsed_ms  # noqa: E402
from src.schemas import ChatMessage, ErrorKind, LLMConfig, ModelResponse, Provider  # noqa: E402
from src.settings import CONFIG_KEY_FIELDS, fallback_chain  # noqa: E402

MODES = ("normal", "streaming", "fallback", "gather", "all")

DEFAULT_PROMPT = "¿Qué es la entropía? Respondé en 2 líneas."
SYSTEM_PROMPT = "Answer in Spanish, concisely."

GATHER_PROMPTS = [
    "Name one planet of the solar system.",
    "Say a prime number under 20.",
    "Name a chemical element.",
    "Say a color.",
    "Name a programming language.",
]


# --------------------------------------------------------------------------- #
# Console plumbing
# --------------------------------------------------------------------------- #
class _ConsoleLogSink:
    """Writes log records to stderr, but first closes any partially printed
    stdout line (streamed tokens have no trailing newline, and the manager
    logs the moment a stream finishes — before the example can print one)."""

    def __init__(self) -> None:
        self.mid_line = False

    def __call__(self, message) -> None:
        if self.mid_line:
            sys.stderr.write("\n")
            self.mid_line = False
        sys.stderr.write(str(message))


_sink = _ConsoleLogSink()


def configure_logging(quiet: bool) -> None:
    """Replace Loguru's default sink with ours, or silence it entirely."""
    logger.remove()
    if not quiet:
        logger.add(
            _sink,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
        )


def stream_print(text: str) -> None:
    """Print a stream chunk without a newline, remembering the open line."""
    print(text, end="", flush=True)
    _sink.mid_line = True


def stream_done() -> None:
    """Close the streamed line."""
    print("\n")
    _sink.mid_line = False


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------- #
# Configs & manager
# --------------------------------------------------------------------------- #
def available_configs(**overrides) -> List[LLMConfig]:
    """Configs from config.yaml's fallback_order, skipping providers without a key."""
    configs = [c for c in fallback_chain(**overrides) if c.api_key is not None]
    if not configs:
        sys.exit(
            "No API keys found. Copy .env.example to .env and fill in at least "
            "one key (GOOGLE_API_KEY has a free tier at aistudio.google.com/apikey)."
        )
    return configs


def build_manager(**overrides) -> Tuple[AsyncLLMManager, List[LLMConfig]]:
    """The two lines every example needs: usable configs + a manager over them."""
    configs = available_configs(**overrides)
    return AsyncLLMManager(configs[0], fallbacks=configs[1:]), configs


def user_messages(prompt: str, *, system: Optional[str] = SYSTEM_PROMPT) -> List[ChatMessage]:
    messages = [ChatMessage(role="user", content=prompt)]
    if system:
        messages.insert(0, ChatMessage(role="system", content=system))
    return messages


def describe_chain(configs: Sequence[LLMConfig]) -> str:
    return " -> ".join(f"{c.provider.value}:{c.model}" for c in configs)


# --------------------------------------------------------------------------- #
# Failure simulation (fallback mode)
# --------------------------------------------------------------------------- #
class AlwaysRateLimitedClient(BaseLLMClient):
    """A fake provider that answers every call with a 429.

    Rate limit is a *retryable* kind, so this exercises the full resilience
    path — ``max_attempts`` tries with exponential backoff + jitter — before
    the manager gives up on it and falls back. No credits, no network: exactly
    what step 5 of the brief ("simulá un error de rate limit") asks for.
    """

    def __init__(self, provider: Provider, model: str) -> None:
        self.provider = provider
        self.model = f"{model} (simulated 429)"

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        return self._error_response(ErrorKind.RATE_LIMIT, "429 Too Many Requests (simulated)")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        return
        yield  # pragma: no cover — marks this method as an async generator


def break_api_key(config: LLMConfig) -> LLMConfig:
    """Copy of ``config`` whose provider key is invalid on purpose.

    Reuses settings' Provider -> field map instead of redeclaring it, so adding
    a provider stays a one-place change.
    """
    return config.model_copy(
        update={CONFIG_KEY_FIELDS[config.provider]: SecretStr("invalid-key-on-purpose")}
    )


# --------------------------------------------------------------------------- #
# The examples
# --------------------------------------------------------------------------- #
async def run_normal(args: argparse.Namespace) -> None:
    """One complete (non-streaming) call: the whole answer arrives at once."""
    section("NORMAL — await manager.generate()")
    manager, configs = build_manager(max_tokens=args.max_tokens or 200)
    print(f"Chain: {describe_chain(configs)}\n")

    response = await manager.generate(user_messages(args.prompt or DEFAULT_PROMPT))

    if response.ok:
        print(f"Answered by: {response.provider.value} ({response.model})")
        print(f"\n{response.content}\n")
        print(f"Total latency: {response.total_latency_ms:.0f} ms")
    else:
        print(f"All providers failed: {response.error}")


async def run_streaming(args: argparse.Namespace) -> None:
    """The same request, streamed: the first token lands in TTFT, not in total."""
    section("STREAMING — async for chunk in manager.generate_stream()")
    manager, configs = build_manager(max_tokens=args.max_tokens or 200)
    print(f"Chain: {describe_chain(configs)}")
    print(f"Streaming from {configs[0].provider.value} ({configs[0].model}):\n")

    async for chunk in manager.generate_stream(user_messages(args.prompt or DEFAULT_PROMPT)):
        stream_print(chunk)
    stream_done()

    metrics = manager.last_stream_metrics
    if metrics is not None:
        print(f"Metrics: {metrics.summary()}")


async def run_fallback(args: argparse.Namespace) -> None:
    """A primary that cannot answer, and the failover that saves the request."""
    section(f"FALLBACK — simulated failure: {args.simulate}")
    configs = available_configs(max_tokens=args.max_tokens or 200)

    if args.simulate == "bad-key":
        # An invalid key is non-retryable (401 auth, or 400 on Gemini), so the
        # switch is immediate — no backoff delays in the way.
        # The healthy primary re-enters as the first fallback, so the failover
        # is demonstrable even when only one provider has a key configured.
        primary = break_api_key(configs[0])
        manager = AsyncLLMManager(primary, fallbacks=configs)
        print(f"Primary: {primary.provider.value} (API key broken on purpose)")
        print(f"Fallbacks: {describe_chain(configs)}\n")
    else:
        # Rate limit is retryable: watch max_attempts backoff delays go by
        # before the manager gives up on the fake provider and switches.
        manager = AsyncLLMManager(configs[0], fallbacks=configs[1:])
        fake = AlwaysRateLimitedClient(configs[0].provider, configs[0].model)
        manager._chain.insert(0, (configs[0], fake))
        print(f"Primary: {configs[0].provider.value} (always returns 429)")
        print(f"Attempts before failover: {configs[0].max_attempts}")
        print(f"Fallbacks: {describe_chain(configs)}\n")

    t0 = perf_counter()
    response = await manager.generate(
        user_messages("Say 'hello' in Spanish, one word.", system=None)
    )
    elapsed = elapsed_ms(t0)

    if response.ok:
        print(f"Answered by: {response.provider.value} ({response.model})")
        print(f"Content: {response.content.strip()}")
        print(f"Total time including the failover: {elapsed:.0f} ms")
    else:
        print(f"All providers failed: {response.error}")


async def run_gather(args: argparse.Namespace) -> None:
    """The same N calls one after another, then all at once."""
    section("GATHER — sequential vs asyncio.gather")
    manager, configs = build_manager(max_tokens=args.max_tokens or 20)
    prompts = [args.prompt] * len(GATHER_PROMPTS) if args.prompt else GATHER_PROMPTS

    def messages(prompt: str) -> List[ChatMessage]:
        return user_messages(f"{prompt} Answer with one word only.", system=None)

    print(f"Running {len(prompts)} prompts against {configs[0].provider.value}...\n")

    t0 = perf_counter()
    sequential = [await manager.generate(messages(p)) for p in prompts]
    t_seq = elapsed_ms(t0)

    t0 = perf_counter()
    concurrent = await asyncio.gather(*(manager.generate(messages(p)) for p in prompts))
    t_gather = elapsed_ms(t0)

    for prompt, response in zip(prompts, concurrent):  # gather preserves launch order
        answer = response.content.strip() if response.ok else f"error: {response.error}"
        print(f"  {prompt:45s} -> {answer}")

    errors = sum(1 for r in [*sequential, *concurrent] if not r.ok)
    print(f"\nSequential: {t_seq / 1000:.2f} s")
    print(f"Concurrent (asyncio.gather): {t_gather / 1000:.2f} s")
    print(f"Speedup: {t_seq / t_gather:.1f}x  ({errors} errors)")


RUNNERS = {
    "normal": run_normal,
    "streaming": run_streaming,
    "fallback": run_fallback,
    "gather": run_gather,
}


async def run_all(args: argparse.Namespace) -> None:
    for name in ("normal", "streaming", "fallback", "gather"):
        await RUNNERS[name](args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Runnable examples of the async, resilient, provider-agnostic LLM client.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  normal      one complete call; prints the answer and total latency\n"
            "  streaming   token-by-token output plus TTFT / total / throughput\n"
            "  fallback    a primary that fails, and the automatic failover\n"
            "  gather      the same 5 prompts sequentially, then with asyncio.gather\n"
            "  all         runs the four above, in order\n"
        ),
    )
    parser.add_argument("mode", choices=MODES, help="which example to run")
    parser.add_argument("--prompt", help="override the user prompt (default depends on the mode)")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="override max_tokens (default: 200, or 20 in gather mode)",
    )
    parser.add_argument(
        "--simulate",
        choices=("bad-key", "rate-limit"),
        default="bad-key",
        help="fallback mode only: which failure to simulate on the primary "
        "(bad-key = non-retryable, immediate switch; rate-limit = retryable, so "
        "the backoff attempts happen first). Default: bad-key",
    )
    parser.add_argument("--quiet", action="store_true", help="silence the structured log output")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.quiet)
    runner = run_all if args.mode == "all" else RUNNERS[args.mode]
    asyncio.run(runner(args))


if __name__ == "__main__":
    main()
