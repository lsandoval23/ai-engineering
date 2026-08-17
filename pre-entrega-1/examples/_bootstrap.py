"""Demo helpers: `import src...` from a plain script, provider filtering by
available key, and console logging that never glues onto streamed text."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas import LLMConfig  # noqa: E402
from src.settings import fallback_chain  # noqa: E402


class _ConsoleLogSink:
    """Writes log records to stderr, but first closes any partially printed
    stdout line (streamed tokens have no trailing newline, and the manager
    logs the moment a stream finishes — before the demo can print one)."""

    def __init__(self) -> None:
        self.mid_line = False

    def __call__(self, message) -> None:
        if self.mid_line:
            sys.stderr.write("\n")
            self.mid_line = False
        sys.stderr.write(str(message))


_sink = _ConsoleLogSink()
logger.remove()
logger.add(
    _sink,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
)


def stream_print(text: str) -> None:
    """Print a stream chunk without a newline, remembering the open line."""
    print(text, end="", flush=True)
    _sink.mid_line = True


def stream_done() -> None:
    """Close the streamed line."""
    print("\n")
    _sink.mid_line = False


def available_configs(**overrides) -> list[LLMConfig]:
    """Configs from config.yaml's fallback_order, skipping providers without a key."""
    configs = [c for c in fallback_chain(**overrides) if c.api_key is not None]
    if not configs:
        sys.exit(
            "No API keys found. Copy .env.example to .env and fill in at least "
            "one key (GOOGLE_API_KEY has a free tier at aistudio.google.com/apikey)."
        )
    return configs
