"""Async test mini-script for the extraction pipeline.

Runs two cases through ``process_text``: a clearly technical text and an
ambiguous, non-technical "stress test". Each case is isolated so one failure
does not hide the other; the exit code is 1 if any case failed.

    python -m src.main                       # default provider (config.yaml)
    python -m src.main --provider openai
    python -m src.main --text "..."          # a single custom text instead

This module is the *presentation* layer: it prints and configures logging.
Every decision (retry, fallback, timeout, model IDs) lives in ``src/chain.py``
and ``config.yaml``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.chain import process_text
from src.settings import PROVIDERS, get_settings

CLEAR_TEXT = (
    "Nuestra API en FastAPI está devolviendo timeouts intermitentes. El caché en Redis "
    "parece saturarse en picos de tráfico y las conexiones a PostgreSQL se agotan porque "
    "el pool está mal dimensionado. Esto está afectando a usuarios en producción."
)
AMBIGUOUS_TEXT = "El sistema anda medio raro últimamente, no sé bien qué está pasando."

CASES = [
    ("Clear technical text", CLEAR_TEXT),
    ("Stress test: ambiguous, non-technical text", AMBIGUOUS_TEXT),
]


async def run_case(title: str, text: str, provider: str) -> bool:
    print(f"\n=== {title} ===\n{text}\n")
    try:
        result = await process_text(text, provider=provider)
    except Exception as error:  # reported, not swallowed: the exit code reflects it
        print(f"FAILED: {type(error).__name__}: {error}")
        return False
    print(result.model_dump_json(indent=2))
    return True


async def main(provider: str, custom_text: str | None) -> int:
    cases = [("Custom text", custom_text)] if custom_text else CASES
    outcomes = [await run_case(title, text, provider) for title, text in cases]
    return 0 if all(outcomes) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=None,
        help="LLM provider (default: default_provider in config.yaml)",
    )
    parser.add_argument("--text", help="run a single custom text instead of the two built-in cases")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    # The Gemini SDK warns about its own "automatic function calling" mode on every
    # call; it is unrelated to this pipeline. httpx stays: it shows which model answered.
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    # Windows consoles often default to a legacy code page; the inputs are Spanish.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    provider = args.provider or get_settings().default_provider
    sys.exit(asyncio.run(main(provider, args.text)))
