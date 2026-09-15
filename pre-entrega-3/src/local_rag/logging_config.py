"""Console logging shared by the entry points, with third-party request chatter kept quiet."""

from __future__ import annotations

import logging

LOG_FORMAT: str = "%(levelname)s %(name)s: %(message)s"
# httpx logs every Hugging Face / Gemini HTTP request at INFO; only its warnings matter here.
NOISY_LOGGERS: tuple[str, ...] = ("httpx",)


def configure_logging(level: int = logging.INFO) -> None:
    """Console logging for the pipeline's own messages; noisy libraries only log warnings."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
