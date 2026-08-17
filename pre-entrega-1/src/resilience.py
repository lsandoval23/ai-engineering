"""Retry with exponential backoff + jitter.

Only transient failures are retried: rate limits (429), connection problems,
timeouts and 5xx server errors. Auth (401) and bad-request (400) errors fail
identically on every attempt, so retrying them just burns time — they return
immediately and let the fallback layer decide.
"""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable

from loguru import logger

from src.schemas import ErrorKind, ModelResponse

RETRYABLE_KINDS = frozenset(
    {ErrorKind.RATE_LIMIT, ErrorKind.CONNECTION, ErrorKind.TIMEOUT, ErrorKind.SERVER}
)


def backoff_delay(
    attempt: int, *, base_delay_s: float, jitter_min: float, jitter_max: float
) -> float:
    """``base * 2^attempt``, scaled by random jitter.

    Jitter avoids synchronized retry waves: without it, every client that
    failed at the same moment retries at exactly the same moments too,
    hammering a service that is already struggling.
    """
    return base_delay_s * (2**attempt) * random.uniform(jitter_min, jitter_max)


async def with_retry(
    call: Callable[[], Awaitable[ModelResponse]],
    *,
    max_attempts: int,
    base_delay_s: float,
    jitter_min: float,
    jitter_max: float,
) -> ModelResponse:
    """Run ``call`` up to ``max_attempts`` times, backing off between attempts.

    ``call`` must return a ModelResponse (clients never raise), so success is
    ``response.error is None`` and retryability comes from ``error_kind``.
    """
    last: ModelResponse | None = None
    for attempt in range(max_attempts):
        response = await call()
        if response.error is None:
            if attempt:
                logger.info(
                    "{provider} succeeded on attempt {n}/{m}",
                    provider=response.provider.value,
                    n=attempt + 1,
                    m=max_attempts,
                )
            return response

        last = response
        if response.error_kind not in RETRYABLE_KINDS:
            logger.warning(
                "{provider} failed with non-retryable {kind}; not retrying: {error}",
                provider=response.provider.value,
                kind=response.error_kind.value if response.error_kind else "unknown",
                error=response.error,
            )
            return response

        if attempt + 1 < max_attempts:
            delay = backoff_delay(
                attempt,
                base_delay_s=base_delay_s,
                jitter_min=jitter_min,
                jitter_max=jitter_max,
            )
            logger.warning(
                "{provider} attempt {n}/{m} failed ({kind}); retrying in {delay:.2f}s",
                provider=response.provider.value,
                n=attempt + 1,
                m=max_attempts,
                kind=response.error_kind.value,
                delay=delay,
            )
            await asyncio.sleep(delay)

    logger.error(
        "{provider} exhausted all {m} attempts: {error}",
        provider=last.provider.value,
        m=max_attempts,
        error=last.error,
    )
    return last
