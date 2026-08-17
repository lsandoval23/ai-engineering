"""Latency metrics: TTFT, total latency and throughput.

All three are ``time.perf_counter()`` arithmetic. Throughput counts stream
chunks per second, which is a proxy for tokens/s (a chunk usually carries one
or a few tokens); the README documents this approximation.
"""

from __future__ import annotations

from time import perf_counter
from typing import Optional


def elapsed_ms(t0: float) -> float:
    """Milliseconds since ``t0`` (a perf_counter timestamp)."""
    return (perf_counter() - t0) * 1000


class StreamMetrics:
    """Collects timing over one streaming call.

    Usage: create right before starting the stream, call :meth:`mark_chunk`
    for every chunk, :meth:`finish` when the stream ends.
    """

    def __init__(self) -> None:
        self._t0 = perf_counter()
        self._first: Optional[float] = None
        self._end: Optional[float] = None
        self.chunks = 0

    def mark_chunk(self) -> None:
        if self._first is None:
            self._first = perf_counter()
        self.chunks += 1

    def finish(self) -> None:
        self._end = perf_counter()

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time To First Token — the number users actually feel."""
        if self._first is None:
            return None
        return (self._first - self._t0) * 1000

    @property
    def total_ms(self) -> Optional[float]:
        if self._end is None:
            return None
        return (self._end - self._t0) * 1000

    @property
    def chunks_per_second(self) -> Optional[float]:
        """Chunk throughput after the first token (proxy for tokens/s)."""
        if self._first is None or self._end is None:
            return None
        generation_time = self._end - self._first
        if generation_time <= 0:
            return None
        return self.chunks / generation_time

    def summary(self) -> str:
        ttft = f"{self.ttft_ms:.0f} ms" if self.ttft_ms is not None else "n/a"
        total = f"{self.total_ms:.0f} ms" if self.total_ms is not None else "n/a"
        rate = (
            f"{self.chunks_per_second:.1f} chunks/s"
            if self.chunks_per_second is not None
            else "n/a"
        )
        return f"TTFT={ttft} · total={total} · {self.chunks} chunks · {rate}"
