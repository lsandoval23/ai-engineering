"""Streams a response token by token and prints the latency metrics.

Run from the repo root:  python examples/demo_streaming.py
"""

from __future__ import annotations

import asyncio

from _bootstrap import available_configs, stream_done, stream_print

from src.manager import AsyncLLMManager
from src.schemas import ChatMessage


async def main() -> None:
    configs = available_configs(max_tokens=200)
    manager = AsyncLLMManager(configs[0], fallbacks=configs[1:])

    messages = [
        ChatMessage(role="system", content="Answer in Spanish, concisely."),
        ChatMessage(role="user", content="¿Qué es la entropía? Respondé en 2 líneas."),
    ]

    print(f"Streaming from {configs[0].provider.value} ({configs[0].model}):\n")
    async for chunk in manager.generate_stream(messages):
        stream_print(chunk)
    stream_done()

    metrics = manager.last_stream_metrics
    if metrics is not None:
        print(f"Metrics: {metrics.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
