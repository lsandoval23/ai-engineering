"""Sequential vs concurrent: the same N calls, one after another and then
with asyncio.gather — the most legible proof the architecture is async.

Run from the repo root:  python examples/demo_gather.py
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from _bootstrap import available_configs

from src.manager import AsyncLLMManager
from src.schemas import ChatMessage

PROMPTS = [
    "Name one planet of the solar system.",
    "Say a prime number under 20.",
    "Name a chemical element.",
    "Say a color.",
    "Name a programming language.",
]


def _messages(prompt: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=f"{prompt} Answer with one word only.")]


async def main() -> None:
    configs = available_configs(max_tokens=20)
    manager = AsyncLLMManager(configs[0], fallbacks=configs[1:])

    print(f"Running {len(PROMPTS)} prompts against {configs[0].provider.value}...\n")

    t0 = perf_counter()
    sequential = [await manager.generate(_messages(p)) for p in PROMPTS]
    t_seq = perf_counter() - t0

    t0 = perf_counter()
    concurrent = await asyncio.gather(*(manager.generate(_messages(p)) for p in PROMPTS))
    t_gather = perf_counter() - t0

    for prompt, response in zip(PROMPTS, concurrent):  # gather preserves launch order
        answer = response.content.strip() if response.ok else f"error: {response.error}"
        print(f"  {prompt:45s} -> {answer}")

    errors = sum(1 for r in [*sequential, *concurrent] if not r.ok)
    print(f"\nSequential: {t_seq:.2f} s")
    print(f"Concurrent (asyncio.gather): {t_gather:.2f} s")
    print(f"Speedup: {t_seq / t_gather:.1f}x  ({errors} errors)")


if __name__ == "__main__":
    asyncio.run(main())
