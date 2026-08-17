"""Simulates a failing primary provider (invalid API key on purpose) and
verifies the fallback answers — slide 42, step 5.

Run from the repo root:  python examples/demo_fallback.py
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from _bootstrap import available_configs
from pydantic import SecretStr

from src.manager import AsyncLLMManager
from src.schemas import ChatMessage, Provider


async def main() -> None:
    configs = available_configs(max_tokens=200)

    # Break the primary's key on purpose. An auth error is non-retryable,
    # so the manager should fail over to the next provider immediately.
    key_field = {
        Provider.OPENAI: "openai_api_key",
        Provider.ANTHROPIC: "anthropic_api_key",
        Provider.GEMINI: "google_api_key",
    }[configs[0].provider]
    broken_primary = configs[0].model_copy(
        update={key_field: SecretStr("invalid-key-on-purpose")}
    )

    if len(configs) < 2:
        print("Only one provider has a key — the fallback will also fail, "
              "but the demo still shows the error handling path.")

    manager = AsyncLLMManager(broken_primary, fallbacks=configs[1:])

    print(f"Primary: {broken_primary.provider.value} (key intentionally broken)")
    print(f"Fallbacks: {[c.provider.value for c in configs[1:]] or 'none'}\n")

    t0 = perf_counter()
    response = await manager.generate(
        [ChatMessage(role="user", content="Say 'hello' in Spanish, one word.")]
    )
    elapsed = (perf_counter() - t0) * 1000

    if response.ok:
        print(f"Answered by: {response.provider.value} ({response.model})")
        print(f"Content: {response.content}")
        print(f"Total time including the failover: {elapsed:.0f} ms")
    else:
        print(f"All providers failed: {response.error}")


if __name__ == "__main__":
    asyncio.run(main())
