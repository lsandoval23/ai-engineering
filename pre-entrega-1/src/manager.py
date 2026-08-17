"""AsyncLLMManager: factory + retry + fallback + semaphore + timeout + metrics.

This is the only class business logic should touch. It holds an ordered chain
of providers (primary first) behind the BaseLLMClient abstraction and:

- creates concrete clients via the Factory (crashing at startup on a missing
  key or unsupported provider — unrecoverable config errors should be loud);
- limits concurrent in-flight calls with an ``asyncio.Semaphore``;
- puts a deadline on every call with ``asyncio.timeout``;
- retries transient failures with exponential backoff + jitter;
- falls back to the next provider when one is exhausted;
- measures TTFT / total latency / throughput and logs every decision.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Sequence, Tuple, Type

from loguru import logger

from src.base import BaseLLMClient, StreamError
from src.clients import AnthropicClient, GeminiClient, OpenAIClient
from src.metrics import StreamMetrics, elapsed_ms
from src.resilience import with_retry
from src.schemas import ChatMessage, ErrorKind, LLMConfig, ModelResponse, Provider
from src.settings import get_settings
from time import perf_counter


class AllProvidersFailedError(RuntimeError):
    """Every provider in the chain failed before producing a single token."""


_CLIENT_CLASSES: Dict[Provider, Type[BaseLLMClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}


class AsyncLLMManager:
    def __init__(self, primary: LLMConfig, fallbacks: Optional[Sequence[LLMConfig]] = None):
        configs = [primary, *(fallbacks or [])]
        self._chain: List[Tuple[LLMConfig, BaseLLMClient]] = [
            (config, self._create_client(config)) for config in configs
        ]
        self._semaphore = asyncio.Semaphore(primary.max_concurrent)
        self.last_stream_metrics: Optional[StreamMetrics] = None
        logger.info(
            "manager ready: chain=[{chain}], max_concurrent={mc}",
            chain=" -> ".join(f"{c.provider.value}:{c.model}" for c, _ in self._chain),
            mc=primary.max_concurrent,
        )

    @staticmethod
    def _create_client(config: LLMConfig) -> BaseLLMClient:
        """Factory: the single place that decides which concrete class to build."""
        client_cls = _CLIENT_CLASSES.get(config.provider)
        if client_cls is None:
            raise ValueError(f"Unsupported provider: {config.provider}")
        if config.api_key is None:
            raise ValueError(f"Missing API key for provider '{config.provider.value}'")
        return client_cls(
            api_key=config.api_key.get_secret_value(),
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
        )

    @staticmethod
    async def _call_with_timeout(
        client: BaseLLMClient, config: LLMConfig, messages: List[ChatMessage]
    ) -> ModelResponse:
        """One provider call under a hard deadline. A timeout becomes a
        retryable ModelResponse error instead of an escaping exception."""
        try:
            async with asyncio.timeout(config.timeout_s):
                return await client.generate(messages)
        except TimeoutError:
            return ModelResponse(
                provider=config.provider,
                model=config.model,
                content="",
                error=f"call exceeded the {config.timeout_s}s timeout",
                error_kind=ErrorKind.TIMEOUT,
            )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Walk the provider chain until one answers; aggregate errors if none does."""
        jitter = get_settings().defaults
        async with self._semaphore:
            failures: List[str] = []
            last_kind: Optional[ErrorKind] = None
            for index, (config, client) in enumerate(self._chain):
                if index:
                    logger.warning(
                        "falling back to {provider} ({model})",
                        provider=config.provider.value,
                        model=config.model,
                    )
                t0 = perf_counter()
                response = await with_retry(
                    lambda c=client, cfg=config: self._call_with_timeout(c, cfg, messages),
                    max_attempts=config.max_attempts,
                    base_delay_s=config.base_delay_s,
                    jitter_min=jitter.jitter_min,
                    jitter_max=jitter.jitter_max,
                )
                latency = elapsed_ms(t0)
                if response.error is None:
                    logger.info(
                        "{provider} answered in {ms:.0f} ms",
                        provider=config.provider.value,
                        ms=latency,
                    )
                    return response.model_copy(update={"total_latency_ms": latency})
                failures.append(f"{config.provider.value}: {response.error}")
                last_kind = response.error_kind

            last_config = self._chain[-1][0]
            logger.error("all providers failed: {failures}", failures=" | ".join(failures))
            return ModelResponse(
                provider=last_config.provider,
                model=last_config.model,
                content="",
                error="All providers failed — " + " | ".join(failures),
                error_kind=last_kind,
            )

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Stream from the first provider that produces a token.

        Fallback policy: switch providers only while nothing has reached the
        user (StreamError / timeout / empty stream before the first token).
        Once tokens are flowing, a failure is surfaced in-stream instead of
        restarting on another provider, which would duplicate output.
        """
        async with self._semaphore:
            failures: List[str] = []
            start = perf_counter()
            for index, (config, client) in enumerate(self._chain):
                metrics = StreamMetrics()
                stream = client.generate_stream(messages)
                try:
                    async with asyncio.timeout(config.timeout_s):
                        first_chunk = await anext(stream)
                except StopAsyncIteration:
                    failures.append(f"{config.provider.value}: stream ended without tokens")
                    continue
                except StreamError as e:
                    failures.append(f"{e.provider.value}: {e}")
                    logger.warning(
                        "stream from {provider} failed before first token ({kind}); "
                        "trying next provider",
                        provider=e.provider.value,
                        kind=e.kind.value,
                    )
                    continue
                except TimeoutError:
                    failures.append(
                        f"{config.provider.value}: no first token within {config.timeout_s}s"
                    )
                    logger.warning(
                        "stream from {provider} timed out before first token; "
                        "trying next provider",
                        provider=config.provider.value,
                    )
                    continue

                if index:
                    logger.info(
                        "fallback to {provider} produced a token {ms:.0f} ms after the "
                        "original request started",
                        provider=config.provider.value,
                        ms=elapsed_ms(start),
                    )
                metrics.mark_chunk()
                yield first_chunk
                async for chunk in stream:
                    metrics.mark_chunk()
                    yield chunk
                metrics.finish()
                self.last_stream_metrics = metrics
                logger.info(
                    "stream from {provider} done: {summary}",
                    provider=config.provider.value,
                    summary=metrics.summary(),
                )
                return

            logger.error(
                "all providers failed before first token: {failures}",
                failures=" | ".join(failures),
            )
            raise AllProvidersFailedError(
                "All providers failed before producing a token — " + " | ".join(failures)
            )
