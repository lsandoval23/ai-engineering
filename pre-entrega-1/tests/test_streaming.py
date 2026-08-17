"""Streaming: ordered chunk delivery, pre-first-token fallback, ABC contract."""

from __future__ import annotations

import pytest

from src.base import BaseLLMClient
from src.manager import AllProvidersFailedError, AsyncLLMManager
from src.schemas import ErrorKind, Provider
from tests.conftest import FailingClient, SuccessClient, make_config


def _manager(*clients) -> AsyncLLMManager:
    configs = [make_config(c.provider) for c in clients]
    manager = AsyncLLMManager(configs[0], fallbacks=configs[1:])
    manager._chain = list(zip(configs, clients))
    return manager


async def _collect(manager, messages):
    return [chunk async for chunk in manager.generate_stream(messages)]


class TestStreaming:
    async def test_chunks_arrive_in_order(self, user_message):
        manager = _manager(SuccessClient(Provider.OPENAI, chunks=["Hola", " ", "mundo"]))
        chunks = await _collect(manager, user_message)
        assert chunks == ["Hola", " ", "mundo"]
        assert "".join(chunks) == "Hola mundo"

    async def test_metrics_are_recorded(self, user_message):
        manager = _manager(SuccessClient(Provider.OPENAI))
        await _collect(manager, user_message)
        metrics = manager.last_stream_metrics
        assert metrics is not None
        assert metrics.chunks == 3
        assert metrics.ttft_ms is not None and metrics.ttft_ms >= 0
        assert metrics.total_ms is not None and metrics.total_ms >= metrics.ttft_ms

    async def test_pre_first_token_failure_falls_back(self, user_message):
        primary = FailingClient(Provider.OPENAI, kind=ErrorKind.RATE_LIMIT)
        secondary = SuccessClient(Provider.ANTHROPIC, chunks=["desde", " anthropic"])
        chunks = await _collect(_manager(primary, secondary), user_message)
        assert "".join(chunks) == "desde anthropic"
        assert primary.calls == 1  # stream fallback happens without re-retrying

    async def test_all_streams_failing_raises(self, user_message):
        manager = _manager(
            FailingClient(Provider.OPENAI), FailingClient(Provider.ANTHROPIC)
        )
        with pytest.raises(AllProvidersFailedError, match="openai"):
            await _collect(manager, user_message)


class TestContract:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseLLMClient()  # abstract methods make instantiation impossible

    def test_incomplete_subclass_cannot_be_instantiated(self):
        class HalfClient(BaseLLMClient):
            async def generate(self, messages):  # generate_stream missing
                ...

        with pytest.raises(TypeError):
            HalfClient()
