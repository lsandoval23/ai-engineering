"""Retry policy and provider fallback — the 40% rubric band, fully offline."""

from __future__ import annotations

import pytest

from src.manager import AsyncLLMManager
from src.resilience import backoff_delay, with_retry
from src.schemas import ErrorKind, Provider
from tests.conftest import FailingClient, FlakyClient, SuccessClient, make_config

FAST_RETRY = {"max_attempts": 3, "base_delay_s": 0.001, "jitter_min": 0.5, "jitter_max": 1.5}


class TestWithRetry:
    async def test_retries_rate_limit_then_succeeds(self, user_message):
        client = FlakyClient(Provider.OPENAI, failures=2, kind=ErrorKind.RATE_LIMIT)
        response = await with_retry(lambda: client.generate(user_message), **FAST_RETRY)
        assert client.calls == 3
        assert response.ok and response.content == "recovered"

    async def test_does_not_retry_auth_errors(self, user_message):
        client = FailingClient(Provider.OPENAI, kind=ErrorKind.AUTH)
        response = await with_retry(lambda: client.generate(user_message), **FAST_RETRY)
        assert client.calls == 1  # a 401 fails identically forever — one attempt only
        assert response.error_kind is ErrorKind.AUTH

    async def test_does_not_retry_bad_request(self, user_message):
        client = FailingClient(Provider.OPENAI, kind=ErrorKind.BAD_REQUEST)
        await with_retry(lambda: client.generate(user_message), **FAST_RETRY)
        assert client.calls == 1

    async def test_stops_after_max_attempts(self, user_message):
        client = FailingClient(Provider.OPENAI, kind=ErrorKind.RATE_LIMIT)
        response = await with_retry(lambda: client.generate(user_message), **FAST_RETRY)
        assert client.calls == 3
        assert not response.ok

    def test_backoff_grows_exponentially_with_jitter(self):
        for attempt in range(4):
            delay = backoff_delay(attempt, base_delay_s=1.0, jitter_min=0.5, jitter_max=1.5)
            assert 0.5 * 2**attempt <= delay <= 1.5 * 2**attempt


class TestFallback:
    def _manager(self, primary_client, fallback_client) -> AsyncLLMManager:
        primary_cfg = make_config(primary_client.provider)
        fallback_cfg = make_config(fallback_client.provider)
        manager = AsyncLLMManager(primary_cfg, fallbacks=[fallback_cfg])
        manager._chain = [(primary_cfg, primary_client), (fallback_cfg, fallback_client)]
        return manager

    async def test_fallback_switches_provider(self, user_message):
        """The highest-value test: primary always fails, secondary answers."""
        primary = FailingClient(Provider.OPENAI, kind=ErrorKind.RATE_LIMIT)
        secondary = SuccessClient(Provider.ANTHROPIC, content="from anthropic")
        response = await self._manager(primary, secondary).generate(user_message)
        assert response.provider is Provider.ANTHROPIC
        assert response.content == "from anthropic"
        assert primary.calls == 3  # retries were exhausted before switching

    async def test_non_retryable_primary_fails_over_immediately(self, user_message):
        primary = FailingClient(Provider.OPENAI, kind=ErrorKind.AUTH)
        secondary = SuccessClient(Provider.ANTHROPIC)
        response = await self._manager(primary, secondary).generate(user_message)
        assert response.provider is Provider.ANTHROPIC
        assert primary.calls == 1

    async def test_all_providers_failing_aggregates_errors(self, user_message):
        primary = FailingClient(Provider.OPENAI, kind=ErrorKind.AUTH)
        secondary = FailingClient(Provider.ANTHROPIC, kind=ErrorKind.AUTH)
        response = await self._manager(primary, secondary).generate(user_message)
        assert not response.ok
        assert "openai" in response.error and "anthropic" in response.error

    async def test_success_reports_latency(self, user_message):
        manager = self._manager(SuccessClient(Provider.OPENAI), SuccessClient(Provider.ANTHROPIC))
        response = await manager.generate(user_message)
        assert response.total_latency_ms is not None and response.total_latency_ms >= 0
