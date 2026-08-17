"""Shared fixtures and fake clients. Every test runs offline — no network."""

from __future__ import annotations

from typing import AsyncGenerator, List, Sequence

import pytest
import yaml
from pydantic import SecretStr

from src.base import BaseLLMClient, StreamError
from src.schemas import ChatMessage, ErrorKind, LLMConfig, ModelResponse, Provider
from src.settings import CONFIG_ENV_VAR, get_settings


def make_config(provider: Provider = Provider.OPENAI, **overrides) -> LLMConfig:
    """A valid config with dummy keys for every provider and fast retries."""
    data = {
        "provider": provider,
        "model": "test-model",
        "openai_api_key": SecretStr("sk-test"),
        "anthropic_api_key": SecretStr("sk-ant-test"),
        "google_api_key": SecretStr("g-test"),
        "base_delay_s": 0.001,  # keep retry tests fast
    }
    data.update(overrides)
    return LLMConfig(**data)


class SuccessClient(BaseLLMClient):
    """Always answers; counts calls and streams the given chunks."""

    def __init__(
        self,
        provider: Provider,
        content: str = "ok",
        chunks: Sequence[str] = ("Hola", " ", "mundo"),
    ):
        self.provider = provider
        self.model = "fake-model"
        self.content = content
        self.chunks = list(chunks)
        self.calls = 0

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        self.calls += 1
        return ModelResponse(provider=self.provider, model=self.model, content=self.content)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        self.calls += 1
        for chunk in self.chunks:
            yield chunk


class FailingClient(BaseLLMClient):
    """Always fails with the given error kind; counts calls."""

    def __init__(self, provider: Provider, kind: ErrorKind = ErrorKind.RATE_LIMIT):
        self.provider = provider
        self.model = "fake-model"
        self.kind = kind
        self.calls = 0

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        self.calls += 1
        return self._error_response(self.kind, f"simulated {self.kind.value} error")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        self.calls += 1
        raise StreamError(self.provider, self.kind, f"simulated {self.kind.value} error")
        yield  # pragma: no cover — marks this as an async generator


class FlakyClient(BaseLLMClient):
    """Fails ``failures`` times, then succeeds — for retry tests."""

    def __init__(
        self,
        provider: Provider,
        failures: int,
        kind: ErrorKind = ErrorKind.RATE_LIMIT,
        content: str = "recovered",
    ):
        self.provider = provider
        self.model = "fake-model"
        self.remaining_failures = failures
        self.kind = kind
        self.content = content
        self.calls = 0

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        self.calls += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            return self._error_response(self.kind, f"simulated {self.kind.value} error")
        return ModelResponse(provider=self.provider, model=self.model, content=self.content)

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        self.calls += 1
        yield self.content


@pytest.fixture
def user_message() -> List[ChatMessage]:
    return [ChatMessage(role="user", content="hola")]


@pytest.fixture
def custom_config_file(tmp_path, monkeypatch):
    """Point the app at a temporary config.yaml, proving boundaries are
    config-driven. Restores the real config (and settings cache) afterwards."""

    def _apply(**overrides) -> None:
        base = {
            "defaults": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "timeout_s": 30.0,
                "max_attempts": 3,
                "base_delay_s": 1.0,
                "jitter_min": 0.5,
                "jitter_max": 1.5,
                "max_concurrent": 5,
            },
            "limits": {
                "temperature_min": 0.0,
                "temperature_max": 2.0,
                "max_tokens_limit": 8192,
            },
            "models": {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-haiku-4-5",
                "gemini": "gemini-flash-latest",
            },
            "fallback_order": ["openai", "anthropic", "gemini"],
        }
        for section, values in overrides.items():
            base[section].update(values)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(base), encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()
