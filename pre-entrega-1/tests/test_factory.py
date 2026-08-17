"""The Factory builds the right concrete client and crashes loudly on
unrecoverable configuration errors (missing key)."""

from __future__ import annotations

import pytest

from src.clients import AnthropicClient, GeminiClient, OpenAIClient
from src.manager import AsyncLLMManager
from src.schemas import Provider
from tests.conftest import make_config

EXPECTED_CLASS = {
    Provider.OPENAI: OpenAIClient,
    Provider.ANTHROPIC: AnthropicClient,
    Provider.GEMINI: GeminiClient,
}


@pytest.mark.parametrize("provider", list(Provider))
def test_factory_builds_matching_client(provider):
    manager = AsyncLLMManager(make_config(provider))
    _, client = manager._chain[0]
    assert isinstance(client, EXPECTED_CLASS[provider])


@pytest.mark.parametrize(
    ("provider", "missing_field"),
    [
        (Provider.OPENAI, "openai_api_key"),
        (Provider.ANTHROPIC, "anthropic_api_key"),
        (Provider.GEMINI, "google_api_key"),
    ],
)
def test_factory_raises_on_missing_key(provider, missing_field):
    config = make_config(provider, **{missing_field: None})
    with pytest.raises(ValueError, match="Missing API key"):
        AsyncLLMManager(config)


def test_fallbacks_join_the_chain():
    manager = AsyncLLMManager(
        make_config(Provider.OPENAI),
        fallbacks=[make_config(Provider.ANTHROPIC), make_config(Provider.GEMINI)],
    )
    providers = [config.provider for config, _ in manager._chain]
    assert providers == [Provider.OPENAI, Provider.ANTHROPIC, Provider.GEMINI]
