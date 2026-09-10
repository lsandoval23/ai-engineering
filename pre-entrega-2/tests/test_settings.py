"""config.yaml is parsed into a validated AppSettings and drives the pipeline."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings import get_settings


def test_shipped_config_is_valid_and_complete():
    settings = get_settings()
    assert settings.default_provider in ("openai", "anthropic", "gemini")
    assert set(settings.models) == {"openai", "anthropic", "gemini"}
    assert settings.defaults.temperature == 0.0
    assert settings.defaults.max_attempts >= 2  # "at least one automatic retry"


def test_custom_config_file_is_honoured(custom_config_file):
    custom_config_file(default_provider="anthropic", models={"anthropic": "claude-custom"})
    settings = get_settings()
    assert settings.default_provider == "anthropic"
    assert settings.models["anthropic"] == "claude-custom"
    assert settings.models["openai"] == "gpt-test"  # untouched entries survive the merge


def test_fallback_models_are_optional_per_provider(custom_config_file):
    custom_config_file(fallback_models={"gemini": "gemini-lite"})
    assert get_settings().fallback_models.get("openai") == "gpt-lite"
    custom_config_file(fallback_models={}, replace=True)
    assert get_settings().fallback_models == {}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"default_provider": "cohere"}, "default_provider"),
        ({"defaults": {"max_attempts": 0}}, "max_attempts"),
        ({"defaults": {"temperature": 3.0}}, "temperature"),
        ({"defaults": {"backoff": {"initial": 5.0, "max": 1.0, "jitter": 1.0}}}, "backoff.max"),
        ({"models": {"openai": "gpt-test", "gemini": "gemini-test"}}, "missing entry"),
    ],
)
def test_malformed_config_fails_fast(custom_config_file, overrides, message):
    custom_config_file(replace="models" in overrides, **overrides)  # replace to drop a provider
    with pytest.raises(ValidationError, match=message):
        get_settings()
