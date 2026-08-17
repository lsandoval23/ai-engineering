"""Pydantic validation: bad values are rejected before any API call,
and both defaults and boundaries come from config.yaml, not from code."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import ChatMessage, LLMConfig, ModelResponse, Provider
from src.settings import get_settings
from tests.conftest import make_config


class TestLLMConfig:
    def test_rejects_temperature_above_limit(self):
        with pytest.raises(ValidationError, match="temperature"):
            make_config(temperature=5)

    def test_rejects_zero_max_tokens(self):
        with pytest.raises(ValidationError, match="max_tokens"):
            make_config(max_tokens=0)

    def test_rejects_negative_timeout(self):
        with pytest.raises(ValidationError, match="timeout_s"):
            make_config(timeout_s=-1)

    def test_rejects_zero_max_attempts(self):
        with pytest.raises(ValidationError, match="max_attempts"):
            make_config(max_attempts=0)

    def test_defaults_come_from_config_yaml(self):
        config = LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini")
        defaults = get_settings().defaults
        assert config.temperature == defaults.temperature
        assert config.max_tokens == defaults.max_tokens
        assert config.max_attempts == defaults.max_attempts
        assert config.max_concurrent == defaults.max_concurrent

    def test_api_key_property_matches_provider(self):
        config = make_config(Provider.ANTHROPIC)
        assert config.api_key is config.anthropic_api_key

    def test_api_key_never_leaks_in_repr(self):
        config = make_config(Provider.OPENAI)
        assert "sk-test" not in repr(config)


class TestConfigDrivenBoundaries:
    """Changing config.yaml changes what is accepted — no code change needed."""

    def test_tightened_temperature_limit_is_enforced(self, custom_config_file):
        custom_config_file(limits={"temperature_max": 1.0})
        with pytest.raises(ValidationError, match="temperature"):
            make_config(temperature=1.5)  # valid under default limits (max 2.0)

    def test_changed_default_is_injected(self, custom_config_file):
        custom_config_file(defaults={"temperature": 0.2})
        assert make_config().temperature == 0.2

    def test_invalid_config_file_fails_fast(self, custom_config_file):
        with pytest.raises(ValidationError):
            custom_config_file(limits={"temperature_min": 3.0})  # min > max
            get_settings()


class TestChatMessage:
    def test_rejects_invalid_role(self):
        with pytest.raises(ValidationError, match="role"):
            ChatMessage(role="robot", content="hola")

    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    def test_accepts_valid_roles(self, role):
        assert ChatMessage(role=role, content="hola").role == role


class TestModelResponse:
    def test_ok_property(self):
        good = ModelResponse(provider=Provider.OPENAI, model="m", content="hi")
        bad = ModelResponse(provider=Provider.OPENAI, model="m", content="", error="boom")
        assert good.ok and not bad.ok

    def test_rejects_negative_metrics(self):
        with pytest.raises(ValidationError, match="ttft_ms"):
            ModelResponse(provider=Provider.OPENAI, model="m", content="hi", ttft_ms=-1)
