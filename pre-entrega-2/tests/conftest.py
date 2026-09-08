"""Offline test doubles and fixtures. Every test runs without network or API keys."""

from __future__ import annotations

from typing import Any, List

import pytest
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.settings import CONFIG_ENV_VAR, get_settings

VALID_ARGS = {
    "technologies": ["FastAPI", "Redis", "PostgreSQL"],
    "criticality_level": "high",
    "technical_summary": "The API times out because Redis saturates and the pool is exhausted.",
}

# Tests that build chains pass this so retries do not actually wait.
FAST_BACKOFF = {"initial": 0.001, "max": 0.01, "jitter": 0.001}


def tool_call_message(args: dict[str, Any], finish_reason: str = "stop") -> AIMessage:
    """What a real provider returns for structured output: a tool call + metadata."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "TechnicalEntities", "args": args, "id": "call_1", "type": "tool_call"}
        ],
        response_metadata={"finish_reason": finish_reason},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


class ScriptedChatModel(BaseChatModel):
    """Replays ``script`` in order: an ``AIMessage`` is returned, an exception is raised.

    Records every message list it receives so tests can assert on what the
    model was actually asked (e.g. that the feedback message was appended).
    """

    script: List[Any]
    calls: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        # with_structured_output needs this; the script already contains tool calls.
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError(
                "script exhausted: the chain called the model more times than expected"
            )
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


@pytest.fixture
def scripted():
    """Factory: ``scripted(item, item, ...)`` -> ScriptedChatModel."""
    return lambda *items: ScriptedChatModel(script=list(items))


BASE_CONFIG = {
    "default_provider": "gemini",
    "defaults": {
        "temperature": 0.0,
        "timeout_s": 30.0,
        "max_attempts": 3,
        "backoff": {"initial": 1.0, "max": 10.0, "jitter": 1.0},
    },
    "models": {"openai": "gpt-test", "anthropic": "claude-test", "gemini": "gemini-test"},
    "fallback_models": {"openai": "gpt-lite", "anthropic": "claude-lite", "gemini": "gemini-lite"},
}


@pytest.fixture
def custom_config_file(tmp_path, monkeypatch):
    """Point the app at a temporary config.yaml, proving behaviour is config-driven.

    ``custom_config_file(**overrides)`` merges each override into the matching
    top-level section of ``BASE_CONFIG``; ``replace=True`` swaps whole sections
    instead. Restores the real config (and settings cache) afterwards.
    """

    def _apply(replace: bool = False, **overrides) -> None:
        config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CONFIG.items()}
        for section, value in overrides.items():
            if not replace and isinstance(value, dict) and isinstance(config.get(section), dict):
                config[section].update(value)
            else:
                config[section] = value
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()
