"""Offline test doubles and fixtures. Unit tests run without network or an API key."""

from __future__ import annotations

import json
from typing import Any, List

import pytest
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from local_rag.config import CONFIG_ENV_VAR, get_settings
from local_rag.store import build_embeddings

BASE_CONFIG: dict[str, Any] = {
    "data_directory": "data",
    "persist_directory": "vectorstore",
    "collection_name": "techcorp_policies",
    "embedding_model_name": "intfloat/multilingual-e5-small",
    "embedding_document_prefix": "passage: ",
    "embedding_query_prefix": "query: ",
    "tiktoken_encoding": "cl100k_base",
    "chunk_size_tokens": 500,
    "chunk_overlap_tokens": 50,
    "top_k": 4,
    "llm": {
        "model_name": "gemini-test",
        "temperature": 0.0,
        "timeout_s": 30.0,
        "max_attempts": 3,
        "backoff": {"initial": 0.001, "max": 0.01, "jitter": 0.001},
    },
    "refusal_sentence": "No tengo acceso a esa información en los documentos disponibles.",
}


def answer_message(answer: str) -> AIMessage:
    """What a real Gemini call returns for the LLMAnswer JSON contract."""
    return AIMessage(content=json.dumps({"answer": answer}, ensure_ascii=False))


class ScriptedChatModel(BaseChatModel):
    """Replays ``script`` in order: an ``AIMessage`` is returned, an exception is raised.

    Records every message list it receives so tests can assert on what the model saw.
    """

    script: List[Any]
    calls: List[List[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

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


@pytest.fixture
def custom_config_file(tmp_path, monkeypatch):
    """Point the app at a temporary config.yaml, proving behaviour is config-driven.

    ``custom_config_file(**overrides)`` merges each override into the matching top-level
    section of ``BASE_CONFIG``; ``replace=True`` swaps whole sections instead.
    """

    def _apply(replace: bool = False, **overrides: Any) -> None:
        config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE_CONFIG.items()}
        for section, value in overrides.items():
            if not replace and isinstance(value, dict) and isinstance(config.get(section), dict):
                config[section].update(value)
            else:
                config[section] = value
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    """Every cached singleton is process-wide; clear them so tests don't leak into each other."""
    yield
    get_settings.cache_clear()
    build_embeddings.cache_clear()
    try:
        from local_rag.chain import get_pipeline

        get_pipeline.cache_clear()
    except ImportError:
        pass
