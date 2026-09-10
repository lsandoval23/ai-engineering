"""Chain behaviour, fully offline: structured output, retries, fallback, entrypoint."""

from __future__ import annotations

import logging

import httpx
import pytest
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.messages import AIMessage, HumanMessage

import src.chain as chain_module
import src.settings as settings_module
from src.chain import SchemaValidationError, build_chain, compose_chain, get_model, process_text
from src.schemas import CriticalityLevel, TechnicalEntities
from tests.conftest import FAST_BACKOFF, VALID_ARGS, tool_call_message

FAST = {"backoff": FAST_BACKOFF}
INPUT = {"text": "Nuestra API en FastAPI devuelve timeouts; Redis se satura."}
LOGGER = "extraction_pipeline"


async def test_happy_path_returns_a_validated_instance(scripted):
    model = scripted(tool_call_message(VALID_ARGS))
    result = await compose_chain(model, **FAST).ainvoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert result.criticality_level is CriticalityLevel.HIGH
    assert result.technologies == ["FastAPI", "Redis", "PostgreSQL"]
    assert len(model.calls) == 1


def test_chain_also_works_synchronously(scripted):
    """The error-aware step defines both bodies, so plain ``invoke`` is supported too."""
    invalid = {**VALID_ARGS, "technologies": []}
    model = scripted(tool_call_message(invalid), tool_call_message(VALID_ARGS))
    result = compose_chain(model, **FAST).invoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert len(model.calls) == 2


async def test_transient_errors_are_retried_with_backoff(scripted, caplog):
    model = scripted(
        ModelRateLimitError("429 too many requests"),
        ModelTimeoutError("read timed out"),
        tool_call_message(VALID_ARGS),
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await compose_chain(model, **FAST).ainvoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert len(model.calls) == 3
    messages = [r.getMessage() for r in caplog.records]
    failures = [m for m in messages if m.startswith("Model call failed with")]
    assert "ModelRateLimitError" in failures[0] and "ModelTimeoutError" in failures[1]
    retries = [m for m in messages if m.startswith("Retry attempt")]
    assert [m.split()[2] for m in retries] == ["2/3", "3/3"]


async def test_raw_httpx_errors_from_the_gemini_sdk_are_retried(scripted, caplog):
    """langchain-google-genai does not wrap timeouts/connection drops; the chain must."""
    model = scripted(
        httpx.TimeoutException("read timed out"),
        httpx.ConnectError("connection refused"),
        tool_call_message(VALID_ARGS),
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await compose_chain(model, **FAST).ainvoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert len(model.calls) == 3
    retries = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Retry attempt")]
    assert len(retries) == 2


async def test_truncated_response_is_detected_and_retried(scripted, caplog):
    model = scripted(
        tool_call_message(VALID_ARGS, finish_reason="length"),
        tool_call_message(VALID_ARGS),
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await compose_chain(model, **FAST).ainvoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert len(model.calls) == 2
    messages = [r.getMessage() for r in caplog.records]
    assert any("Response truncated by the provider (finish_reason=length)" in m for m in messages)
    assert any(m.startswith("Retry attempt 2/3") for m in messages)


async def test_validation_failure_triggers_error_aware_retry(scripted, caplog):
    invalid = {**VALID_ARGS, "technologies": ["", "  "]}
    model = scripted(tool_call_message(invalid), tool_call_message(VALID_ARGS))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await compose_chain(model, **FAST).ainvoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert len(model.calls) == 2  # one corrective call, no blind retries in between
    previous_answer, last_message = model.calls[1][-2:]
    # The model sees its own rejected answer as an assistant turn (no dangling tool call)...
    assert isinstance(previous_answer, AIMessage)
    assert not previous_answer.tool_calls
    assert '"technologies"' in previous_answer.content
    # ...followed by the validation error naming the field.
    assert isinstance(last_message, HumanMessage)
    assert "did not pass schema validation" in last_message.content
    assert "technologies" in last_message.content  # Pydantic names the field
    assert any("re-invoking once with the error as feedback" in r.getMessage() for r in caplog.records)


async def test_answer_without_structured_output_is_normalized_and_retried(scripted):
    """Any parser failure (refusal, bare ValueError, no tool call) becomes a format error."""
    plain_text = AIMessage(content="I cannot help with that.", response_metadata={"finish_reason": "stop"})
    model = scripted(plain_text, tool_call_message(VALID_ARGS))
    result = await compose_chain(model, **FAST).ainvoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert len(model.calls) == 2
    previous_answer = model.calls[1][-2]
    assert isinstance(previous_answer, AIMessage)
    assert previous_answer.content == "I cannot help with that."


async def test_second_validation_failure_propagates(scripted):
    invalid = {**VALID_ARGS, "technologies": []}
    model = scripted(tool_call_message(invalid), tool_call_message(invalid))
    with pytest.raises(SchemaValidationError) as info:
        await compose_chain(model, **FAST).ainvoke(INPUT)
    assert isinstance(info.value, OutputParserException)  # the normalized format-error family
    assert "technologies" in str(info.value)
    assert len(model.calls) == 2


async def test_fallback_model_answers_when_primary_exhausts_retries(scripted, caplog):
    primary = scripted(*[ModelRateLimitError("429")] * 3)
    fallback = scripted(tool_call_message(VALID_ARGS))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = await compose_chain(primary, fallback_model=fallback, **FAST).ainvoke(INPUT)
    assert isinstance(result, TechnicalEntities)
    assert len(primary.calls) == 3
    assert len(fallback.calls) == 1
    assert any("switching to the fallback model" in r.getMessage() for r in caplog.records)


async def test_permanent_errors_are_not_retried(scripted):
    primary = scripted(ModelAuthenticationError("401 invalid api key"))
    fallback = scripted(ModelAuthenticationError("401 invalid api key"))
    with pytest.raises(ModelAuthenticationError):
        await compose_chain(primary, fallback_model=fallback, **FAST).ainvoke(INPUT)
    assert len(primary.calls) == 1  # a bad key fails identically on every attempt
    assert len(fallback.calls) == 1


async def test_process_text_logs_and_returns(scripted, monkeypatch, caplog):
    model = scripted(tool_call_message(VALID_ARGS))
    monkeypatch.setattr(chain_module, "get_chain", lambda provider: compose_chain(model, **FAST))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = await process_text(INPUT["text"], provider="gemini")
    assert isinstance(result, TechnicalEntities)
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("[gemini] Processing text") for m in messages)
    assert any("[gemini] Validated extraction" in m and "FastAPI" in m for m in messages)


async def test_process_text_reraises_after_logging_error(scripted, monkeypatch, caplog):
    model = scripted(ModelAuthenticationError("401"))
    monkeypatch.setattr(chain_module, "get_chain", lambda provider: compose_chain(model, **FAST))
    with caplog.at_level(logging.ERROR, logger=LOGGER), pytest.raises(ModelAuthenticationError):
        await process_text(INPUT["text"], provider="gemini")
    assert any(r.levelno == logging.ERROR and "Extraction failed" in r.getMessage() for r in caplog.records)


async def test_process_text_logs_error_when_the_chain_cannot_be_built(monkeypatch, caplog):
    """A missing key or bad config fails inside the try, so it is logged like any other failure."""

    def broken_get_chain(provider):
        raise ValueError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(chain_module, "get_chain", broken_get_chain)
    with caplog.at_level(logging.ERROR, logger=LOGGER), pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        await process_text(INPUT["text"], provider="gemini")
    assert any(r.levelno == logging.ERROR and "GOOGLE_API_KEY" in r.getMessage() for r in caplog.records)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unsupported provider 'foo'"):
        build_chain("foo")


def test_missing_api_key_fails_fast_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        get_model("openai")


def test_get_model_is_driven_by_config(monkeypatch, custom_config_file):
    """Model ID, temperature and timeout come from config.yaml; SDK retries are off."""
    custom_config_file(models={"openai": "gpt-from-config"}, defaults={"timeout_s": 7.5})
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = get_model("openai")
    assert model.model_name == "gpt-from-config"
    assert model.request_timeout == 7.5
    assert model.temperature == 0
    assert model.max_retries == 0


def test_build_chain_uses_config_default_provider_and_fallback(monkeypatch, custom_config_file):
    custom_config_file(default_provider="openai", fallback_models={"openai": "gpt-lite"})
    monkeypatch.setattr(settings_module, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    built = []
    monkeypatch.setattr(chain_module, "get_model", lambda p, m=None: built.append((p, m)) or scripted_ok())
    build_chain()  # no provider given -> config default
    assert built == [("openai", None), ("openai", "gpt-lite")]


def scripted_ok():
    from tests.conftest import ScriptedChatModel

    return ScriptedChatModel(script=[tool_call_message(VALID_ARGS)])
