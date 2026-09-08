"""LCEL extraction chain: prompt | structured model | checks, made resilient.

Layers, innermost first (each one maps to a failure class from the course):

1. ``prompt | model.with_structured_output(TechnicalEntities) | check`` — the
   pipeline itself. ``include_raw=True`` keeps the raw ``AIMessage`` so the
   check step can read the provider's finish reason and detect a truncated
   response instead of only bouncing off a validation error.
2. ``.with_retry(...)`` — exponential backoff with jitter for **transient**
   failures: rate limits, timeouts, connection drops, 5xx and truncation.
3. Error-aware retry — on a **format** failure (the object did not validate)
   the chain is invoked once more with the validation error appended to the
   conversation, so the model can correct itself.
4. ``.with_fallbacks([...])`` — if the primary model still fails, the same
   stack runs on a lighter model of the same provider.

**Permanent** errors (bad key, bad request, unknown model) are never retried:
they fall through to the fallback and then to the caller.

Every tunable (provider, model IDs, temperature, timeout, attempts, backoff)
comes from ``config.yaml`` through :mod:`src.settings`; nothing is hardcoded here.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    OutputParserException,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.retry import ExponentialJitterParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.schemas import TechnicalEntities
from src.settings import PROVIDERS, Provider, get_settings, require_api_key

logger = logging.getLogger("extraction_pipeline")

# Provider-specific spellings of "the model ran out of output tokens".
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS"})


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class TruncatedResponseError(RuntimeError):
    """The provider cut the response short, so the JSON cannot be complete."""


# Transient: worth another attempt after a backoff. These are langchain-core's
# normalized classes; every provider integration maps its SDK errors onto them.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    ModelRateLimitError,
    ModelTimeoutError,
    ModelConnectionError,
    ModelAPIError,  # 5xx / overloaded
    TruncatedResponseError,
)
# Format: the model answered, but the object does not satisfy the contract.
FORMAT_ERRORS: tuple[type[BaseException], ...] = (ValidationError, OutputParserException)


# --------------------------------------------------------------------------- #
# Prompt — a template, never an f-string. The input key is exactly "text".
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a senior technical analyst. Extract structured information from the "
    "text the user provides.\n"
    "1. Identify the technologies, frameworks, tools or services mentioned.\n"
    "2. Assess the criticality level of the problem or architecture described.\n"
    "3. Write a brief technical summary (one or two sentences) in the same "
    "language as the input text."
)

FEEDBACK_TEMPLATE = (
    "Your previous answer did not pass schema validation:\n{error}\n\n"
    "Answer again, fixing exactly the reported problem and following the schema."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{text}"),
        # Only filled by the error-aware retry; ordinary calls pass just {"text"}.
        MessagesPlaceholder("feedback", optional=True),
    ]
)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def _validate_provider(provider: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(
            f"unsupported provider {provider!r}; expected one of {', '.join(PROVIDERS)}"
        )


def get_model(provider: Provider, model_id: str | None = None) -> BaseChatModel:
    """Return a configured chat model for ``provider``.

    ``temperature`` and ``timeout`` come from ``config.yaml``. ``max_retries=0``
    disables the SDKs' hidden retries so that ``with_retry`` is the single,
    observable retry layer. Keys come from the environment (``.env`` via
    python-dotenv); a missing key fails fast with a clear message.
    """
    _validate_provider(provider)
    require_api_key(provider)
    settings = get_settings()

    model_id = model_id or settings.models[provider]
    common: dict[str, Any] = {
        "temperature": settings.defaults.temperature,
        "timeout": settings.defaults.timeout_s,
        "max_retries": 0,
    }
    if provider == "openai":
        return ChatOpenAI(model=model_id, **common)
    if provider == "anthropic":
        return ChatAnthropic(model=model_id, **common)
    return ChatGoogleGenerativeAI(model=model_id, **common)


# --------------------------------------------------------------------------- #
# Chain composition
# --------------------------------------------------------------------------- #
def _finish_reason(message: AIMessage) -> str | None:
    """Provider-agnostic finish reason (OpenAI/Gemini: finish_reason; Anthropic: stop_reason)."""
    metadata = message.response_metadata or {}
    return metadata.get("finish_reason") or metadata.get("stop_reason")


def _check_and_unwrap(output: dict[str, Any]) -> TechnicalEntities:
    """Turn the ``include_raw`` dict into a validated object, or raise.

    Truncation is checked before parsing: a cut-off JSON also fails validation,
    but the root cause is the token limit, and that is what the log should say.
    """
    raw: AIMessage = output["raw"]
    reason = _finish_reason(raw)
    usage = raw.usage_metadata or {}
    logger.info(
        "Model responded: finish_reason=%s input_tokens=%s output_tokens=%s",
        reason,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
    )
    if reason in TRUNCATION_REASONS:
        logger.warning("Response truncated by the provider (finish_reason=%s)", reason)
        raise TruncatedResponseError(
            f"response cut short by the provider (finish_reason={reason!r}); "
            "the structured output is incomplete"
        )
    if output.get("parsing_error") is not None:
        raise output["parsing_error"]
    if output.get("parsed") is None:
        raise OutputParserException("model returned no structured output")
    return output["parsed"]


class RetryLogger(BaseCallbackHandler):
    """Makes the retry layer visible in the logs.

    ``with_retry`` tags every attempt after the first with ``retry:attempt:N``
    on the run it re-executes, so one ``on_chain_start`` carrying that tag is
    exactly one retry. Model-call failures are logged from ``on_llm_error``.
    """

    RETRY_TAG = "retry:attempt:"

    def __init__(self, max_attempts: int) -> None:
        self.max_attempts = max_attempts

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        for tag in tags or []:
            if tag.startswith(self.RETRY_TAG):
                attempt = int(tag[len(self.RETRY_TAG) :])
                logger.warning(
                    "Retry attempt %d/%d after exponential backoff", attempt, self.max_attempts
                )

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        logger.warning("Model call failed with %s: %s", type(error).__name__, error)


def _with_error_aware_retry(chain: Runnable) -> Runnable:
    """One corrective re-invocation when the output fails validation.

    Unlike the blind retry, the second call tells the model *what* was wrong:
    Pydantic's error already names the offending field.
    """

    async def run(inputs: dict[str, Any], config: RunnableConfig) -> TechnicalEntities:
        try:
            return await chain.ainvoke(inputs, config=config)
        except FORMAT_ERRORS as error:
            logger.warning(
                "Output failed validation (%s); re-invoking once with the error as feedback",
                type(error).__name__,
            )
            feedback = HumanMessage(content=FEEDBACK_TEMPLATE.format(error=error))
            return await chain.ainvoke({**inputs, "feedback": [feedback]}, config=config)

    return RunnableLambda(run, name="error_aware_retry")


def _log_fallback_switch(inputs: dict[str, Any]) -> dict[str, Any]:
    """First step of the fallback branch: say why the primary gave up."""
    error = inputs.get("fallback_error")
    logger.warning(
        "Primary model failed with %s: %s; switching to the fallback model",
        type(error).__name__,
        error,
    )
    return {key: value for key, value in inputs.items() if key != "fallback_error"}


def _resilient_extraction(
    model: BaseChatModel, *, max_attempts: int, backoff: ExponentialJitterParams
) -> Runnable:
    """The graded chain shape, plus the two retry layers, for one model."""
    core = (
        prompt
        | model.with_structured_output(TechnicalEntities, include_raw=True)
        | RunnableLambda(_check_and_unwrap)
    )
    resilient = core.with_retry(
        stop_after_attempt=max_attempts,
        wait_exponential_jitter=True,
        retry_if_exception_type=TRANSIENT_ERRORS,
        exponential_jitter_params=backoff,
    )
    return _with_error_aware_retry(resilient)


def compose_chain(
    model: BaseChatModel,
    *,
    fallback_model: BaseChatModel | None = None,
    max_attempts: int | None = None,
    backoff: ExponentialJitterParams | None = None,
) -> Runnable:
    """Compose the full chain around any chat model (tests inject fakes here).

    Input: ``{"text": str}``. Output: a validated ``TechnicalEntities``.
    ``max_attempts`` and ``backoff`` default to the values in ``config.yaml``.
    """
    defaults = get_settings().defaults
    attempts = max_attempts or defaults.max_attempts
    wait: ExponentialJitterParams = backoff or defaults.backoff.model_dump()  # type: ignore[assignment]

    primary = _resilient_extraction(model, max_attempts=attempts, backoff=wait)
    chain = primary
    if fallback_model is not None:
        backup = RunnableLambda(_log_fallback_switch) | _resilient_extraction(
            fallback_model, max_attempts=attempts, backoff=wait
        )
        chain = primary.with_fallbacks([backup], exception_key="fallback_error")
    return chain.with_config(callbacks=[RetryLogger(attempts)])


def build_chain(provider: Provider | None = None) -> Runnable:
    """Build the extraction chain for ``provider`` (default from ``config.yaml``)."""
    settings = get_settings()
    provider = provider or settings.default_provider
    _validate_provider(provider)
    fallback_id = settings.fallback_models.get(provider)
    return compose_chain(
        get_model(provider),
        fallback_model=get_model(provider, fallback_id) if fallback_id else None,
    )


@lru_cache(maxsize=None)
def get_chain(provider: Provider) -> Runnable:
    """``build_chain`` once per provider: model clients hold HTTP sessions."""
    return build_chain(provider)


# --------------------------------------------------------------------------- #
# Async entrypoint
# --------------------------------------------------------------------------- #
async def process_text(text: str, provider: Provider | None = None) -> TechnicalEntities:
    """Run the chain asynchronously and return a validated object, or raise.

    There is no third outcome: on failure the error is logged and re-raised,
    never replaced by a silent default.
    """
    provider = provider or get_settings().default_provider
    chain = get_chain(provider)
    logger.info("[%s] Processing text (%d characters)", provider, len(text))
    try:
        result = await chain.ainvoke({"text": text})
    except Exception as error:
        logger.error(
            "[%s] Extraction failed after retries and fallback: %s: %s",
            provider,
            type(error).__name__,
            error,
        )
        raise
    logger.info("[%s] Validated extraction: %s", provider, result.model_dump(mode="json"))
    return result
