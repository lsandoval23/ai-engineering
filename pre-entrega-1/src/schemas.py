"""Pydantic schemas: the validated data contracts of the whole client.

Design note: tunable defaults (temperature, retries, timeouts, ...) and
validation boundaries (temperature range, token cap) are NOT hardcoded here.
They come from ``config.yaml`` via :mod:`src.settings` — see the
``_fill_defaults`` and ``_within_limits`` validators on :class:`LLMConfig`.
Only mathematical invariants (e.g. "a timeout must be positive") live in code.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ErrorKind(str, Enum):
    """Classification of a failed call, used to decide retry/fallback behavior.

    Retryable kinds (rate_limit, connection, timeout, server) are declared in
    :mod:`src.resilience`; auth and bad_request fail identically forever and
    must never be retried.
    """

    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    SERVER = "server"
    API = "api"


class ChatMessage(BaseModel):
    role: str = Field(description="'user', 'assistant' or 'system'")
    content: str

    @field_validator("role")
    @classmethod
    def _valid_role(cls, v: str) -> str:
        allowed = {"user", "assistant", "system"}
        if v not in allowed:
            raise ValueError(f"role must be one of {allowed}, got: '{v}'")
        return v


# LLMConfig fields whose defaults come from config.yaml's `defaults` section.
_DEFAULTED_FIELDS = (
    "temperature",
    "max_tokens",
    "timeout_s",
    "max_attempts",
    "base_delay_s",
    "max_concurrent",
)


class LLMConfig(BaseModel):
    """Everything needed to build and drive one provider client.

    All fields except ``provider``, ``model`` and the API keys default to the
    values in ``config.yaml`` and are range-checked against its ``limits``
    section.
    """

    provider: Provider
    model: str
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None

    # Defaults injected from config.yaml by _fill_defaults (None = "use default")
    temperature: float = None  # type: ignore[assignment]
    max_tokens: int = None  # type: ignore[assignment]
    timeout_s: float = None  # type: ignore[assignment]
    max_attempts: int = None  # type: ignore[assignment]
    base_delay_s: float = None  # type: ignore[assignment]
    max_concurrent: int = None  # type: ignore[assignment]

    @model_validator(mode="before")
    @classmethod
    def _fill_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            from src.settings import get_settings  # deferred to avoid an import cycle

            defaults = get_settings().defaults
            for name in _DEFAULTED_FIELDS:
                if data.get(name) is None:
                    data[name] = getattr(defaults, name)
        return data

    @model_validator(mode="after")
    def _within_limits(self) -> "LLMConfig":
        from src.settings import get_settings  # deferred to avoid an import cycle

        limits = get_settings().limits
        if not (limits.temperature_min <= self.temperature <= limits.temperature_max):
            raise ValueError(
                f"temperature must be between {limits.temperature_min} and "
                f"{limits.temperature_max} (config.yaml limits), got {self.temperature}"
            )
        if not (0 < self.max_tokens <= limits.max_tokens_limit):
            raise ValueError(
                f"max_tokens must be between 1 and {limits.max_tokens_limit} "
                f"(config.yaml limits), got {self.max_tokens}"
            )
        # Structural invariants — not tunable, so they live in code.
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
        if self.base_delay_s <= 0:
            raise ValueError(f"base_delay_s must be positive, got {self.base_delay_s}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be at least 1, got {self.max_concurrent}")
        return self

    @property
    def api_key(self) -> Optional[SecretStr]:
        """The key matching this config's provider, or None if absent."""
        return {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.GEMINI: self.google_api_key,
        }[self.provider]


class ModelResponse(BaseModel):
    """Result object returned by every client call — errors included.

    Clients never raise provider exceptions; they return a ModelResponse with
    ``error``/``error_kind`` set, so ``error is None`` is the success check and
    ``error_kind`` drives retries and fallback without string parsing.
    """

    provider: Provider
    model: str
    content: str
    error: Optional[str] = None
    error_kind: Optional[ErrorKind] = None

    # Metrics, filled by the manager (ge=0 is a physical invariant).
    ttft_ms: Optional[float] = Field(default=None, ge=0)
    total_latency_ms: Optional[float] = Field(default=None, ge=0)
    tokens_per_second: Optional[float] = Field(default=None, ge=0)

    @property
    def ok(self) -> bool:
        return self.error is None
