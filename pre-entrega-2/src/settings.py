"""Application settings: config.yaml (tunables) + .env (secrets).

Two configuration sources, deliberately separate:

- ``config.yaml`` — every tunable default: default provider, model IDs,
  fallback models, temperature, timeout, retry policy. Parsed once into
  :class:`AppSettings`, so a malformed config file fails fast at startup with a
  clear ``ValidationError`` instead of surfacing mid-request.
- ``.env`` — API keys only, loaded with python-dotenv and read by the LangChain
  model classes straight from the environment. Nothing here ever holds a key.

Set the ``EXTRACTION_PIPELINE_CONFIG`` environment variable to point at an
alternative YAML file (the tests use it to prove behaviour is config-driven).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, get_args

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

CONFIG_ENV_VAR = "EXTRACTION_PIPELINE_CONFIG"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

Provider = Literal["openai", "anthropic", "gemini"]
PROVIDERS: tuple[str, ...] = get_args(Provider)

ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


class Backoff(BaseModel):
    """Parameters of LangChain's ``wait_exponential_jitter`` (seconds)."""

    initial: float = Field(gt=0)
    max: float = Field(gt=0)
    jitter: float = Field(ge=0)

    @model_validator(mode="after")
    def _coherent(self) -> "Backoff":
        if self.max < self.initial:
            raise ValueError(f"backoff.max ({self.max}) is below backoff.initial ({self.initial})")
        return self


class Defaults(BaseModel):
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_s: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    backoff: Backoff


class AppSettings(BaseModel):
    default_provider: Provider
    defaults: Defaults
    models: Dict[Provider, str]
    fallback_models: Dict[Provider, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _every_provider_has_a_model(self) -> "AppSettings":
        missing = [p for p in PROVIDERS if p not in self.models]
        if missing:
            raise ValueError(f"models: missing entry for provider(s) {', '.join(missing)}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and validate the YAML config (cached after the first call)."""
    path = Path(os.getenv(CONFIG_ENV_VAR, str(_DEFAULT_CONFIG_PATH)))
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppSettings.model_validate(raw)


def require_api_key(provider: str) -> None:
    """Fail fast, with a clear message, if the provider's key is not in the environment."""
    load_dotenv()
    env_key = ENV_KEYS[provider]
    if not os.getenv(env_key):
        raise ValueError(f"{env_key} is not set; add it to .env (see .env.example)")
