"""Application settings: config.yaml (tunables) + .env (secrets).

Two configuration sources, deliberately separate:

- ``config.yaml`` — every tunable default and validation boundary (sampling
  defaults, retry policy, timeouts, model IDs, fallback order). Parsed once
  into :class:`AppSettings`, so a malformed config file fails fast at startup
  with a clear ``ValidationError`` instead of surfacing mid-request.
- ``.env`` — API keys only, loaded with python-dotenv. Keys are wrapped in
  ``SecretStr`` so they never leak into logs or reprs.

Set the ``LLM_CLIENT_CONFIG`` environment variable to point at an alternative
YAML file (used by the tests to prove boundaries are config-driven).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, model_validator

from src.schemas import LLMConfig, Provider

CONFIG_ENV_VAR = "LLM_CLIENT_CONFIG"
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class Defaults(BaseModel):
    temperature: float
    max_tokens: int
    timeout_s: float
    max_attempts: int
    base_delay_s: float
    jitter_min: float
    jitter_max: float
    max_concurrent: int

    @model_validator(mode="after")
    def _coherent(self) -> "Defaults":
        if not (0 < self.jitter_min <= self.jitter_max):
            raise ValueError(
                f"jitter range must satisfy 0 < jitter_min <= jitter_max, "
                f"got [{self.jitter_min}, {self.jitter_max}]"
            )
        return self


class Limits(BaseModel):
    temperature_min: float
    temperature_max: float
    max_tokens_limit: int

    @model_validator(mode="after")
    def _coherent(self) -> "Limits":
        if self.temperature_min > self.temperature_max:
            raise ValueError(
                f"temperature_min ({self.temperature_min}) exceeds "
                f"temperature_max ({self.temperature_max})"
            )
        return self


class AppSettings(BaseModel):
    defaults: Defaults
    limits: Limits
    models: Dict[Provider, str]
    fallback_order: List[Provider]


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and validate the YAML config (cached after the first call)."""
    path = Path(os.getenv(CONFIG_ENV_VAR, str(_DEFAULT_CONFIG_PATH)))
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppSettings.model_validate(raw)


_ENV_KEY_NAMES = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GEMINI: "GOOGLE_API_KEY",
}

_CONFIG_KEY_FIELDS = {
    Provider.OPENAI: "openai_api_key",
    Provider.ANTHROPIC: "anthropic_api_key",
    Provider.GEMINI: "google_api_key",
}


def api_key_for(provider: Provider) -> Optional[SecretStr]:
    """Read a provider's API key from the environment / .env file."""
    load_dotenv()
    value = os.getenv(_ENV_KEY_NAMES[provider])
    return SecretStr(value) if value else None


def config_for(provider: Provider, **overrides) -> LLMConfig:
    """Build an LLMConfig for one provider from config.yaml + .env."""
    settings = get_settings()
    data = {
        "provider": provider,
        "model": settings.models[provider],
        _CONFIG_KEY_FIELDS[provider]: api_key_for(provider),
    }
    data.update(overrides)
    return LLMConfig(**data)


def fallback_chain(**overrides) -> List[LLMConfig]:
    """Configs for every provider in config.yaml's ``fallback_order``."""
    return [config_for(p, **overrides) for p in get_settings().fallback_order]
