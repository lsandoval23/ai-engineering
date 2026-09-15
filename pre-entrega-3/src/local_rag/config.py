"""Application settings: config.yaml (tunables) validated into AppSettings, plus .env (secrets)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_ENV_VAR: str = "LOCAL_RAG_CONFIG"
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
API_KEY_ENV_VAR: str = "GOOGLE_API_KEY"

# The brief's "infinite context" rule: keep top_k between 3 and 5.
MIN_TOP_K: int = 3
MAX_TOP_K: int = 5


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


class LLMSettings(BaseModel):
    """Chat-model parameters (the model id is config, not code)."""

    model_name: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    timeout_s: float = Field(gt=0)
    max_attempts: int = Field(ge=1)
    backoff: Backoff


class AppSettings(BaseModel):
    """Every tunable of the system, loaded once from config.yaml and validated."""

    data_directory: str = Field(min_length=1)
    persist_directory: str = Field(min_length=1)
    collection_name: str = Field(min_length=1)

    # The one place the embedding model is named: ingest and query both read it here.
    embedding_model_name: str = Field(min_length=1)
    embedding_document_prefix: str = ""
    embedding_query_prefix: str = ""

    tiktoken_encoding: str = Field(min_length=1)
    chunk_size_tokens: int = Field(ge=1)
    chunk_overlap_tokens: int = Field(ge=0)

    top_k: int = Field(ge=MIN_TOP_K, le=MAX_TOP_K)

    llm: LLMSettings

    # User-facing (Spanish): the fixed sentence returned when the context lacks the answer.
    refusal_sentence: str = Field(min_length=1)

    @model_validator(mode="after")
    def _overlap_below_chunk_size(self) -> "AppSettings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be smaller than "
                f"chunk_size_tokens ({self.chunk_size_tokens})"
            )
        return self

    @property
    def data_path(self) -> Path:
        """Absolute path of the documents folder."""
        return _resolve(self.data_directory)

    @property
    def persist_path(self) -> Path:
        """Absolute path of the ChromaDB persistence folder."""
        return _resolve(self.persist_directory)


def _resolve(directory: str) -> Path:
    path = Path(directory)
    return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and validate the YAML config (cached after the first call).

    Set ``LOCAL_RAG_CONFIG`` to point at an alternative file (the tests do).
    """
    path = Path(os.getenv(CONFIG_ENV_VAR, str(DEFAULT_CONFIG_PATH)))
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AppSettings.model_validate(raw)


def load_environment() -> None:
    """Load ``.env`` from the project root into the process environment (secrets only)."""
    load_dotenv(PROJECT_ROOT / ".env")


def require_api_key() -> None:
    """Fail fast, with a clear message, if the LLM key is not in the environment."""
    load_environment()
    if not os.getenv(API_KEY_ENV_VAR):
        raise ValueError(f"{API_KEY_ENV_VAR} is not set; add it to .env (see .env.example)")
