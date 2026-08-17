"""The provider contract: BaseLLMClient (ABC) and the streaming error signal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List

from src.schemas import ChatMessage, ErrorKind, ModelResponse, Provider


class StreamError(Exception):
    """A stream failed before producing its first token.

    Raised (instead of yielding error text) so the manager can distinguish
    "nothing reached the user yet — safe to fall back" from a mid-stream
    failure, where tokens were already delivered and falling back would
    duplicate output.
    """

    def __init__(self, provider: Provider, kind: ErrorKind, message: str):
        super().__init__(message)
        self.provider = provider
        self.kind = kind


class BaseLLMClient(ABC):
    """Contract every LLM client must fulfil, regardless of the real provider.

    Concrete clients must set ``provider`` and ``model`` attributes, never let
    provider exceptions escape ``generate`` (they return a ModelResponse with
    ``error``/``error_kind`` instead), and raise :class:`StreamError` from
    ``generate_stream`` only before the first token.
    """

    provider: Provider
    model: str

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Generate a complete response (non-streaming)."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Generate the response token by token (streaming)."""
        raise NotImplementedError
        yield  # pragma: no cover — marks this method as an async generator

    def _error_response(self, kind: ErrorKind, message: str) -> ModelResponse:
        """Uniform error result used by all concrete clients."""
        return ModelResponse(
            provider=self.provider,
            model=self.model,
            content="",
            error=message,
            error_kind=kind,
        )
