"""OpenAI client.

Fixes over the hint notebook:
- content can legitimately be None (tool calls, content filtering); it is
  coalesced to "" and ValidationError is caught anyway, so a Pydantic error
  can never escape the client (notebook bug 03).
- AuthenticationError / BadRequestError are classified as non-retryable via
  ErrorKind instead of being lumped in with transient failures.
- max_completion_tokens replaces the deprecated max_tokens parameter
  (notebook bug 07).
"""

from __future__ import annotations

from typing import AsyncGenerator, List, Tuple

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError

from src.base import BaseLLMClient, StreamError
from src.schemas import ChatMessage, ErrorKind, ModelResponse, Provider


def _classify(exc: APIError) -> Tuple[ErrorKind, str]:
    """Map an OpenAI SDK exception to an ErrorKind (isinstance order matters:
    subclasses before their parents)."""
    if isinstance(exc, RateLimitError):
        return ErrorKind.RATE_LIMIT, f"OpenAI rate limit exceeded: {exc}"
    if isinstance(exc, AuthenticationError):
        return ErrorKind.AUTH, f"OpenAI authentication failed: {exc}"
    if isinstance(exc, BadRequestError):
        return ErrorKind.BAD_REQUEST, f"OpenAI rejected the request: {exc}"
    if isinstance(exc, InternalServerError):
        return ErrorKind.SERVER, f"OpenAI server error: {exc}"
    if isinstance(exc, APITimeoutError):
        return ErrorKind.TIMEOUT, f"OpenAI request timed out: {exc}"
    if isinstance(exc, APIConnectionError):
        return ErrorKind.CONNECTION, f"OpenAI connection error: {exc}"
    return ErrorKind.API, f"OpenAI API error: {exc}"


class OpenAIClient(BaseLLMClient):
    provider = Provider.OPENAI

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, timeout_s: float):
        self._client = AsyncOpenAI(api_key=api_key, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )
            return ModelResponse(
                provider=self.provider,
                model=self.model,
                content=response.choices[0].message.content or "",
            )
        except APIError as e:
            return self._error_response(*_classify(e))
        except ValidationError as e:
            return self._error_response(ErrorKind.API, f"OpenAI response failed validation: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        yielded_any = False
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yielded_any = True
                    yield delta
        except APIError as e:
            kind, message = _classify(e)
            if not yielded_any:
                raise StreamError(self.provider, kind, message) from e
            yield f"\n[stream interrupted: {message}]"
