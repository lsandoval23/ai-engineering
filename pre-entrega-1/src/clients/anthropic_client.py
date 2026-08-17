"""Anthropic client.

Fixes over the hint notebook:
- role="system" messages are hoisted into Anthropic's top-level ``system=``
  parameter instead of being passed inside ``messages=`` — the same split
  GeminiClient already did (notebook bug 02, the demo-breaking one).
- AuthenticationError / BadRequestError classified as non-retryable.
"""

from __future__ import annotations

from typing import AsyncGenerator, List, Optional, Tuple

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from src.base import BaseLLMClient, StreamError
from src.schemas import ChatMessage, ErrorKind, ModelResponse, Provider


def _classify(exc: APIError) -> Tuple[ErrorKind, str]:
    if isinstance(exc, RateLimitError):
        return ErrorKind.RATE_LIMIT, f"Anthropic rate limit exceeded: {exc}"
    if isinstance(exc, AuthenticationError):
        return ErrorKind.AUTH, f"Anthropic authentication failed: {exc}"
    if isinstance(exc, BadRequestError):
        return ErrorKind.BAD_REQUEST, f"Anthropic rejected the request: {exc}"
    if isinstance(exc, InternalServerError):
        return ErrorKind.SERVER, f"Anthropic server error: {exc}"
    if isinstance(exc, APITimeoutError):
        return ErrorKind.TIMEOUT, f"Anthropic request timed out: {exc}"
    if isinstance(exc, APIConnectionError):
        return ErrorKind.CONNECTION, f"Anthropic connection error: {exc}"
    return ErrorKind.API, f"Anthropic API error: {exc}"


class AnthropicClient(BaseLLMClient):
    provider = Provider.ANTHROPIC

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, timeout_s: float):
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_s)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _convert_messages(messages: List[ChatMessage]) -> Tuple[List[dict], Optional[str]]:
        """Split system messages out: Anthropic takes them as a separate
        top-level parameter, not as a role inside ``messages``."""
        system_parts = [m.content for m in messages if m.role == "system"]
        chat = [m.model_dump() for m in messages if m.role != "system"]
        system = "\n\n".join(system_parts) if system_parts else None
        return chat, system

    def _request_kwargs(self, messages: List[ChatMessage]) -> dict:
        chat, system = self._convert_messages(messages)
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,  # mandatory for Anthropic, unlike OpenAI
            "temperature": self.temperature,
            "messages": chat,
        }
        if system is not None:
            kwargs["system"] = system
        return kwargs

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._request_kwargs(messages))
            content = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            return ModelResponse(provider=self.provider, model=self.model, content=content)
        except APIError as e:
            return self._error_response(*_classify(e))

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        yielded_any = False
        try:
            async with self._client.messages.stream(**self._request_kwargs(messages)) as stream:
                async for text in stream.text_stream:
                    yielded_any = True
                    yield text
        except APIError as e:
            kind, message = _classify(e)
            if not yielded_any:
                raise StreamError(self.provider, kind, message) from e
            yield f"\n[stream interrupted: {message}]"
