"""Gemini client (google-genai SDK).

Fixes over the hint notebook:
- No more bare ``except Exception`` (notebook bug 04): only the SDK's typed
  ``errors.APIError`` family is caught, classified by HTTP status. A bug in
  our own code now propagates instead of masquerading as a provider error.
"""

from __future__ import annotations

from typing import AsyncGenerator, List, Optional, Tuple

from google import genai
from google.genai import errors, types

from src.base import BaseLLMClient, StreamError
from src.schemas import ChatMessage, ErrorKind, ModelResponse, Provider


def _classify(exc: errors.APIError) -> Tuple[ErrorKind, str]:
    code = getattr(exc, "code", None)
    if isinstance(exc, errors.ServerError):
        return ErrorKind.SERVER, f"Gemini server error: {exc}"
    if code == 429:
        return ErrorKind.RATE_LIMIT, f"Gemini rate limit exceeded: {exc}"
    if code in (401, 403):
        return ErrorKind.AUTH, f"Gemini authentication failed: {exc}"
    if code == 400:
        return ErrorKind.BAD_REQUEST, f"Gemini rejected the request: {exc}"
    return ErrorKind.API, f"Gemini API error: {exc}"


class GeminiClient(BaseLLMClient):
    provider = Provider.GEMINI

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int, timeout_s: float):
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),  # milliseconds
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _convert_messages(
        messages: List[ChatMessage],
    ) -> Tuple[List[types.Content], Optional[str]]:
        """Gemini separates the system prompt and calls the assistant role 'model'."""
        contents: List[types.Content] = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                gemini_role = "model" if m.role == "assistant" else "user"
                contents.append(
                    types.Content(role=gemini_role, parts=[types.Part(text=m.content)])
                )
        return contents, system_instruction

    def _generation_config(self, system_instruction: Optional[str]) -> types.GenerateContentConfig:
        # Note: use a non-thinking model (config.yaml points at a -lite one).
        # Thinking models count their reasoning tokens against max_output_tokens
        # and can stream zero visible text on small budgets.
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            contents, system_instruction = self._convert_messages(messages)
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._generation_config(system_instruction),
            )
            return ModelResponse(
                provider=self.provider, model=self.model, content=response.text or ""
            )
        except errors.APIError as e:
            return self._error_response(*_classify(e))

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        yielded_any = False
        try:
            contents, system_instruction = self._convert_messages(messages)
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._generation_config(system_instruction),
            )
            async for chunk in stream:
                if chunk.text:
                    yielded_any = True
                    yield chunk.text
        except errors.APIError as e:
            kind, message = _classify(e)
            if not yielded_any:
                raise StreamError(self.provider, kind, message) from e
            yield f"\n[stream interrupted: {message}]"
