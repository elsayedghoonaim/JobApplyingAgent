"""Configurable Gemini/OpenRouter LLM client."""

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from jobapply.settings import get_settings
from jobapply.utils.redaction import redact_string


@dataclass
class LLMResponse:
    """Minimal response wrapper used by the workflow nodes."""

    content: str


_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    return _http_client


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), 30.0)
        except ValueError:
            pass
    try:
        for detail in response.json().get("error", {}).get("details", []):
            if str(detail.get("@type", "")).endswith("RetryInfo"):
                raw = str(detail.get("retryDelay", "")).removesuffix("s")
                return min(float(raw), 30.0)
    except (TypeError, ValueError):
        pass
    return min(2.0**attempt, 8.0)


def _extract_text(data: dict[str, Any]) -> str:
    """Extract text from a native Gemini generateContent response."""
    candidates = data.get("candidates") or []
    if not candidates:
        feedback = data.get("promptFeedback") or {}
        raise RuntimeError(f"Gemma returned no candidates: {feedback}")
    parts = candidates[0].get("content", {}).get("parts", [])
    final_parts = [part.get("text", "") for part in parts if not part.get("thought")]
    if not any(final_parts):
        final_parts = [part.get("text", "") for part in parts]
    text = "\n".join(part for part in final_parts if part).strip()
    if not text:
        raise RuntimeError("Gemma returned an empty response")
    return text


def _extract_openrouter_text(data: dict[str, Any]) -> str:
    """Extract text from an OpenAI-compatible OpenRouter response."""
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {data.get('error') or {}}")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    text = str(content or "").strip()
    if not text:
        raise RuntimeError("OpenRouter returned an empty response")
    return text


class GemmaChat:
    """Async chat interface backed by Gemini or OpenRouter."""

    def __init__(
        self,
        api_key: str,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        response_mime_type: str | None = None,
        response_json_schema: dict[str, Any] | None = None,
    ):
        settings = get_settings()
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.fallback_models = settings.llm_fallback_models_list
        self.api_key = api_key
        self.base_url = settings.llm_base_url.rstrip("/").removesuffix("/openai")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.response_mime_type = response_mime_type
        self.response_json_schema = response_json_schema

    async def ainvoke(self, prompt: str) -> LLMResponse:
        if self.provider == "openrouter":
            url = f"{self.base_url.removesuffix('/chat/completions')}/chat/completions"
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            }
            fallbacks = [model for model in self.fallback_models if model != self.model]
            if fallbacks:
                payload["models"] = fallbacks
            if self.response_mime_type == "application/json":
                payload["response_format"] = {"type": "json_object"}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/jobapply",
                "X-Title": "JobApply",
            }
        else:
            url = f"{self.base_url}/models/{quote(self.model, safe='')}:generateContent"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_output_tokens,
                },
            }
            if self.response_mime_type:
                payload["generationConfig"]["responseMimeType"] = self.response_mime_type
            if self.response_json_schema:
                payload["generationConfig"]["responseJsonSchema"] = self.response_json_schema
            headers = {"x-goog-api-key": self.api_key}

        client = _get_http_client()
        for attempt in range(3):
            try:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                    response.raise_for_status()
                    extractor = (
                        _extract_openrouter_text if self.provider == "openrouter" else _extract_text
                    )
                    data = response.json()
                    if self.provider == "openrouter" and not data.get("choices"):
                        error = data.get("error") or {}
                        try:
                            embedded_status = int(error.get("code", 0))
                        except (TypeError, ValueError):
                            embedded_status = 0
                        if embedded_status in (429, 500, 502, 503, 504) and attempt < 2:
                            await asyncio.sleep(min(2.0**attempt, 8.0))
                            continue
                    return LLMResponse(content=extractor(data))
                await asyncio.sleep(_retry_delay(response, attempt))
            except httpx.HTTPStatusError as exc:
                msg = redact_string(str(exc), extra_secrets=[self.api_key])
                provider_name = "OpenRouter" if self.provider == "openrouter" else "Gemini"
                raise RuntimeError(f"{provider_name} API HTTP error: {msg}") from None
            except Exception as exc:
                if attempt == 2 or not isinstance(
                    exc, (httpx.TransportError, httpx.TimeoutException)
                ):
                    msg = redact_string(str(exc), extra_secrets=[self.api_key])
                    provider_name = "OpenRouter" if self.provider == "openrouter" else "Gemini"
                    raise RuntimeError(f"{provider_name} API request failed: {msg}") from None
                await asyncio.sleep(min(2.0**attempt, 8.0))
        raise AssertionError("unreachable")


def get_llm(
    temperature: float = 0.3,
    max_output_tokens: int = 1024,
    response_mime_type: str | None = None,
    response_json_schema: dict[str, Any] | None = None,
) -> GemmaChat:
    """Return the configured Gemini or OpenRouter client."""
    import os

    settings = get_settings()
    key_name = "OPENROUTER_API_KEY" if settings.llm_provider == "openrouter" else "GOOGLE_API_KEY"
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"{settings.llm_provider} requires {key_name} to be set")
    return GemmaChat(
        api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
        response_json_schema=response_json_schema,
    )


async def close_llm_client() -> None:
    """Close the shared HTTP connection pool at application shutdown."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None
