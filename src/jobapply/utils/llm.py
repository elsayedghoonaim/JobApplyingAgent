"""Single-model Gemma client for the Google generateContent REST API."""

import asyncio
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from jobapply.settings import get_settings


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
    return min(2.0 ** attempt, 8.0)


def _extract_text(data: dict[str, Any]) -> str:
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


class GemmaChat:
    """Async chat interface backed only by ``gemma-4-31b-it``."""

    def __init__(
        self,
        api_key: str,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        response_mime_type: str | None = None,
        response_json_schema: dict[str, Any] | None = None,
    ):
        settings = get_settings()
        self.model = settings.llm_model
        self.api_key = api_key
        self.base_url = settings.llm_base_url.rstrip("/").removesuffix("/openai")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.response_mime_type = response_mime_type
        self.response_json_schema = response_json_schema

    async def ainvoke(self, prompt: str) -> LLMResponse:
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
        client = _get_http_client()
        for attempt in range(3):
            response = await client.post(url, params={"key": self.api_key}, json=payload)
            if response.status_code not in (429, 503) or attempt == 2:
                response.raise_for_status()
                return LLMResponse(content=_extract_text(response.json()))
            await asyncio.sleep(_retry_delay(response, attempt))
        raise AssertionError("unreachable")


def get_llm(
    temperature: float = 0.3,
    max_output_tokens: int = 1024,
    response_mime_type: str | None = None,
    response_json_schema: dict[str, Any] | None = None,
) -> GemmaChat:
    """Return the project's only configured LLM client."""
    import os

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Gemma requires GOOGLE_API_KEY to be set")
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
