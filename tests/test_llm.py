"""Tests for OpenRouter request routing and embedded provider errors."""

from unittest.mock import AsyncMock

import pytest

from jobapply.utils.llm import GemmaChat


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.headers = {}

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        return None


@pytest.mark.asyncio
async def test_openrouter_retries_embedded_provider_overload(monkeypatch):
    client = AsyncMock()
    client.post.side_effect = [
        FakeResponse(
            {
                "choices": [],
                "error": {
                    "message": "provider overloaded",
                    "code": 503,
                    "metadata": {"error_type": "provider_overloaded"},
                },
            }
        ),
        FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]}),
    ]
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)
    sleep = AsyncMock()
    monkeypatch.setattr("jobapply.utils.llm.asyncio.sleep", sleep)

    llm = GemmaChat(api_key="test-key", response_mime_type="application/json")
    response = await llm.ainvoke("return JSON")

    assert response.content == '{"ok": true}'
    assert client.post.await_count == 2
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_openrouter_adds_free_router_as_model_fallback(monkeypatch):
    client = AsyncMock()
    client.post.return_value = FakeResponse(
        {"choices": [{"message": {"content": '{"ok": true}'}}]}
    )
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)

    llm = GemmaChat(api_key="test-key", response_mime_type="application/json")
    await llm.ainvoke("return JSON")

    payload = client.post.await_args.kwargs["json"]
    assert payload["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert payload["models"] == ["openrouter/free"]
