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


@pytest.fixture
def openrouter_settings(monkeypatch):
    from jobapply.settings import Settings

    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        openrouter_model="nvidia/nemotron-3-ultra-550b-a55b:free",
    )
    monkeypatch.setattr("jobapply.utils.llm.get_settings", lambda: settings)
    return settings


@pytest.mark.asyncio
async def test_openrouter_retries_embedded_provider_overload(monkeypatch, openrouter_settings):
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
async def test_openrouter_uses_only_the_configured_model(monkeypatch, openrouter_settings):
    client = AsyncMock()
    client.post.return_value = FakeResponse({"choices": [{"message": {"content": '{"ok": true}'}}]})
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)

    llm = GemmaChat(api_key="test-key", response_mime_type="application/json")
    await llm.ainvoke("return JSON")

    payload = client.post.await_args.kwargs["json"]
    assert payload["model"] == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert "models" not in payload


@pytest.fixture
def anthropic_settings(monkeypatch):
    from jobapply.settings import Settings

    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_model="claude-test-model",
        anthropic_base_url="https://api.anthropic.com/v1",
    )
    monkeypatch.setattr("jobapply.utils.llm.get_settings", lambda: settings)
    return settings


@pytest.mark.asyncio
async def test_anthropic_routes_messages_and_extracts_only_text(monkeypatch, anthropic_settings):
    client = AsyncMock()
    client.post.return_value = FakeResponse(
        {
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
            "stop_reason": "end_turn",
        }
    )
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)
    llm = GemmaChat(api_key="synthetic-key", max_output_tokens=256)
    response = await llm.ainvoke("hello Claude")
    assert response.content == "hello\nworld"
    args = client.post.await_args
    assert args.args[0] == "https://api.anthropic.com/v1/messages"
    assert args.kwargs["headers"]["x-api-key"] == "synthetic-key"
    assert args.kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert args.kwargs["json"] == {
        "model": "claude-test-model",
        "messages": [{"role": "user", "content": "hello Claude"}],
        "max_tokens": 256,
    }


@pytest.mark.asyncio
async def test_anthropic_json_schema_and_full_endpoint(monkeypatch, anthropic_settings):
    import json

    anthropic_settings.anthropic_base_url = "https://api.anthropic.com/v1/messages/"
    client = AsyncMock()
    client.post.return_value = FakeResponse({"content": [{"type": "text", "text": '{"ok": true}'}]})
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    llm = GemmaChat(
        api_key="test-key", response_mime_type="application/json", response_json_schema=schema
    )
    assert (await llm.ainvoke("return JSON")).content == '{"ok": true}'
    args = client.post.await_args
    assert args.args[0] == "https://api.anthropic.com/v1/messages"
    assert json.dumps(schema) in args.kwargs["json"]["system"]
    assert "models" not in args.kwargs["json"]
    assert "response_format" not in args.kwargs["json"]


@pytest.mark.asyncio
async def test_anthropic_retries_overload(monkeypatch, anthropic_settings):
    client = AsyncMock()
    client.post.side_effect = [
        FakeResponse({"error": {"type": "overloaded_error"}}, status_code=529),
        FakeResponse({"content": [{"type": "text", "text": "success"}]}),
    ]
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)
    sleep = AsyncMock()
    monkeypatch.setattr("jobapply.utils.llm.asyncio.sleep", sleep)
    assert (await GemmaChat(api_key="test-key").ainvoke("hello")).content == "success"
    assert client.post.await_count == 2
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.parametrize(
    "data",
    [
        {"content": []},
        {"content": [{"type": "thinking", "thinking": "hidden"}]},
        {"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"},
    ],
)
def test_anthropic_rejects_empty_or_truncated_response(data):
    from jobapply.utils.llm import _extract_anthropic_text

    with pytest.raises(RuntimeError, match="Anthropic"):
        _extract_anthropic_text(data)


def test_anthropic_factory_requires_matching_key(monkeypatch, anthropic_settings):
    from jobapply.utils.llm import get_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JOBAPPLY_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "irrelevant-key")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_llm()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-anthropic-key")
    assert get_llm().api_key == "synthetic-anthropic-key"


def test_anthropic_doctor_and_secret_redaction(monkeypatch, anthropic_settings):
    from jobapply.doctor import check_credentials_configured, check_gemma_model_constraint
    from jobapply.utils.redaction import redact_data, redact_string

    assert check_gemma_model_constraint(anthropic_settings).status == "PASS"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JOBAPPLY_ANTHROPIC_API_KEY", raising=False)
    assert "ANTHROPIC_API_KEY" in check_credentials_configured(anthropic_settings).detail
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-anthropic-secret")
    assert "synthetic-anthropic-secret" not in redact_string("key: synthetic-anthropic-secret")
    assert "synthetic-anthropic-secret" not in str(
        redact_data({"x-api-key": "synthetic-anthropic-secret"})
    )


@pytest.mark.asyncio
async def test_anthropic_doctor_uses_read_only_probe(monkeypatch, anthropic_settings):
    from jobapply import doctor

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-key")
    probe = AsyncMock(return_value=({"data": [{"id": "claude-test-model"}]}, None))
    monkeypatch.setattr(doctor, "_bounded_http_json", probe)
    result = await doctor.check_live_google(anthropic_settings, doctor.default_google_models_check)
    assert result.status == "PASS"
    probe.assert_awaited_once_with(
        "https://api.anthropic.com/v1/models",
        method="GET",
        headers={"x-api-key": "synthetic-key", "anthropic-version": "2023-06-01"},
    )


@pytest.mark.parametrize("provider", ["gemini", "openrouter", "anthropic"])
def test_provider_selects_its_own_model_and_url(monkeypatch, provider):
    from jobapply.settings import Settings

    settings = Settings(
        _env_file=None,
        llm_provider=provider,
        gemini_model="gemini-test-model",
        gemini_base_url="https://gemini.example/v1",
        openrouter_model="openrouter-test-model",
        openrouter_base_url="https://openrouter.example/v1",
        anthropic_model="anthropic-test-model",
        anthropic_base_url="https://anthropic.example/v1",
    )
    assert settings.llm_model == f"{provider}-test-model"
    assert settings.llm_base_url == f"https://{provider}.example/v1"
    monkeypatch.setattr("jobapply.utils.llm.get_settings", lambda: settings)
    llm = GemmaChat(api_key="test-key")
    assert llm.model == settings.llm_model
    assert llm.base_url == settings.llm_base_url


def test_anthropic_requires_model(monkeypatch, anthropic_settings):
    from jobapply.utils.llm import get_llm

    anthropic_settings.anthropic_model = ""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-key")
    with pytest.raises(RuntimeError, match="JOBAPPLY_ANTHROPIC_MODEL"):
        get_llm()


@pytest.mark.asyncio
async def test_anthropic_http_error_includes_redacted_api_reason(monkeypatch, anthropic_settings):
    import httpx

    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        json={"error": {"message": "temperature is deprecated; synthetic-secret"}},
    )
    client = AsyncMock()
    client.post.return_value = response
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)
    with pytest.raises(RuntimeError, match="temperature is deprecated") as exc:
        await GemmaChat(api_key="synthetic-secret").ainvoke("test")
    assert "synthetic-secret" not in str(exc.value)
    assert "temperature" not in client.post.await_args.kwargs["json"]


@pytest.mark.asyncio
async def test_anthropic_can_disable_thinking_for_short_output(monkeypatch, anthropic_settings):
    client = AsyncMock()
    client.post.return_value = FakeResponse(
        {"content": [{"type": "text", "text": "Cover letter"}], "stop_reason": "end_turn"}
    )
    monkeypatch.setattr("jobapply.utils.llm._get_http_client", lambda: client)
    await GemmaChat(api_key="test-key", max_output_tokens=2048, thinking_enabled=False).ainvoke(
        "write letter"
    )
    payload = client.post.await_args.kwargs["json"]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 2048
