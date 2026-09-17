import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jobapply.utils.telegram import TelegramClient, extract_control_command
from jobapply.utils.telegram_control import (
    SubmittedJobsReport,
    TelegramControlService,
    today_utc_bounds,
)


def test_control_commands_require_configured_chat_and_normalize_bot_suffix():
    assert (
        extract_control_command(
            {"chat": {"id": "42"}, "text": "/REPORT@JobApplyBot now"},
            "42",
        )
        == "/report"
    )
    assert extract_control_command({"chat": {"id": "7"}, "text": "/stop"}, "42") is None
    assert extract_control_command({"chat": {"id": "42"}, "text": "stop"}, "42") is None
    assert extract_control_command({"chat": {"id": "42"}, "text": "/unknown"}, "42") is None


@pytest.mark.asyncio
async def test_controller_poll_dispatches_authorized_commands_and_advances_cursor(monkeypatch):
    repo = SimpleNamespace(
        claim_poll_lease=AsyncMock(return_value="lease-1"),
        get_cursor=AsyncMock(return_value=10),
        save_cursor=AsyncMock(return_value=True),
        release_poll_lease=AsyncMock(return_value=True),
    )
    client = TelegramClient.__new__(TelegramClient)
    client.settings = SimpleNamespace(
        telegram_poll_lease_seconds=30,
        telegram_chat_id="42",
        telegram_bot_token="test-token",
    )
    client.telegram_repo = repo
    client.bot_chat_key = "bot-chat"
    client._last_update_id = None
    client._base_url = "https://api.telegram.test/bottest-token"

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "ok": True,
            "result": [
                {"update_id": 11, "message": {"chat": {"id": "42"}, "text": "/report"}},
                {"update_id": 12, "message": {"chat": {"id": "7"}, "text": "/stop"}},
            ],
        },
    )

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            assert json["offset"] == 11
            return response

    monkeypatch.setattr(
        "jobapply.utils.telegram.httpx.AsyncClient",
        lambda *args, **kwargs: FakeHttpClient(),
    )
    handler = AsyncMock()

    handled = await client.poll_control_commands_once(handler, timeout=0)

    assert handled == 1
    handler.assert_awaited_once_with("/report")
    assert repo.save_cursor.await_args_list[-1].args == ("bot-chat", 12, "lease-1")
    repo.release_poll_lease.assert_awaited_once_with("bot-chat", "lease-1")


@pytest.mark.asyncio
async def test_controller_poll_yields_while_application_question_needs_lease(monkeypatch):
    repo = SimpleNamespace(claim_poll_lease=AsyncMock(return_value="lease-1"))
    client = TelegramClient.__new__(TelegramClient)
    client.telegram_repo = repo
    monkeypatch.setattr("jobapply.utils.telegram._priority_poll_waiters", 1)

    handled = await client.poll_control_commands_once(AsyncMock(), timeout=0)

    assert handled == 0
    repo.claim_poll_lease.assert_not_awaited()


def test_today_bounds_use_cairo_calendar_day():
    local_now, start_utc, end_utc = today_utc_bounds(
        now=datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
        timezone_name="Africa/Cairo",
    )
    assert local_now.date().isoformat() == "2026-08-31"
    assert end_utc - start_utc == timedelta(days=1)
    assert start_utc < datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc) < end_utc


class _Cursor:
    def __init__(self, docs):
        self.docs = docs

    def sort(self, key, direction):
        self.docs.sort(key=lambda item: item[key])
        return self

    async def to_list(self, length):
        return self.docs[:length]


class _Collection:
    def __init__(self, docs):
        self.docs = docs
        self.query = None

    def find(self, query, projection):
        self.query = query
        return _Cursor(list(self.docs))


@pytest.mark.asyncio
async def test_report_queries_confirmed_submissions_and_formats_every_job():
    docs = [
        {
            "job_id": "1",
            "submitted_at": datetime(2026, 8, 31, 9, 30, tzinfo=timezone.utc),
            "metadata": {
                "title": "ML <Engineer>",
                "company": "A & B",
                "url": "https://example.test/1?a=1&b=2",
            },
        },
        {
            "job_id": "2",
            "submitted_at": datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc),
            "metadata": {"title": "AI Engineer", "company": "Acme"},
        },
    ]
    collection = _Collection(docs)
    settings = SimpleNamespace(
        report_timezone="Africa/Cairo",
        mongodb_db="jobapply",
        application_attempts_collection="application_attempts",
    )
    report = SubmittedJobsReport(settings=settings, collection=collection)

    local_now, jobs = await report.get_today(datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc))
    messages = report.format_messages(local_now, jobs)

    assert collection.query["status"] == "submitted"
    assert "$gte" in collection.query["submitted_at"]
    assert "$lt" in collection.query["submitted_at"]
    rendered = "\n".join(messages)
    assert "Total: 2" in rendered
    assert "ML &lt;Engineer&gt;" in rendered
    assert "A &amp; B" in rendered
    assert "AI Engineer" in rendered


class _FakeClient:
    def __init__(self):
        self.send_message = AsyncMock()


@pytest.mark.asyncio
async def test_controller_run_status_stop_and_restart():
    client = _FakeClient()
    started = asyncio.Event()

    async def runner():
        started.set()
        await asyncio.Event().wait()

    service = TelegramControlService(runner, client_factory=lambda: client)
    assert await service.start_workflow()
    await asyncio.wait_for(started.wait(), timeout=1)
    assert service.is_running()

    await service.handle_command("/status")
    assert "running" in client.send_message.await_args.args[0]

    assert await service.stop_workflow()
    await asyncio.sleep(0.3)
    assert service.workflow_task.cancelled()
    await service._announce_finished_task()
    assert not service.is_running()

    started.clear()
    await service.handle_command("/run")
    await asyncio.wait_for(started.wait(), timeout=1)
    assert service.is_running()
    service.workflow_task.cancel()
    await asyncio.gather(service.workflow_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_controller_report_sends_html():
    client = _FakeClient()
    fake_report = SimpleNamespace(
        get_today=AsyncMock(
            return_value=(datetime(2026, 8, 31, tzinfo=timezone.utc), [{"job_id": "1"}])
        ),
        format_messages=lambda now, jobs: ["<b>REPORT</b>"],
    )
    service = TelegramControlService(
        AsyncMock(),
        client_factory=lambda: client,
        report_factory=lambda: fake_report,
    )

    await service.handle_command("/report")

    client.send_message.assert_awaited_once_with("<b>REPORT</b>", parse_mode="HTML")
