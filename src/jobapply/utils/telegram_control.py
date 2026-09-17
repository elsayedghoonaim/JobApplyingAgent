"""Persistent Telegram supervisor and submitted-jobs reporting."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta, timezone, tzinfo
from html import escape
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from jobapply.models.application import AttemptStatus
from jobapply.settings import get_settings
from jobapply.utils.mongo import MongoClientManager
from jobapply.utils.observability import log_event
from jobapply.utils.telegram import (
    TelegramClient,
    set_control_command_handler,
)

WorkflowRunner = Callable[[], Coroutine[Any, Any, None]]


def report_timezone(name: str) -> tzinfo:
    """Resolve an IANA timezone, falling back to the configured machine timezone."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now().astimezone().tzinfo or timezone.utc


def today_utc_bounds(
    *,
    now: datetime | None = None,
    timezone_name: str = "Africa/Cairo",
) -> tuple[datetime, datetime, datetime]:
    """Return local now and the UTC half-open bounds for its calendar day."""
    zone = report_timezone(timezone_name)
    local_now = now or datetime.now(zone)
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=zone)
    else:
        local_now = local_now.astimezone(zone)
    local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    local_end = local_start + timedelta(days=1)
    return local_now, local_start.astimezone(timezone.utc), local_end.astimezone(timezone.utc)


class SubmittedJobsReport:
    """Read-only report query over confirmed application attempts."""

    def __init__(self, *, settings: Any | None = None, collection: Any | None = None):
        self.settings = settings or get_settings()
        if collection is None:
            client = MongoClientManager.get_client()
            collection = client[self.settings.mongodb_db][
                self.settings.application_attempts_collection
            ]
        self.collection = collection

    async def get_today(self, now: datetime | None = None) -> tuple[datetime, list[dict]]:
        local_now, start_utc, end_utc = today_utc_bounds(
            now=now,
            timezone_name=self.settings.report_timezone,
        )
        cursor = self.collection.find(
            {
                "status": AttemptStatus.SUBMITTED.value,
                "submitted_at": {"$gte": start_utc, "$lt": end_utc},
            },
            {
                "_id": 0,
                "job_id": 1,
                "submitted_at": 1,
                "metadata.title": 1,
                "metadata.company": 1,
                "metadata.url": 1,
            },
        )
        if hasattr(cursor, "sort"):
            cursor = cursor.sort("submitted_at", 1)
        if hasattr(cursor, "to_list"):
            result = cursor.to_list(length=1000)
            documents = await result if inspect.isawaitable(result) else result
        else:
            documents = list(cursor)
        return local_now, list(documents or [])

    def format_messages(self, local_now: datetime, jobs: list[dict]) -> list[str]:
        """Format all jobs into bounded standalone Telegram HTML messages."""
        header = (
            "<b>📊 TODAY'S SUBMITTED JOBS</b>\n"
            f"Date: {local_now.strftime('%d %B %Y')}\n"
            f"Total: {len(jobs)}"
        )
        if not jobs:
            return [header + "\n\nNo confirmed applications were submitted today."]

        zone = report_timezone(self.settings.report_timezone)
        blocks: list[str] = []
        for index, job in enumerate(jobs, start=1):
            metadata = job.get("metadata") or {}
            submitted_at = job.get("submitted_at")
            if isinstance(submitted_at, datetime):
                if submitted_at.tzinfo is None:
                    submitted_at = submitted_at.replace(tzinfo=timezone.utc)
                time_label = submitted_at.astimezone(zone).strftime("%H:%M")
            else:
                time_label = "Unknown"
            title = escape(str(metadata.get("title") or f"Job {job.get('job_id', 'unknown')}"))
            company = escape(str(metadata.get("company") or "Unknown company"))
            url = escape(str(metadata.get("url") or ""))
            block = (
                f"{index}. <b>{title}</b>\n"
                f"Company: {company}\n"
                f"Submitted: {time_label}"
            )
            if url:
                block += f"\n{url}"
            blocks.append(block)

        messages: list[str] = []
        current = header
        for block in blocks:
            candidate = f"{current}\n\n{block}"
            if len(candidate) > 3500 and current != header:
                messages.append(current)
                current = f"<b>📊 TODAY'S SUBMITTED JOBS — CONTINUED</b>\n\n{block}"
            else:
                current = candidate
        messages.append(current)
        return messages


class TelegramControlService:
    """Keep Telegram commands available while workflows start, stop, and finish."""

    def __init__(
        self,
        runner: WorkflowRunner,
        *,
        client_factory: Callable[[], TelegramClient] = TelegramClient,
        report_factory: Callable[[], SubmittedJobsReport] = SubmittedJobsReport,
    ):
        self.runner = runner
        self.client_factory = client_factory
        self.report_factory = report_factory
        self.workflow_task: asyncio.Task[None] | None = None
        self._last_announced_task: asyncio.Task[None] | None = None

    def is_running(self) -> bool:
        return self.workflow_task is not None and not self.workflow_task.done()

    async def _send(self, text: str, *, html: bool = False) -> None:
        await self.client_factory().send_message(text, parse_mode="HTML" if html else None)

    async def start_workflow(self, *, announce: bool = True) -> bool:
        if self.is_running():
            if announce:
                await self._send("The agent is already running.")
            return False
        self.workflow_task = asyncio.create_task(self.runner(), name="jobapply-telegram-run")
        self._last_announced_task = None
        if announce:
            await self._send("▶️ Agent started. Use /stop to stop it safely.")
        return True

    async def stop_workflow(self) -> bool:
        if not self.is_running():
            await self._send("The agent is already stopped. Use /run to start it.")
            return False
        task = self.workflow_task
        if task is None:
            return False
        await self._send("⏹ Stop requested. Saving state and stopping safely.")
        asyncio.get_running_loop().call_later(0.2, task.cancel)
        return True

    async def send_report(self) -> None:
        report = self.report_factory()
        local_now, jobs = await report.get_today()
        for message in report.format_messages(local_now, jobs):
            await self._send(message, html=True)

    async def send_status(self) -> None:
        status = "running" if self.is_running() else "stopped"
        await self._send(
            f"<b>AGENT STATUS</b>\nStatus: {status}\n\n"
            "Commands: /report · /stop · /run · /status",
            html=True,
        )

    async def handle_command(self, command: str) -> None:
        if command == "/report":
            await self.send_report()
        elif command == "/stop":
            await self.stop_workflow()
        elif command == "/run":
            await self.start_workflow()
        elif command == "/status":
            await self.send_status()

    async def _announce_finished_task(self) -> None:
        task = self.workflow_task
        if task is None or not task.done() or task is self._last_announced_task:
            return
        self._last_announced_task = task
        if task.cancelled():
            await self._send("⏹ Agent stopped. Use /run when you want to start again.")
        else:
            error = task.exception()
            if error is None:
                await self._send("✅ Agent run finished. The controller is still online.")
            else:
                log_event(
                    "error",
                    "telegram_control.workflow_failed",
                    "Telegram-controlled workflow failed.",
                    node="telegram_control",
                    exc=error,
                )
                await self._send(
                    "⚠️ Agent run ended with an error. Use /run to start a new run."
                )
        self.workflow_task = None

    async def serve(self, *, start_immediately: bool = True) -> None:
        """Run until locally interrupted; the controller remains alive between runs."""
        set_control_command_handler(self.handle_command)
        try:
            await self._send(
                "<b>🎛 TELEGRAM CONTROL ONLINE</b>\n"
                "/report — today's submitted jobs\n"
                "/stop — stop the active agent safely\n"
                "/run — start a new agent run\n"
                "/status — show current status",
                html=True,
            )
            if start_immediately:
                await self.start_workflow(announce=False)
            while True:
                await self._announce_finished_task()
                try:
                    await self.client_factory().poll_control_commands_once(
                        self.handle_command,
                        timeout=2,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log_event(
                        "warning",
                        "telegram_control.poll_failed",
                        "Telegram controller poll failed; retrying.",
                        node="telegram_control",
                        exc=exc,
                    )
                    await asyncio.sleep(1)
                await asyncio.sleep(0.1)
        finally:
            set_control_command_handler(None)
            task = self.workflow_task
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
