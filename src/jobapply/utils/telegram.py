"""Telegram client with checked delivery, durable correlated replies, and outbox recovery."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Optional

import httpx
from langsmith.run_helpers import trace

from jobapply.models.telegram import (
    CorrelationStatus,
    CorrelationWaitResult,
    OutboxDeliveryResult,
    OutboxDrainResult,
    OutboxStatus,
)
from jobapply.settings import get_settings
from jobapply.utils.observability import log_event
from jobapply.utils.redaction import redact_string
from jobapply.utils.telegram_storage import (
    CALLBACK_DATA_PREFIX,
    MAX_INLINE_ACTIONS,
    NotificationOutboxRepository,
    TelegramApiDefiniteRejectError,
    TelegramRepository,
    TelegramTransportAmbiguousError,
    build_inline_keyboard,
    make_bot_chat_key,
)
from jobapply.utils.tracing import get_telegram_metadata

MAX_CALLBACK_ANSWER_LENGTH = 200
CONTROL_COMMANDS = frozenset({"/report", "/run", "/stop", "/status"})
ControlCommandHandler = Callable[[str], Awaitable[None]]
_control_command_handler: ControlCommandHandler | None = None
_priority_poll_waiters = 0


def set_control_command_handler(handler: ControlCommandHandler | None) -> None:
    """Register the in-process Telegram supervisor command handler."""
    global _control_command_handler
    _control_command_handler = handler


def extract_control_command(message: dict, expected_chat_id: str) -> str | None:
    """Return one authorized normalized bot command from a Telegram message."""
    if str(message.get("chat", {}).get("id", "")) != str(expected_chat_id):
        return None
    text = str(message.get("text") or "").strip()
    if not text.startswith("/"):
        return None
    token = text.split(maxsplit=1)[0].casefold()
    command = token.split("@", 1)[0]
    return command if command in CONTROL_COMMANDS else None


def correlated_reply_text(
    message: dict,
    configured_chat_id: str,
    nonce: str,
    reply_to_message_id: int | None,
) -> str | None:
    """Return text for a new same-chat message or explicit correlated reply.

    Rejects stale messages older than the prompt, cross-prompt replies, and mismatched chat IDs.
    """
    chat_id = str(message.get("chat", {}).get("id", ""))
    if chat_id != str(configured_chat_id):
        return None

    text = str(message.get("text") or "")
    if not text:
        return None

    msg_id = message.get("message_id")
    replied_to = message.get("reply_to_message", {}).get("message_id")

    # Stale message check: message older than or equal to prompt cannot correlate
    if (
        reply_to_message_id is not None
        and isinstance(msg_id, int)
        and msg_id <= reply_to_message_id
    ):
        return None

    is_direct_reply = reply_to_message_id is not None and replied_to == reply_to_message_id

    # If replying to a different message (not our prompt), do not bind to avoid collision
    if (
        replied_to is not None
        and reply_to_message_id is not None
        and replied_to != reply_to_message_id
    ):
        return None

    is_new_chat_message = (
        reply_to_message_id is not None and isinstance(msg_id, int) and msg_id > reply_to_message_id
    )
    has_nonce = bool(nonce and nonce in text)

    # Standalone nonce check when prompt_message_id is not set
    if reply_to_message_id is None:
        if not has_nonce:
            return None
        return text.replace(nonce, "").strip(" []:-")

    if is_direct_reply:
        return text.replace(nonce, "").strip(" []:-") if has_nonce else text.strip()

    if is_new_chat_message:
        if has_nonce:
            return text.replace(nonce, "").strip(" []:-")
        if replied_to is None:
            # Plain next message under single active waiter
            return text.strip()

    return None


class TelegramClient:
    """Telegram Bot API client with durable cursors, correlations, and outbox."""

    def __init__(self):
        self.settings = get_settings()
        self._last_update_id: int | None = None
        self._base_url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}"
        self.telegram_repo = TelegramRepository()
        self.outbox_repo = NotificationOutboxRepository()
        self.bot_chat_key = make_bot_chat_key(
            self.settings.telegram_bot_token, self.settings.telegram_chat_id
        )

    async def send_message(
        self, text: str, parse_mode: str | None = None, reply_markup: dict | None = None
    ) -> int | None:
        """Send checked messages and return the last Telegram message ID.

        Inline keyboards ride on the first chunk so buttons stay attached to
        the correlated prompt itself.
        """
        async with trace(
            "telegram_send_message",
            run_type="tool",
            metadata={"message_length": len(text), "parse_mode": parse_mode or "plain"},
        ) as run_tree:
            chunks = [text[index : index + 4000] for index in range(0, len(text), 4000)] or [""]
            message_id = None
            for chunk_index, chunk in enumerate(chunks):
                message_id = await self._send_single_message(
                    chunk,
                    parse_mode,
                    reply_markup if chunk_index == 0 else None,
                )
                if len(chunks) > 1:
                    await asyncio.sleep(0.5)
            if run_tree:
                run_tree.metadata.update(get_telegram_metadata(message_sent=True))
                run_tree.metadata["chunks_sent"] = len(chunks)
            return message_id

    async def _send_single_message(
        self, text: str, parse_mode: str | None, reply_markup: dict | None = None
    ) -> int | None:
        payload: dict[str, Any] = {"chat_id": self.settings.telegram_chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup
        token = self.settings.telegram_bot_token
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(f"{self._base_url}/sendMessage", json=payload)
                response.raise_for_status()
                data = response.json()
                if not data.get("ok"):
                    desc = redact_string(
                        str(data.get("description", "unknown error")),
                        extra_secrets=[token],
                    )
                    error_code = data.get("error_code")
                    if isinstance(error_code, int) and 400 <= error_code < 500:
                        raise TelegramApiDefiniteRejectError(
                            f"Telegram sendMessage rejected: {desc}"
                        )
                    raise TelegramTransportAmbiguousError(f"Telegram sendMessage error: {desc}")
                return data.get("result", {}).get("message_id")
        except httpx.HTTPStatusError as http_err:
            sanitized = redact_string(str(http_err), extra_secrets=[token])
            if http_err.response is not None and 400 <= http_err.response.status_code < 500:
                raise TelegramApiDefiniteRejectError(
                    f"Telegram sendMessage client error: {sanitized}"
                ) from None
            raise TelegramTransportAmbiguousError(
                f"Telegram sendMessage transport error: {sanitized}"
            ) from None
        except TelegramApiDefiniteRejectError:
            raise
        except Exception as exc:
            sanitized = redact_string(str(exc), extra_secrets=[token])
            raise TelegramTransportAmbiguousError(
                f"Telegram sendMessage transport error: {sanitized}"
            ) from None

    async def poll_control_commands_once(
        self,
        handler: ControlCommandHandler,
        *,
        timeout: int = 2,
    ) -> int:
        """Poll and dispatch authorized controller commands under the durable cursor lease."""
        # Form Q&A is latency-sensitive.  Once a question waiter announces that
        # it needs the single Telegram getUpdates lease, do not let the
        # background command poller reacquire and starve it between short polls.
        if _priority_poll_waiters:
            return 0
        lease = await self.telegram_repo.claim_poll_lease(
            self.bot_chat_key,
            lease_duration_seconds=max(timeout + 5, self.settings.telegram_poll_lease_seconds),
        )
        if not lease:
            return 0
        handled = 0
        token = self.settings.telegram_bot_token
        try:
            durable_cursor = await self.telegram_repo.get_cursor(self.bot_chat_key)
            if durable_cursor is not None:
                self._last_update_id = max(self._last_update_id or 0, durable_cursor)
            offset = 0 if self._last_update_id is None else self._last_update_id + 1
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 5.0, connect=5.0)) as client:
                try:
                    response = await client.post(
                        f"{self._base_url}/getUpdates",
                        json={"offset": offset, "timeout": max(0, timeout)},
                    )
                except httpx.ReadTimeout:
                    # An idle Telegram long poll can reach the local read
                    # timeout without indicating a controller failure.
                    return 0
                response.raise_for_status()
                data = response.json()
                if not data.get("ok"):
                    desc = redact_string(
                        str(data.get("description", "unknown error")),
                        extra_secrets=[token],
                    )
                    raise RuntimeError(f"Telegram getUpdates failed: {desc}")
                for update in data.get("result", []):
                    update_id = int(update.get("update_id", 0))
                    command = extract_control_command(
                        update.get("message", {}),
                        self.settings.telegram_chat_id,
                    )
                    if command:
                        await handler(command)
                        handled += 1
                    saved = await self.telegram_repo.save_cursor(
                        self.bot_chat_key, update_id, lease
                    )
                    if not saved:
                        raise RuntimeError("Telegram controller cursor save failed")
                    self._last_update_id = max(self._last_update_id or 0, update_id)
            return handled
        finally:
            await self.telegram_repo.release_poll_lease(self.bot_chat_key, lease)

    async def wait_for_correlated_reply(
        self,
        nonce: str,
        timeout: int = 300,
        reply_to_message_id: int | None = None,
    ) -> Optional[str]:
        """Wait for a direct Telegram reply with durable cursor progression and periodic lease renewal."""
        async with trace(
            "telegram_wait_for_reply",
            run_type="tool",
            metadata=get_telegram_metadata(nonce=nonce, timeout=timeout),
        ) as run_tree:
            durable_cursor = await self.telegram_repo.get_cursor(self.bot_chat_key)
            if durable_cursor is not None:
                self._last_update_id = max(self._last_update_id or 0, durable_cursor)

            lease_duration = max(timeout + 30, self.settings.telegram_poll_lease_seconds)
            poll_lease = await self.telegram_repo.claim_poll_lease(
                self.bot_chat_key, lease_duration_seconds=lease_duration
            )
            if not poll_lease:
                return None

            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            token = self.settings.telegram_bot_token
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                    while loop.time() < deadline:
                        # Renew poll lease on each iteration
                        renewed = await self.telegram_repo.renew_poll_lease(
                            self.bot_chat_key, poll_lease, lease_duration_seconds=lease_duration
                        )
                        if not renewed:
                            await self.telegram_repo.release_poll_lease(
                                self.bot_chat_key, poll_lease
                            )
                            return None

                        poll_timeout = max(1, min(5, int(deadline - loop.time())))
                        offset = 0 if self._last_update_id is None else self._last_update_id + 1
                        response = await client.post(
                            f"{self._base_url}/getUpdates",
                            json={"offset": offset, "timeout": poll_timeout},
                        )
                        response.raise_for_status()
                        data = response.json()
                        if not data.get("ok"):
                            desc = redact_string(
                                str(data.get("description", "unknown error")),
                                extra_secrets=[token],
                            )
                            raise RuntimeError(f"Telegram getUpdates failed: {desc}")
                        for update in data.get("result", []):
                            up_id = update.get("update_id", 0)
                            saved = await self.telegram_repo.save_cursor(
                                self.bot_chat_key, up_id, lease_id=poll_lease
                            )
                            if saved:
                                self._last_update_id = max(self._last_update_id or 0, up_id)

                            reply = correlated_reply_text(
                                update.get("message", {}),
                                self.settings.telegram_chat_id,
                                nonce,
                                reply_to_message_id,
                            )
                            if reply is not None:
                                await self.telegram_repo.release_poll_lease(
                                    self.bot_chat_key, poll_lease
                                )
                                if run_tree:
                                    run_tree.metadata.update(
                                        get_telegram_metadata(
                                            nonce=nonce, timeout=timeout, timed_out=False
                                        )
                                    )
                                return reply
            except Exception as exc:
                await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
                sanitized = redact_string(str(exc), extra_secrets=[token])
                raise RuntimeError(f"Telegram getUpdates error: {sanitized}") from None

            await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
            if run_tree:
                run_tree.metadata.update(
                    get_telegram_metadata(nonce=nonce, timeout=timeout, timed_out=True)
                )
            return None

    async def _answer_callback_query(self, callback_query_id: str, text: str = "") -> bool:
        """Acknowledge one callback after durable acceptance; best-effort and bounded.

        Failures degrade into a bounded structured warning: the reply is already
        durable, so an ack failure can never undo acceptance.
        """
        token = self.settings.telegram_bot_token
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                response = await client.post(
                    f"{self._base_url}/answerCallbackQuery",
                    json={
                        "callback_query_id": callback_query_id,
                        "text": redact_string(str(text or ""))[:MAX_CALLBACK_ANSWER_LENGTH],
                    },
                )
                response.raise_for_status()
                data = response.json()
                if not data.get("ok"):
                    desc = redact_string(
                        str(data.get("description", "unknown error")),
                        extra_secrets=[token],
                    )
                    log_event(
                        "warning",
                        "telegram.answer_callback_rejected",
                        "answerCallbackQuery rejected; durable reply unaffected.",
                        node="telegram_client",
                        details={"reason": desc[:120]},
                    )
                    return False
            return True
        except Exception as exc:
            sanitized = redact_string(str(exc), extra_secrets=[token])
            log_event(
                "warning",
                "telegram.answer_callback_failed",
                "answerCallbackQuery failed; durable reply unaffected.",
                node="telegram_client",
                details={"reason": sanitized[:120]},
            )
            return False

    def callback_data_is_for_current_prompt(
        self,
        callback_update: dict,
        configured_chat_id: str,
        expected_nonce: str,
        prompt_message_id: int | None,
    ) -> tuple[bool, Optional[str], Optional[int]]:
        """Validate a raw callback_query update against the active correlation.

        Enforces the exact opaque codec prefix, a bounded nonnegative option
        index, matching chat/nonce/prompt. Any mismatch never satisfies the
        current waiter.
        """
        message = callback_update.get("message") or {}
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != str(configured_chat_id):
            return False, None, None
        data = callback_update.get("data")
        parts = str(data or "").split("|")
        if len(parts) != 3 or parts[0] != CALLBACK_DATA_PREFIX:
            return False, None, None
        nonce = parts[1]
        if not nonce or nonce != expected_nonce:
            return False, None, None
        try:
            index = int(parts[2])
        except ValueError:
            return False, None, None
        if index < 0 or index > MAX_INLINE_ACTIONS:
            return False, None, None
        if prompt_message_id is None or message.get("message_id") != prompt_message_id:
            return False, None, None
        return True, str(callback_update.get("id") or ""), index

    @staticmethod
    def _bounded_buttoned_prompt(text: str, nonce: str, max_len: int = 3800) -> str:
        """Fit one buttoned correlation into a single Telegram transport message.

        The durable nonce reference is preserved so replies stay correlatable
        even when long prompt bodies are truncated.
        """
        body = str(text or "")
        if len(body) <= max_len:
            return body
        truncated = body[:max_len]
        if nonce and nonce in truncated:
            return truncated
        reserve = len(nonce) + 16 if nonce else 0
        cut = max(0, max_len - reserve)
        trimmed = truncated[:cut].rstrip()
        if nonce:
            trimmed = f"{trimmed}\n\n[Ref: `{nonce}`]"
        return trimmed

    async def send_and_wait_for_reply(
        self,
        correlation_key: str,
        purpose: str,
        run_id: str,
        prompt_text: str,
        timeout: int = 300,
        job_id: str | None = None,
        metadata: dict | None = None,
        inline_actions: list[str] | None = None,
    ) -> CorrelationWaitResult:
        """Durable, restart-safe prompt sending and correlated reply waiting with lease renewal.

        When ``inline_actions`` is provided (bounded small option lists), the
        prompt carries a Telegram inline keyboard whose opaque callback data is
        correlated to this exact prompt/nonce. Ordinary text replies remain
        fully accepted alongside buttons. If the buttoned prompt cannot be
        delivered, existing fail-closed infrastructure behavior applies and no
        second ambiguous prompt is ever sent automatically.
        """
        # 1. Register correlation intent in MongoDB
        try:
            reg = await self.telegram_repo.register_intent(
                correlation_key=correlation_key,
                purpose=purpose,
                run_id=run_id,
                prompt_text=prompt_text,
                job_id=job_id,
                metadata=metadata,
                inline_actions=inline_actions,
            )
        except Exception as reg_exc:
            return CorrelationWaitResult(
                status=CorrelationStatus.STORAGE_ERROR,
                error_reason=f"Failed to register intent: {type(reg_exc).__name__}",
            )

        if not reg.registered:
            return CorrelationWaitResult(
                status=CorrelationStatus.STORAGE_ERROR,
                error_reason=reg.reason or "registration_rejected",
                nonce=reg.nonce,
            )

        # 2. Check terminal / cached states
        if reg.status == CorrelationStatus.CONSUMED:
            return CorrelationWaitResult(
                status=CorrelationStatus.CONSUMED,
                nonce=reg.nonce,
            )
        if reg.status == CorrelationStatus.TIMED_OUT:
            return CorrelationWaitResult(
                status=CorrelationStatus.TIMED_OUT,
                timed_out=True,
                nonce=reg.nonce,
            )
        if reg.status == CorrelationStatus.PROMPT_DELIVERY_UNKNOWN:
            return CorrelationWaitResult(
                status=CorrelationStatus.PROMPT_DELIVERY_UNKNOWN,
                error_reason="prompt_delivery_unknown",
                nonce=reg.nonce,
            )
        if reg.status == CorrelationStatus.PROMPT_FAILED:
            return CorrelationWaitResult(
                status=CorrelationStatus.PROMPT_FAILED,
                error_reason="prompt_previously_rejected",
                nonce=reg.nonce,
            )

        # 3. Check if already replied
        if reg.status == CorrelationStatus.REPLIED and reg.reply_text is not None:
            lease = await self.telegram_repo.claim_waiter(
                correlation_key, lease_duration_seconds=30
            )
            if lease.acquired and lease.lease_id:
                # Reply metadata (kind/index) lives on the durable document.
                corr_doc = await self.telegram_repo.get_correlation(correlation_key)
                if corr_doc is not None and corr_doc.reply_kind == "callback":
                    # Crash recovery for a durably accepted button press: finish
                    # cursor/acknowledge recovery under exact leases before
                    # consuming, preserving reply → cursor → ack → consume.
                    recovered = await self._recover_cached_callback_reply(
                        correlation_key=correlation_key,
                        reg=reg,
                        waiter_lease_id=lease.lease_id,
                    )
                    return recovered
                consume_res = await self.telegram_repo.consume_reply(
                    correlation_key, lease.lease_id
                )
                if consume_res.consumed:
                    return CorrelationWaitResult(
                        status=CorrelationStatus.REPLIED,
                        reply_text=consume_res.reply_text,
                        nonce=reg.nonce,
                    )

        # 4. Acquire waiter lease AND bot/chat polling lease BEFORE sending any new prompt
        lease_duration = max(timeout + 30, self.settings.telegram_poll_lease_seconds)
        waiter_lease = await self.telegram_repo.claim_waiter(
            correlation_key, lease_duration_seconds=lease_duration
        )
        if not waiter_lease.acquired or not waiter_lease.lease_id:
            return CorrelationWaitResult(
                status=waiter_lease.status,
                error_reason=waiter_lease.reason or "cannot_acquire_waiter_lease",
                nonce=reg.nonce,
            )

        global _priority_poll_waiters
        _priority_poll_waiters += 1
        try:
            poll_lease = None
            for _ in range(100):
                poll_lease = await self.telegram_repo.claim_poll_lease(
                    self.bot_chat_key, lease_duration_seconds=lease_duration
                )
                if poll_lease:
                    break
                await asyncio.sleep(0.1)
        finally:
            _priority_poll_waiters = max(0, _priority_poll_waiters - 1)
        if not poll_lease:
            await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
            return CorrelationWaitResult(
                status=CorrelationStatus.STORAGE_ERROR,
                error_reason="concurrent_bot_poller_active",
                nonce=reg.nonce,
            )

        # 5. Check if prompt already sent; if not, mark in-flight and dispatch
        if reg.prompt_message_id is not None:
            prompt_message_id = reg.prompt_message_id
        else:
            in_flight_marked = await self.telegram_repo.mark_prompt_in_flight(
                correlation_key, waiter_lease.lease_id
            )
            if not in_flight_marked:
                await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
                await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
                return CorrelationWaitResult(
                    status=CorrelationStatus.STORAGE_ERROR,
                    error_reason="cannot_mark_prompt_in_flight",
                    nonce=reg.nonce,
                )

            # One message carries text and buttons together; button delivery
            # failure is fail-closed, never a silent second plain-text prompt.
            # Buttoned correlations are always a single transport message so the
            # recorded prompt message ID is exactly the message whose buttons
            # callbacks reference.
            reply_markup = (
                {"inline_keyboard": build_inline_keyboard(reg.nonce, reg.inline_actions)}
                if reg.inline_actions
                else None
            )
            try:
                structured_parse_mode = (
                    "HTML" if purpose in {"form_qa", "approval"} else None
                )
                if reg.inline_actions:
                    buttoned_text = self._bounded_buttoned_prompt(reg.prompt_text, reg.nonce)
                    sent_id = await self._send_single_message(
                        buttoned_text, structured_parse_mode, reply_markup
                    )
                else:
                    sent_id = await self.send_message(
                        reg.prompt_text, structured_parse_mode, reply_markup
                    )
                if sent_id is None:
                    raise TelegramTransportAmbiguousError("sendMessage returned None message_id")
            except TelegramApiDefiniteRejectError as reject_err:
                failed_marked = await self.telegram_repo.mark_prompt_failed(
                    correlation_key,
                    f"DefiniteReject ({type(reject_err).__name__})",
                    waiter_lease.lease_id,
                )
                await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
                await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
                return CorrelationWaitResult(
                    status=CorrelationStatus.PROMPT_FAILED
                    if failed_marked
                    else CorrelationStatus.STORAGE_ERROR,
                    error_reason="prompt_rejected_by_api",
                    nonce=reg.nonce,
                )
            except Exception as send_err:
                unk_marked = await self.telegram_repo.mark_prompt_delivery_unknown(
                    correlation_key,
                    f"AmbiguousSend ({type(send_err).__name__})",
                    waiter_lease.lease_id,
                )
                await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
                await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
                return CorrelationWaitResult(
                    status=CorrelationStatus.PROMPT_DELIVERY_UNKNOWN
                    if unk_marked
                    else CorrelationStatus.STORAGE_ERROR,
                    error_reason="prompt_delivery_unknown",
                    nonce=reg.nonce,
                )

            recorded = await self.telegram_repo.record_prompt_sent(
                correlation_key, sent_id, waiter_lease.lease_id
            )
            if not recorded:
                unk_marked = await self.telegram_repo.mark_prompt_delivery_unknown(
                    correlation_key,
                    "record_prompt_sent_failed",
                    waiter_lease.lease_id,
                )
                await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
                await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
                return CorrelationWaitResult(
                    status=CorrelationStatus.PROMPT_DELIVERY_UNKNOWN
                    if unk_marked
                    else CorrelationStatus.STORAGE_ERROR,
                    error_reason="prompt_delivery_unknown_after_send",
                    nonce=reg.nonce,
                )
            prompt_message_id = sent_id

        # 6. Polling loop with exact reply-before-cursor ordering and lease renewals
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        token = self.settings.telegram_bot_token
        try:
            durable_cursor = await self.telegram_repo.get_cursor(self.bot_chat_key)
        except Exception:
            await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
            await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
            return CorrelationWaitResult(
                status=CorrelationStatus.STORAGE_ERROR,
                error_reason="cursor_read_failed",
                nonce=reg.nonce,
            )

        if durable_cursor is not None:
            self._last_update_id = max(self._last_update_id or 0, durable_cursor)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                while loop.time() < deadline:
                    # Renew poll lease periodically
                    poll_renewed = await self.telegram_repo.renew_poll_lease(
                        self.bot_chat_key, poll_lease, lease_duration_seconds=lease_duration
                    )
                    if not poll_renewed:
                        await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
                        await self.telegram_repo.release_waiter(
                            correlation_key, waiter_lease.lease_id
                        )
                        return CorrelationWaitResult(
                            status=CorrelationStatus.STORAGE_ERROR,
                            error_reason="poll_lease_renewal_failed",
                            nonce=reg.nonce,
                        )

                    poll_timeout = max(1, min(5, int(deadline - loop.time())))
                    offset = 0 if self._last_update_id is None else self._last_update_id + 1
                    response = await client.post(
                        f"{self._base_url}/getUpdates",
                        json={"offset": offset, "timeout": poll_timeout},
                    )
                    response.raise_for_status()
                    data = response.json()
                    if not data.get("ok"):
                        desc = redact_string(
                            str(data.get("description", "unknown error")),
                            extra_secrets=[token],
                        )
                        raise RuntimeError(f"Telegram getUpdates failed: {desc}")

                    for update in data.get("result", []):
                        up_id = update.get("update_id", 0)
                        msg = update.get("message", {})
                        callback_update = update.get("callback_query")

                        # ── Inline-button callback path ──
                        if callback_update:
                            matches, cb_query_id, _cb_index = (
                                self.callback_data_is_for_current_prompt(
                                    callback_update,
                                    self.settings.telegram_chat_id,
                                    reg.nonce,
                                    prompt_message_id,
                                )
                            )
                            if not matches:
                                # Foreign chat/nonce/prompt: never satisfies this
                                # waiter; advance the cursor and clear the spinner.
                                cursor_saved = await self.telegram_repo.save_cursor(
                                    self.bot_chat_key, up_id, poll_lease
                                )
                                if not cursor_saved:
                                    await self.telegram_repo.release_poll_lease(
                                        self.bot_chat_key, poll_lease
                                    )
                                    await self.telegram_repo.release_waiter(
                                        correlation_key, waiter_lease.lease_id
                                    )
                                    return CorrelationWaitResult(
                                        status=CorrelationStatus.STORAGE_ERROR,
                                        error_reason="cursor_save_failed_nonmatching",
                                        nonce=reg.nonce,
                                    )
                                self._last_update_id = max(self._last_update_id or 0, up_id)
                                await self._answer_callback_query(cb_query_id or "")
                                continue

                            rec_res = await self.telegram_repo.record_callback_reply(
                                correlation_key,
                                callback_update.get("data"),
                                up_id,
                                prompt_message_id,
                                waiter_lease.lease_id,
                                callback_query_id=cb_query_id or None,
                            )
                            if rec_res.duplicate and not rec_res.accepted:
                                # Idempotent redelivery: cursor advances, spinner
                                # clears, and the current waiter keeps waiting.
                                cursor_saved = await self.telegram_repo.save_cursor(
                                    self.bot_chat_key, up_id, poll_lease
                                )
                                if not cursor_saved:
                                    await self.telegram_repo.release_poll_lease(
                                        self.bot_chat_key, poll_lease
                                    )
                                    await self.telegram_repo.release_waiter(
                                        correlation_key, waiter_lease.lease_id
                                    )
                                    return CorrelationWaitResult(
                                        status=CorrelationStatus.STORAGE_ERROR,
                                        error_reason="cursor_save_failed_duplicate_callback",
                                        nonce=reg.nonce,
                                    )
                                self._last_update_id = max(self._last_update_id or 0, up_id)
                                await self._answer_callback_query(
                                    cb_query_id or "", "Already recorded"
                                )
                                continue

                            if rec_res.accepted:
                                # Durable reply first, then cursor, then ack.
                                cursor_saved = await self.telegram_repo.save_cursor(
                                    self.bot_chat_key, up_id, poll_lease
                                )
                                if not cursor_saved:
                                    await self.telegram_repo.release_poll_lease(
                                        self.bot_chat_key, poll_lease
                                    )
                                    await self.telegram_repo.release_waiter(
                                        correlation_key, waiter_lease.lease_id
                                    )
                                    return CorrelationWaitResult(
                                        status=CorrelationStatus.STORAGE_ERROR,
                                        error_reason="cursor_save_failed",
                                        nonce=reg.nonce,
                                    )

                                self._last_update_id = max(self._last_update_id or 0, up_id)
                                await self._answer_callback_query(
                                    cb_query_id or "", f"Recorded: {rec_res.reply_text or ''}"
                                )
                                consume_res = await self.telegram_repo.consume_reply(
                                    correlation_key, waiter_lease.lease_id
                                )
                                await self.telegram_repo.release_poll_lease(
                                    self.bot_chat_key, poll_lease
                                )
                                await self.telegram_repo.release_waiter(
                                    correlation_key, waiter_lease.lease_id
                                )
                                if not consume_res.consumed:
                                    return CorrelationWaitResult(
                                        status=CorrelationStatus.STORAGE_ERROR,
                                        error_reason="consume_reply_failed",
                                        nonce=reg.nonce,
                                    )

                                return CorrelationWaitResult(
                                    status=CorrelationStatus.REPLIED,
                                    reply_text=consume_res.reply_text,
                                    reply_kind=consume_res.reply_kind,
                                    reply_option_index=consume_res.reply_option_index,
                                    nonce=reg.nonce,
                                )

                            # Rejected for another reason: treat like any
                            # non-matching update (advance cursor only).
                            cursor_saved = await self.telegram_repo.save_cursor(
                                self.bot_chat_key, up_id, poll_lease
                            )
                            if not cursor_saved:
                                await self.telegram_repo.release_poll_lease(
                                    self.bot_chat_key, poll_lease
                                )
                                await self.telegram_repo.release_waiter(
                                    correlation_key, waiter_lease.lease_id
                                )
                                return CorrelationWaitResult(
                                    status=CorrelationStatus.STORAGE_ERROR,
                                    error_reason="cursor_save_failed_nonmatching",
                                    nonce=reg.nonce,
                                )
                            self._last_update_id = max(self._last_update_id or 0, up_id)
                            await self._answer_callback_query(cb_query_id or "")
                            continue

                        reply = correlated_reply_text(
                            msg,
                            self.settings.telegram_chat_id,
                            reg.nonce,
                            prompt_message_id,
                        )

                        command = extract_control_command(
                            msg,
                            self.settings.telegram_chat_id,
                        )
                        if command and _control_command_handler is not None:
                            await _control_command_handler(command)
                            cursor_saved = await self.telegram_repo.save_cursor(
                                self.bot_chat_key, up_id, poll_lease
                            )
                            if not cursor_saved:
                                await self.telegram_repo.release_poll_lease(
                                    self.bot_chat_key, poll_lease
                                )
                                await self.telegram_repo.release_waiter(
                                    correlation_key, waiter_lease.lease_id
                                )
                                return CorrelationWaitResult(
                                    status=CorrelationStatus.STORAGE_ERROR,
                                    error_reason="cursor_save_failed_control_command",
                                    nonce=reg.nonce,
                                )
                            self._last_update_id = max(self._last_update_id or 0, up_id)
                            continue

                        if reply is not None:
                            msg_id = msg.get("message_id", 0)
                            rec_res = await self.telegram_repo.record_reply(
                                correlation_key,
                                reply,
                                up_id,
                                msg_id,
                                waiter_lease.lease_id,
                            )
                            if rec_res.accepted:
                                # Save cursor under exact poll lease
                                cursor_saved = await self.telegram_repo.save_cursor(
                                    self.bot_chat_key, up_id, poll_lease
                                )
                                if not cursor_saved:
                                    await self.telegram_repo.release_poll_lease(
                                        self.bot_chat_key, poll_lease
                                    )
                                    await self.telegram_repo.release_waiter(
                                        correlation_key, waiter_lease.lease_id
                                    )
                                    return CorrelationWaitResult(
                                        status=CorrelationStatus.STORAGE_ERROR,
                                        error_reason="cursor_save_failed",
                                        nonce=reg.nonce,
                                    )

                                self._last_update_id = max(self._last_update_id or 0, up_id)
                                consume_res = await self.telegram_repo.consume_reply(
                                    correlation_key, waiter_lease.lease_id
                                )
                                await self.telegram_repo.release_poll_lease(
                                    self.bot_chat_key, poll_lease
                                )
                                await self.telegram_repo.release_waiter(
                                    correlation_key, waiter_lease.lease_id
                                )
                                if not consume_res.consumed:
                                    return CorrelationWaitResult(
                                        status=CorrelationStatus.STORAGE_ERROR,
                                        error_reason="consume_reply_failed",
                                        nonce=reg.nonce,
                                    )

                                return CorrelationWaitResult(
                                    status=CorrelationStatus.REPLIED,
                                    reply_text=consume_res.reply_text,
                                    reply_kind=consume_res.reply_kind,
                                    reply_option_index=consume_res.reply_option_index,
                                    nonce=reg.nonce,
                                )
                        else:
                            cursor_saved = await self.telegram_repo.save_cursor(
                                self.bot_chat_key, up_id, poll_lease
                            )
                            if not cursor_saved:
                                await self.telegram_repo.release_poll_lease(
                                    self.bot_chat_key, poll_lease
                                )
                                await self.telegram_repo.release_waiter(
                                    correlation_key, waiter_lease.lease_id
                                )
                                return CorrelationWaitResult(
                                    status=CorrelationStatus.STORAGE_ERROR,
                                    error_reason="cursor_save_failed_nonmatching",
                                    nonce=reg.nonce,
                                )
                            self._last_update_id = max(self._last_update_id or 0, up_id)

        except asyncio.CancelledError:
            await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
            await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
            raise
        except Exception as exc:
            await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
            await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
            sanitized = redact_string(str(exc), extra_secrets=[token])
            raise RuntimeError(f"Telegram getUpdates error: {sanitized}") from None

        # Deadline reached: mark timed out
        await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
        timed_out_ok = await self.telegram_repo.mark_timed_out(
            correlation_key, waiter_lease.lease_id
        )
        await self.telegram_repo.release_waiter(correlation_key, waiter_lease.lease_id)
        if not timed_out_ok:
            curr = await self.telegram_repo.get_correlation(correlation_key)
            if curr and curr.status in (CorrelationStatus.REPLIED, CorrelationStatus.CONSUMED):
                return CorrelationWaitResult(
                    status=curr.status,
                    reply_text=curr.reply_text,
                    reply_kind=curr.reply_kind,
                    reply_option_index=curr.reply_option_index,
                    nonce=reg.nonce,
                )
            return CorrelationWaitResult(
                status=CorrelationStatus.STORAGE_ERROR,
                error_reason="mark_timed_out_failed",
                nonce=reg.nonce,
            )

        return CorrelationWaitResult(
            status=CorrelationStatus.TIMED_OUT,
            timed_out=True,
            nonce=reg.nonce,
        )

    async def _recover_cached_callback_reply(
        self,
        *,
        correlation_key: str,
        reg,
        waiter_lease_id: str,
    ) -> CorrelationWaitResult:
        """Complete cursor/ack recovery for an already-recorded callback reply.

        Ordering is preserved: the durable reply already exists; the persisted
        ``reply_update_id`` is advanced into the durable cursor, the persisted
        callback-query ID (when available) is acknowledged, and only then is the
        reply consumed and returned. A crash at any boundary stays idempotent:
        the next invocation repeats whichever steps are still outstanding.
        """
        poll_lease = await self.telegram_repo.claim_poll_lease(
            self.bot_chat_key,
            lease_duration_seconds=self.settings.telegram_poll_lease_seconds,
        )
        if not poll_lease:
            await self.telegram_repo.release_waiter(correlation_key, waiter_lease_id)
            return CorrelationWaitResult(
                status=CorrelationStatus.STORAGE_ERROR,
                error_reason="concurrent_bot_poller_active",
                nonce=reg.nonce,
            )

        try:
            # The register result carries identity only; reply metadata lives on
            # the durable correlation document.
            corr_doc = await self.telegram_repo.get_correlation(correlation_key)
            if corr_doc is None:
                return CorrelationWaitResult(
                    status=CorrelationStatus.STORAGE_ERROR,
                    error_reason="correlation_not_found_during_callback_recovery",
                    nonce=reg.nonce,
                )

            reply_update_id = corr_doc.reply_update_id
            if isinstance(reply_update_id, int) and reply_update_id > 0:
                cursor_saved = await self.telegram_repo.save_cursor(
                    self.bot_chat_key, reply_update_id, poll_lease
                )
                if not cursor_saved:
                    return CorrelationWaitResult(
                        status=CorrelationStatus.STORAGE_ERROR,
                        error_reason="cursor_save_failed_callback_recovery",
                        nonce=reg.nonce,
                    )
                self._last_update_id = max(self._last_update_id or 0, reply_update_id)

            cbq_id = corr_doc.reply_callback_query_id
            if cbq_id:
                await self._answer_callback_query(cbq_id, f"Recorded: {corr_doc.reply_text or ''}")

            consume_res = await self.telegram_repo.consume_reply(correlation_key, waiter_lease_id)
            if not consume_res.consumed:
                return CorrelationWaitResult(
                    status=CorrelationStatus.STORAGE_ERROR,
                    error_reason="consume_reply_failed",
                    nonce=reg.nonce,
                )
            return CorrelationWaitResult(
                status=CorrelationStatus.REPLIED,
                reply_text=consume_res.reply_text,
                reply_kind=consume_res.reply_kind or "callback",
                reply_option_index=consume_res.reply_option_index,
                nonce=reg.nonce,
            )
        finally:
            await self.telegram_repo.release_poll_lease(self.bot_chat_key, poll_lease)
            await self.telegram_repo.release_waiter(correlation_key, waiter_lease_id)

    async def drain_outbox(self, limit: int = 10) -> OutboxDrainResult:
        """Drain due queued outbox records with checked delivery and bounded backoff."""
        claim = await self.outbox_repo.claim_due_records(limit=limit)
        if not claim.claimed or not claim.records:
            return OutboxDrainResult()

        sent_count = 0
        failed_count = 0
        unknown_count = 0
        errors: list[str] = []

        for record in claim.records:
            record_lease = record.lease_id
            if not record_lease:
                continue

            all_chunks_ok = True
            for chunk in record.chunks:
                if chunk.sent and chunk.message_id is not None:
                    continue

                # Mark chunk in flight
                marked_flight = await self.outbox_repo.mark_chunk_in_flight(
                    record.idempotency_key, chunk.chunk_index, record_lease
                )
                if not marked_flight:
                    all_chunks_ok = False
                    errors.append(
                        f"Outbox in-flight transition failed for chunk {chunk.chunk_index}"
                    )
                    break

                try:
                    msg_id = await self._send_single_message(chunk.text, None)
                    if msg_id is None:
                        raise TelegramTransportAmbiguousError(
                            "sendMessage returned None message_id"
                        )
                except TelegramApiDefiniteRejectError as reject_err:
                    all_chunks_ok = False
                    errors.append(f"Outbox send rejected ({type(reject_err).__name__})")
                    failed_marked = await self.outbox_repo.mark_record_failed(
                        record.idempotency_key,
                        f"ApiReject ({type(reject_err).__name__})",
                        lease_id=record_lease,
                        current_attempts=record.attempts,
                        reset_in_flight_chunk=chunk.chunk_index,
                    )
                    if failed_marked:
                        failed_count += 1
                    break
                except Exception as ambig_err:
                    all_chunks_ok = False
                    errors.append(f"Outbox send ambiguous ({type(ambig_err).__name__})")
                    unk_marked = await self.outbox_repo.mark_delivery_unknown(
                        record.idempotency_key,
                        f"SendAmbiguous ({type(ambig_err).__name__})",
                        lease_id=record_lease,
                    )
                    if unk_marked:
                        unknown_count += 1
                    break

                # Update chunk confirmed sent
                chunk_marked = await self.outbox_repo.mark_chunk_sent(
                    record.idempotency_key, chunk.chunk_index, msg_id, record_lease
                )
                if not chunk_marked:
                    all_chunks_ok = False
                    unk_marked = await self.outbox_repo.mark_delivery_unknown(
                        record.idempotency_key,
                        "Mongo chunk update failed after checked Telegram send",
                        lease_id=record_lease,
                    )
                    if unk_marked:
                        unknown_count += 1
                    break

            if all_chunks_ok:
                record_marked = await self.outbox_repo.mark_record_sent(
                    record.idempotency_key, lease_id=record_lease
                )
                if record_marked:
                    sent_count += 1
                else:
                    unk_marked = await self.outbox_repo.mark_delivery_unknown(
                        record.idempotency_key,
                        "Mongo record mark_sent failed after all chunks delivered",
                        lease_id=record_lease,
                    )
                    if unk_marked:
                        unknown_count += 1

        return OutboxDrainResult(
            total_processed=len(claim.records),
            sent_count=sent_count,
            failed_count=failed_count,
            unknown_count=unknown_count,
            errors=errors,
        )

    async def enqueue_and_deliver(
        self,
        idempotency_key: str,
        notification_type: str,
        run_id: str,
        text: str,
        job_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> OutboxDeliveryResult:
        """Enqueue notification and attempt delivery."""
        enq = await self.outbox_repo.enqueue_notification(
            idempotency_key=idempotency_key,
            notification_type=notification_type,
            run_id=run_id,
            payload_text=text,
            job_id=job_id,
            metadata=metadata,
        )
        if not enq.enqueued and not enq.already_exists:
            return OutboxDeliveryResult(
                success=False,
                status=OutboxStatus.FAILED,
                idempotency_key=idempotency_key,
                reason=enq.reason or "enqueue_failed",
            )

        # Drain outbox to deliver due records
        await self.drain_outbox(limit=10)

        record = await self.outbox_repo.get_record(idempotency_key)
        if record is not None:
            total_chunks = len(record.chunks)
            chunks_sent = sum(1 for c in record.chunks if c.sent)
            return OutboxDeliveryResult(
                success=(record.status == OutboxStatus.SENT),
                status=record.status,
                idempotency_key=idempotency_key,
                chunks_sent=chunks_sent,
                total_chunks=total_chunks,
                reason=record.error_reason,
            )
        return OutboxDeliveryResult(
            success=False,
            status=OutboxStatus.FAILED,
            idempotency_key=idempotency_key,
            reason="record_not_found",
        )
