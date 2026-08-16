"""Telegram client with checked delivery and correlated replies."""

import asyncio
from typing import Optional

import httpx
from langsmith.run_helpers import trace

from jobapply.settings import get_settings
from jobapply.utils.redaction import redact_string
from jobapply.utils.tracing import get_telegram_metadata


def correlated_reply_text(
    message: dict,
    configured_chat_id: str,
    nonce: str,
    reply_to_message_id: int | None,
) -> str | None:
    """Return text for a new same-chat message or explicit correlated reply."""
    chat_id = str(message.get("chat", {}).get("id", ""))
    text = message.get("text", "")
    replied_to = message.get("reply_to_message", {}).get("message_id")
    is_direct_reply = reply_to_message_id is not None and replied_to == reply_to_message_id
    is_new_chat_message = (
        reply_to_message_id is not None
        and isinstance(message.get("message_id"), int)
        and message["message_id"] > reply_to_message_id
    )
    has_nonce = nonce in text
    if (
        chat_id != str(configured_chat_id)
        or not text
        or not (is_direct_reply or is_new_chat_message or has_nonce)
    ):
        return None
    return text.replace(nonce, "").strip(" []:-") if has_nonce else text.strip()


class TelegramClient:
    """Small Telegram Bot API client."""

    def __init__(self):
        self.settings = get_settings()
        self._last_update_id: int | None = None
        self._base_url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}"

    async def send_message(self, text: str, parse_mode: str | None = None) -> int | None:
        """Send checked messages and return the last Telegram message ID."""
        async with trace(
            "telegram_send_message",
            run_type="tool",
            metadata={"message_length": len(text), "parse_mode": parse_mode or "plain"},
        ) as run_tree:
            chunks = [text[index : index + 4096] for index in range(0, len(text), 4096)] or [""]
            message_id = None
            for chunk in chunks:
                message_id = await self._send_single_message(chunk, parse_mode)
                if len(chunks) > 1:
                    await asyncio.sleep(0.5)
            if run_tree:
                run_tree.metadata.update(get_telegram_metadata(message_sent=True))
                run_tree.metadata["chunks_sent"] = len(chunks)
            return message_id

    async def _send_single_message(self, text: str, parse_mode: str | None) -> int | None:
        payload = {"chat_id": self.settings.telegram_chat_id, "text": text}
        if parse_mode:
            payload["parse_mode"] = parse_mode
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
                    raise RuntimeError(f"Telegram sendMessage failed: {desc}")
                return data.get("result", {}).get("message_id")
        except Exception as exc:
            sanitized = redact_string(str(exc), extra_secrets=[token])
            raise RuntimeError(f"Telegram sendMessage error: {sanitized}") from None

    async def wait_for_correlated_reply(
        self,
        nonce: str,
        timeout: int = 300,
        reply_to_message_id: int | None = None,
    ) -> Optional[str]:
        """Wait for a direct Telegram reply, with ``nonce`` as a fallback."""
        async with trace(
            "telegram_wait_for_reply",
            run_type="tool",
            metadata=get_telegram_metadata(nonce=nonce, timeout=timeout),
        ) as run_tree:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            token = self.settings.telegram_bot_token
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                    while loop.time() < deadline:
                        poll_timeout = max(1, min(5, int(deadline - loop.time())))
                        offset = -1 if self._last_update_id is None else self._last_update_id + 1
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
                            self._last_update_id = max(
                                self._last_update_id or 0,
                                update["update_id"],
                            )
                            reply = correlated_reply_text(
                                update.get("message", {}),
                                self.settings.telegram_chat_id,
                                nonce,
                                reply_to_message_id,
                            )
                            if reply is not None:
                                if run_tree:
                                    run_tree.metadata.update(
                                        get_telegram_metadata(
                                            nonce=nonce, timeout=timeout, timed_out=False
                                        )
                                    )
                                return reply
            except Exception as exc:
                sanitized = redact_string(str(exc), extra_secrets=[token])
                raise RuntimeError(f"Telegram getUpdates error: {sanitized}") from None

            if run_tree:
                run_tree.metadata.update(
                    get_telegram_metadata(nonce=nonce, timeout=timeout, timed_out=True)
                )
            return None
