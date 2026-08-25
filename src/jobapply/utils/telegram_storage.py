"""Durable MongoDB persistence for Telegram correlations, cursors, and notification outbox."""

import asyncio
import concurrent.futures
import hashlib
import inspect
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Optional
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient

from jobapply.models.telegram import (
    CorrelationConsumeResult,
    CorrelationLeaseResult,
    CorrelationRegisterResult,
    CorrelationReplyResult,
    CorrelationStatus,
    OutboxClaimResult,
    OutboxEnqueueResult,
    OutboxRecord,
    OutboxStatus,
    TelegramCorrelation,
)
from jobapply.settings import get_settings
from jobapply.utils.dedup import (
    bound_and_redact_metadata,
    bound_string,
    canonicalize_job_id,
)
from jobapply.utils.mongo import MongoClientManager
from jobapply.utils.redaction import redact_string


class TelegramStorageError(RuntimeError):
    """Base error for Telegram storage operations."""

    pass


class TelegramApiDefiniteRejectError(RuntimeError):
    """Definitive API rejection error (prompt/message definitely rejected, terminal/safe retry)."""

    pass


class TelegramTransportAmbiguousError(RuntimeError):
    """Ambiguous transport/timeout error (message may or may not have been delivered)."""

    pass


SENSITIVE_PROMPT_PATTERNS = (
    r"\b(otp|2fa|two[- ]factor|verification[- ]code|security[- ]code|pin[- ]code)\b",
    r"\b(password|passcode|secret[- ]code)\b",
    r"\b(captcha|recaptcha|arkose|funcaptcha|puzzle[- ]answer)\b",
)


def contains_sensitive_security_challenge(text: str) -> bool:
    """Return True if text contains OTP, 2FA, password, or CAPTCHA keywords."""
    if not text:
        return False
    low = text.lower()
    for pat in SENSITIVE_PROMPT_PATTERNS:
        if re.search(pat, low):
            return True
    return False


def make_bot_chat_key(bot_token: str, chat_id: str) -> str:
    """Generate a sanitized, collision-resistant non-secret key for bot/chat cursor isolation."""
    raw_tok = str(bot_token or "").strip()
    raw_chat = str(chat_id or "").strip()
    tok_hash = hashlib.sha256(raw_tok.encode("utf-8")).hexdigest()
    chat_hash = hashlib.sha256(raw_chat.encode("utf-8")).hexdigest()
    return f"b_{tok_hash[:24]}_c_{chat_hash[:24]}"


# ── Inline keyboard callback data codec ──────────────────────────────────────
#
# Callback data is opaque and bounded (<=64 bytes per Telegram's limit): it
# carries only a fixed prefix, the correlation nonce, and a button index.
# Question text, job data, secrets, and answers never enter callback data.

CALLBACK_DATA_PREFIX = "jb"
CALLBACK_DATA_MAX_BYTES = 64
MAX_INLINE_ACTIONS = 8
MAX_INLINE_ACTION_LENGTH = 64
INLINE_KEYBOARD_BUTTONS_PER_ROW = 3


def sanitize_inline_actions(actions: Optional[list[str]]) -> list[str]:
    """Bound, redact, and cap an inline action label list."""
    if not actions:
        return []
    sanitized: list[str] = []
    for action in actions:
        clean = bound_string(str(action or ""), max_length=MAX_INLINE_ACTION_LENGTH)
        if clean:
            sanitized.append(clean)
        if len(sanitized) >= MAX_INLINE_ACTIONS:
            break
    return sanitized


def encode_callback_data(nonce: str, index: int) -> str:
    """Encode one opaque callback payload bound to the exact prompt nonce."""
    return f"{CALLBACK_DATA_PREFIX}|{nonce}|{int(index)}"


def decode_callback_data(data: str | None) -> Optional[tuple[str, int]]:
    """Decode opaque callback data; returns (nonce, index) or None when malformed."""
    if not isinstance(data, str):
        return None
    parts = data.split("|")
    if len(parts) != 3 or parts[0] != CALLBACK_DATA_PREFIX:
        return None
    nonce = parts[1]
    try:
        index = int(parts[2])
    except ValueError:
        return None
    if index < 0:
        return None
    return nonce, index


def build_inline_keyboard(nonce: str, actions: list[str]) -> list[list[dict]]:
    """Build Telegram inline keyboard rows with bounded opaque callback data."""
    rows: list[list[dict]] = []
    row: list[dict] = []
    for index, action in enumerate(actions[:MAX_INLINE_ACTIONS]):
        row.append(
            {
                "text": action[:MAX_INLINE_ACTION_LENGTH],
                "callback_data": encode_callback_data(nonce, index)[:CALLBACK_DATA_MAX_BYTES],
            }
        )
        if len(row) >= INLINE_KEYBOARD_BUTTONS_PER_ROW:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


def actions_payload_signature(actions: Optional[list[str]]) -> str:
    """Stable short signature of a sanitized action list for identity comparison."""
    import json as _json

    canonical = _json.dumps(
        sanitize_inline_actions(actions), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class TelegramPersistenceManager:
    """Shared single-client manager with independent index lifecycle states."""

    _lock: ClassVar[threading.Lock] = threading.Lock()

    # Separate index state for correlations and cursors
    _correlations_init_future: ClassVar[Optional[concurrent.futures.Future]] = None
    _correlations_index_succeeded: ClassVar[bool] = False
    _correlations_index_error: ClassVar[Optional[str]] = None

    # Separate index state for outbox
    _outbox_init_future: ClassVar[Optional[concurrent.futures.Future]] = None
    _outbox_index_succeeded: ClassVar[bool] = False
    _outbox_index_error: ClassVar[Optional[str]] = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        return MongoClientManager.get_client()

    @classmethod
    def _reset_state(cls) -> None:
        """Synchronously reset all persistence index lifecycle states."""
        with cls._lock:
            cls._correlations_init_future = None
            cls._correlations_index_succeeded = False
            cls._correlations_index_error = None
            cls._outbox_init_future = None
            cls._outbox_index_succeeded = False
            cls._outbox_index_error = None

    @classmethod
    async def close(cls) -> None:
        """Close shared MongoDB client and reset all persistence lifecycle states."""
        await MongoClientManager.close()


MongoClientManager.register_reset_hook(TelegramPersistenceManager._reset_state)


class TelegramRepository:
    """Repository for Telegram correlation intents and getUpdates cursors in MongoDB."""

    def __init__(self):
        settings = get_settings()
        self.client = TelegramPersistenceManager.get_client()
        self.db = self.client[settings.mongodb_db]
        self.correlations_col = self.db[settings.telegram_correlations_collection]
        self.cursors_col = self.db[settings.telegram_cursors_collection]

    async def ensure_indexes(self) -> bool:
        """Ensure unique and TTL indexes once per process/lifecycle for correlations and cursors."""
        with TelegramPersistenceManager._lock:
            if TelegramPersistenceManager._correlations_index_succeeded:
                return True
            if TelegramPersistenceManager._correlations_index_error is not None:
                raise TelegramStorageError(TelegramPersistenceManager._correlations_index_error)

            fut = TelegramPersistenceManager._correlations_init_future
            if fut is None:
                fut = concurrent.futures.Future()
                TelegramPersistenceManager._correlations_init_future = fut
                is_initiator = True
            else:
                is_initiator = False

        if is_initiator:
            try:
                if hasattr(self.correlations_col, "create_index"):
                    res1 = self.correlations_col.create_index("correlation_key", unique=True)
                    if inspect.isawaitable(res1):
                        await res1
                    res2 = self.correlations_col.create_index("expires_at", expireAfterSeconds=0)
                    if inspect.isawaitable(res2):
                        await res2
                    res3 = self.correlations_col.create_index(
                        [("status", 1), ("lease_expires_at", 1)]
                    )
                    if inspect.isawaitable(res3):
                        await res3

                if hasattr(self.cursors_col, "create_index"):
                    res4 = self.cursors_col.create_index("bot_chat_key", unique=True)
                    if inspect.isawaitable(res4):
                        await res4
                    res5 = self.cursors_col.create_index([("lease_expires_at", 1)])
                    if inspect.isawaitable(res5):
                        await res5

                with TelegramPersistenceManager._lock:
                    TelegramPersistenceManager._correlations_index_succeeded = True
                    TelegramPersistenceManager._correlations_index_error = None
                fut.set_result(True)
                return True
            except Exception as e:
                sanitized_err = f"Telegram index initialization failed ({type(e).__name__})"
                with TelegramPersistenceManager._lock:
                    TelegramPersistenceManager._correlations_index_succeeded = False
                    TelegramPersistenceManager._correlations_index_error = sanitized_err
                fut.set_result(False)
                raise TelegramStorageError(sanitized_err)
        else:
            loop = asyncio.get_running_loop()
            await asyncio.wrap_future(fut, loop=loop)
            with TelegramPersistenceManager._lock:
                if TelegramPersistenceManager._correlations_index_succeeded:
                    return True
                raise TelegramStorageError(
                    TelegramPersistenceManager._correlations_index_error
                    or "Telegram index initialization failed"
                )

    async def reconcile_expired_correlations(self) -> int:
        """Reconcile expired prompt-in-flight correlations to PROMPT_DELIVERY_UNKNOWN."""
        await self.ensure_indexes()
        now = datetime.now(timezone.utc)
        res = self.correlations_col.update_many(
            {
                "status": CorrelationStatus.PENDING.value,
                "prompt_in_flight": True,
                "lease_expires_at": {"$lt": now},
            },
            {
                "$set": {
                    "status": CorrelationStatus.PROMPT_DELIVERY_UNKNOWN.value,
                    "prompt_in_flight": False,
                    "error_reason": "prompt_in_flight_lease_expired_manual_reconciliation_required",
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        return getattr(res, "modified_count", 0) if res else 0

    async def claim_poll_lease(
        self, bot_chat_key: str, lease_duration_seconds: int = 30
    ) -> Optional[str]:
        """Claim exclusive expiring polling lease without masking insert failures."""
        await self.ensure_indexes()
        safe_key = bound_string(bot_chat_key, max_length=128)
        if not safe_key:
            return None
        now = datetime.now(timezone.utc)
        lease_id = f"poll_{uuid4().hex[:12]}"
        lease_expires = now + timedelta(seconds=max(5, lease_duration_seconds))

        # 1. Ensure cursor doc exists; if insert throws, verify doc exists via re-read
        try:
            ins = self.cursors_col.insert_one(
                {
                    "bot_chat_key": safe_key,
                    "last_update_id": 0,
                    "created_at": now,
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            )
            if inspect.isawaitable(ins):
                await ins
        except Exception:
            re_read = self.cursors_col.find_one({"bot_chat_key": safe_key})
            if inspect.isawaitable(re_read):
                re_read = await re_read
            if not re_read:
                raise TelegramStorageError("Cursor document initialization failed during race")

        # 2. Conditional CAS update without upsert to acquire lease
        res = self.cursors_col.find_one_and_update(
            {
                "bot_chat_key": safe_key,
                "$or": [
                    {"lease_id": None},
                    {"lease_expires_at": {"$lt": now}},
                ],
            },
            {
                "$set": {
                    "lease_id": lease_id,
                    "lease_expires_at": lease_expires,
                    "updated_at": now,
                }
            },
            upsert=False,
            return_document=True if hasattr(self.cursors_col, "find_one_and_update") else False,
        )
        if inspect.isawaitable(res):
            res = await res
        if res:
            return lease_id
        return None

    async def renew_poll_lease(
        self, bot_chat_key: str, lease_id: str, lease_duration_seconds: int = 30
    ) -> bool:
        """Renew active polling lease."""
        await self.ensure_indexes()
        safe_key = bound_string(bot_chat_key, max_length=128)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease:
            return False
        now = datetime.now(timezone.utc)
        lease_expires = now + timedelta(seconds=max(5, lease_duration_seconds))
        res = self.cursors_col.update_one(
            {"bot_chat_key": safe_key, "lease_id": safe_lease},
            {"$set": {"lease_expires_at": lease_expires, "updated_at": now}},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def release_poll_lease(self, bot_chat_key: str, lease_id: str) -> bool:
        """Release active polling lease."""
        await self.ensure_indexes()
        safe_key = bound_string(bot_chat_key, max_length=128)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease:
            return False
        now = datetime.now(timezone.utc)
        res = self.cursors_col.update_one(
            {"bot_chat_key": safe_key, "lease_id": safe_lease},
            {"$set": {"lease_id": None, "lease_expires_at": None, "updated_at": now}},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def get_cursor(self, bot_chat_key: str) -> Optional[int]:
        """Get the latest durable Telegram update ID for a bot/chat identity."""
        await self.ensure_indexes()
        safe_key = bound_string(bot_chat_key, max_length=128)
        if not safe_key:
            return None
        res = self.cursors_col.find_one({"bot_chat_key": safe_key})
        if inspect.isawaitable(res):
            res = await res
        if not res:
            return None
        if not isinstance(res.get("last_update_id"), int):
            raise TelegramStorageError("Malformed cursor document: invalid last_update_id")
        return int(res["last_update_id"])

    async def save_cursor(self, bot_chat_key: str, update_id: int, lease_id: str) -> bool:
        """Atomically persist durable update cursor under exact lease without upsert."""
        await self.ensure_indexes()
        safe_key = bound_string(bot_chat_key, max_length=128)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease or not isinstance(update_id, int):
            return False
        now = datetime.now(timezone.utc)
        res = self.cursors_col.update_one(
            {"bot_chat_key": safe_key, "lease_id": safe_lease},
            {
                "$max": {"last_update_id": update_id},
                "$set": {"updated_at": now},
            },
            upsert=False,
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def get_correlation(self, correlation_key: str) -> Optional[TelegramCorrelation]:
        """Fetch correlation record by deterministic correlation key, failing closed on malformed doc."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        if not safe_key:
            return None
        res = self.correlations_col.find_one({"correlation_key": safe_key})
        if inspect.isawaitable(res):
            res = await res
        if not res:
            return None
        try:
            return TelegramCorrelation(**res)
        except Exception as exc:
            raise TelegramStorageError(f"Malformed correlation document ({type(exc).__name__})")

    async def register_intent(
        self,
        correlation_key: str,
        purpose: str,
        run_id: str,
        prompt_text: str,
        job_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        inline_actions: Optional[list[str]] = None,
    ) -> CorrelationRegisterResult:
        """Register a correlation intent with collision-resistant hashing and stable nonce."""
        await self.ensure_indexes()
        await self.reconcile_expired_correlations()

        safe_key = bound_string(correlation_key, max_length=160)
        safe_purpose = bound_string(purpose, max_length=64)
        safe_run_id = bound_string(run_id, max_length=64)
        safe_job_id = canonicalize_job_id(job_id) if job_id else None
        raw_prompt = redact_string(prompt_text or "")
        safe_meta = bound_and_redact_metadata(metadata or {})
        safe_actions = sanitize_inline_actions(inline_actions)

        if not safe_key or not safe_purpose or not safe_run_id or not raw_prompt:
            return CorrelationRegisterResult(
                registered=False,
                is_new=False,
                correlation_key=safe_key,
                nonce="",
                prompt_text="",
                inline_actions=safe_actions,
                status=CorrelationStatus.STORAGE_ERROR,
                reason="invalid_empty_fields",
            )

        if contains_sensitive_security_challenge(raw_prompt):
            raise TelegramStorageError(
                "Sensitive security challenge detected; refusing to persist over Telegram."
            )

        nonce = hashlib.sha256(f"{safe_key}:{raw_prompt}".encode("utf-8")).hexdigest()[:12]
        if nonce not in raw_prompt:
            rendered_prompt = f"{raw_prompt}\n\n[Ref: `{nonce}`]"
        else:
            rendered_prompt = raw_prompt
        safe_prompt = bound_string(rendered_prompt, max_length=4096)
        prompt_hash = hashlib.sha256(safe_prompt.encode("utf-8")).hexdigest()

        now = datetime.now(timezone.utc)
        settings = get_settings()
        ttl_expires = now + timedelta(seconds=settings.telegram_correlation_ttl_seconds)

        existing = await self.get_correlation(safe_key)
        if existing is not None:
            if (
                existing.purpose != safe_purpose
                or existing.run_id != safe_run_id
                or existing.job_id != safe_job_id
                or existing.prompt_hash != prompt_hash
                or actions_payload_signature(existing.inline_actions)
                != actions_payload_signature(safe_actions)
            ):
                return CorrelationRegisterResult(
                    registered=False,
                    is_new=False,
                    correlation_key=safe_key,
                    nonce=existing.nonce,
                    prompt_text=existing.prompt_text,
                    inline_actions=existing.inline_actions,
                    prompt_in_flight=existing.prompt_in_flight,
                    prompt_message_id=existing.prompt_message_id,
                    status=existing.status,
                    reply_text=existing.reply_text,
                    reason="correlation_key_conflict_mismatched_payload",
                )

            return CorrelationRegisterResult(
                registered=True,
                is_new=False,
                correlation_key=safe_key,
                nonce=existing.nonce,
                prompt_text=existing.prompt_text,
                inline_actions=existing.inline_actions,
                prompt_in_flight=existing.prompt_in_flight,
                prompt_message_id=existing.prompt_message_id,
                status=existing.status,
                reply_text=existing.reply_text,
            )

        new_doc = {
            "correlation_key": safe_key,
            "purpose": safe_purpose,
            "run_id": safe_run_id,
            "job_id": safe_job_id,
            "prompt_hash": prompt_hash,
            "nonce": nonce,
            "prompt_text": safe_prompt,
            "inline_actions": safe_actions,
            "prompt_in_flight": False,
            "prompt_message_id": None,
            "prompt_sent_at": None,
            "status": CorrelationStatus.PENDING.value,
            "lease_id": None,
            "lease_expires_at": None,
            "reply_text": None,
            "reply_update_id": None,
            "reply_message_id": None,
            "replied_at": None,
            "consumed_at": None,
            "created_at": now,
            "updated_at": now,
            "expires_at": ttl_expires,
            "metadata": safe_meta,
        }

        try:
            res = self.correlations_col.insert_one(new_doc)
            if inspect.isawaitable(res):
                await res
            return CorrelationRegisterResult(
                registered=True,
                is_new=True,
                correlation_key=safe_key,
                nonce=nonce,
                prompt_text=safe_prompt,
                inline_actions=safe_actions,
                prompt_in_flight=False,
                prompt_message_id=None,
                status=CorrelationStatus.PENDING,
            )
        except Exception:
            re_read = await self.get_correlation(safe_key)
            if re_read is not None:
                if (
                    re_read.purpose != safe_purpose
                    or re_read.run_id != safe_run_id
                    or re_read.job_id != safe_job_id
                    or re_read.prompt_hash != prompt_hash
                    or actions_payload_signature(re_read.inline_actions)
                    != actions_payload_signature(safe_actions)
                ):
                    return CorrelationRegisterResult(
                        registered=False,
                        is_new=False,
                        correlation_key=safe_key,
                        nonce=re_read.nonce,
                        prompt_text=re_read.prompt_text,
                        inline_actions=re_read.inline_actions,
                        prompt_in_flight=re_read.prompt_in_flight,
                        prompt_message_id=re_read.prompt_message_id,
                        status=re_read.status,
                        reason="correlation_key_conflict_mismatched_payload",
                    )
                return CorrelationRegisterResult(
                    registered=True,
                    is_new=False,
                    correlation_key=safe_key,
                    nonce=re_read.nonce,
                    prompt_text=re_read.prompt_text,
                    inline_actions=re_read.inline_actions,
                    prompt_in_flight=re_read.prompt_in_flight,
                    prompt_message_id=re_read.prompt_message_id,
                    status=re_read.status,
                    reply_text=re_read.reply_text,
                )
            raise TelegramStorageError("Failed to register correlation intent during race")

    async def mark_prompt_in_flight(
        self,
        correlation_key: str,
        lease_id: str,
    ) -> bool:
        """Mark prompt in-flight under exact waiter lease before HTTP dispatch."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        now = datetime.now(timezone.utc)
        res = self.correlations_col.update_one(
            {
                "correlation_key": safe_key,
                "status": CorrelationStatus.PENDING.value,
                "lease_id": safe_lease,
                "prompt_message_id": None,
                "prompt_in_flight": False,
            },
            {"$set": {"prompt_in_flight": True, "updated_at": now}},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def record_prompt_sent(
        self,
        correlation_key: str,
        prompt_message_id: int,
        lease_id: str,
    ) -> bool:
        """Record the Telegram prompt message ID after checked delivery under exact lease."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease or not isinstance(prompt_message_id, int):
            return False
        now = datetime.now(timezone.utc)
        res = self.correlations_col.update_one(
            {
                "correlation_key": safe_key,
                "status": CorrelationStatus.PENDING.value,
                "lease_id": safe_lease,
                "prompt_message_id": None,
            },
            {
                "$set": {
                    "prompt_message_id": prompt_message_id,
                    "prompt_in_flight": False,
                    "prompt_sent_at": now,
                    "updated_at": now,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def mark_prompt_failed(self, correlation_key: str, reason: str, lease_id: str) -> bool:
        """Record terminal prompt failure for definitive API rejection under exact lease."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        safe_reason = bound_string(redact_string(reason), max_length=256)
        now = datetime.now(timezone.utc)
        res = self.correlations_col.update_one(
            {
                "correlation_key": safe_key,
                "status": CorrelationStatus.PENDING.value,
                "lease_id": safe_lease,
                "prompt_message_id": None,
            },
            {
                "$set": {
                    "status": CorrelationStatus.PROMPT_FAILED.value,
                    "prompt_in_flight": False,
                    "error_reason": safe_reason,
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def mark_prompt_delivery_unknown(
        self, correlation_key: str, reason: str, lease_id: str
    ) -> bool:
        """Record prompt-delivery-unknown state under exact lease."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        safe_reason = bound_string(redact_string(reason), max_length=256)
        now = datetime.now(timezone.utc)
        res = self.correlations_col.update_one(
            {
                "correlation_key": safe_key,
                "status": CorrelationStatus.PENDING.value,
                "lease_id": safe_lease,
                "prompt_message_id": None,
            },
            {
                "$set": {
                    "status": CorrelationStatus.PROMPT_DELIVERY_UNKNOWN.value,
                    "prompt_in_flight": False,
                    "error_reason": safe_reason,
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def claim_waiter(
        self,
        correlation_key: str,
        lease_duration_seconds: int = 30,
    ) -> CorrelationLeaseResult:
        """Atomically claim an exclusive expiring waiter lease on a correlation."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        if not safe_key:
            return CorrelationLeaseResult(acquired=False, reason="invalid_key")

        now = datetime.now(timezone.utc)
        lease_id = f"w_{uuid4().hex[:12]}"
        expires_at = now + timedelta(seconds=max(5, lease_duration_seconds))

        query = {
            "correlation_key": safe_key,
            "status": {
                "$in": [
                    CorrelationStatus.PENDING.value,
                    CorrelationStatus.REPLIED.value,
                ]
            },
            "$or": [
                {"lease_id": None},
                {"lease_expires_at": {"$lt": now}},
            ],
        }
        update = {
            "$set": {
                "lease_id": lease_id,
                "lease_expires_at": expires_at,
                "updated_at": now,
            }
        }
        res = self.correlations_col.find_one_and_update(
            query,
            update,
            return_document=True
            if hasattr(self.correlations_col, "find_one_and_update")
            else False,
        )
        if inspect.isawaitable(res):
            res = await res

        if res:
            try:
                corr = TelegramCorrelation(**res)
                return CorrelationLeaseResult(
                    acquired=True,
                    lease_id=lease_id,
                    status=corr.status,
                    reply_text=corr.reply_text,
                )
            except Exception as exc:
                raise TelegramStorageError(f"Malformed correlation document ({type(exc).__name__})")

        existing = await self.get_correlation(safe_key)
        if existing is None:
            return CorrelationLeaseResult(acquired=False, reason="correlation_not_found")
        if existing.status in (
            CorrelationStatus.CONSUMED,
            CorrelationStatus.TIMED_OUT,
            CorrelationStatus.PROMPT_DELIVERY_UNKNOWN,
            CorrelationStatus.PROMPT_FAILED,
        ):
            return CorrelationLeaseResult(
                acquired=False,
                status=existing.status,
                reason=f"correlation_{existing.status.value}",
            )
        return CorrelationLeaseResult(
            acquired=False,
            status=existing.status,
            reason="active_lease_held_by_other_waiter",
        )

    async def release_waiter(self, correlation_key: str, lease_id: str) -> bool:
        """Release active waiter lease."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease:
            return False
        now = datetime.now(timezone.utc)
        res = self.correlations_col.update_one(
            {"correlation_key": safe_key, "lease_id": safe_lease},
            {"$set": {"lease_id": None, "lease_expires_at": None, "updated_at": now}},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def record_reply(
        self,
        correlation_key: str,
        reply_text: str,
        update_id: int,
        message_id: int,
        lease_id: str,
    ) -> CorrelationReplyResult:
        """Atomically persist reply receipt under exact waiter lease."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease:
            return CorrelationReplyResult(
                accepted=False,
                status=CorrelationStatus.PENDING,
                reason="invalid_identifiers",
            )

        if contains_sensitive_security_challenge(reply_text):
            raise TelegramStorageError(
                "Sensitive security challenge response detected; refusing to persist."
            )

        safe_reply = bound_string(redact_string(reply_text), max_length=512)
        now = datetime.now(timezone.utc)
        settings = get_settings()
        ttl_expires = now + timedelta(seconds=settings.telegram_correlation_ttl_seconds)

        res = self.correlations_col.find_one_and_update(
            {
                "correlation_key": safe_key,
                "status": CorrelationStatus.PENDING.value,
                "lease_id": safe_lease,
                "prompt_message_id": {"$ne": None, "$lt": message_id},
            },
            {
                "$set": {
                    "status": CorrelationStatus.REPLIED.value,
                    "reply_text": safe_reply,
                    "reply_kind": "text",
                    "reply_option_index": None,
                    "reply_update_id": int(update_id),
                    "reply_message_id": int(message_id),
                    "replied_at": now,
                    "updated_at": now,
                    "expires_at": ttl_expires,
                }
            },
            return_document=True
            if hasattr(self.correlations_col, "find_one_and_update")
            else False,
        )
        if inspect.isawaitable(res):
            res = await res

        if res:
            return CorrelationReplyResult(
                accepted=True,
                status=CorrelationStatus.REPLIED,
                reply_text=safe_reply,
                reply_kind="text",
            )

        existing = await self.get_correlation(safe_key)
        if existing:
            return CorrelationReplyResult(
                accepted=False,
                status=existing.status,
                reply_text=existing.reply_text,
                reason=f"cannot_record_reply_in_status_{existing.status.value}",
            )
        return CorrelationReplyResult(
            accepted=False,
            status=CorrelationStatus.PENDING,
            reason="correlation_not_found",
        )

    async def record_callback_reply(
        self,
        correlation_key: str,
        callback_data: str | None,
        update_id: int,
        prompt_message_id: int,
        lease_id: str,
        callback_query_id: str | None = None,
    ) -> CorrelationReplyResult:
        """Atomically persist a validated inline-button callback under the exact waiter lease.

        Validation is fail-closed: chat identity and waiter lease are enforced by
        the query, the opaque callback nonce must match this correlation, the
        button index must resolve against the persisted action list, and the
        callback must originate from the exact durable prompt message. A
        duplicate redelivery is reported idempotently and never satisfies a
        second waiter. The bounded callback-query ID is persisted so a crash at
        any later boundary can still acknowledge it during recovery.
        """
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        safe_cbq_id = bound_string(callback_query_id or "", max_length=64) or None
        if not safe_key or not safe_lease:
            return CorrelationReplyResult(
                accepted=False,
                status=CorrelationStatus.PENDING,
                reason="invalid_identifiers",
            )

        existing = await self.get_correlation(safe_key)
        if existing is None:
            return CorrelationReplyResult(
                accepted=False,
                status=CorrelationStatus.PENDING,
                reason="correlation_not_found",
            )

        decoded = decode_callback_data(callback_data)
        if decoded is None:
            return CorrelationReplyResult(
                accepted=False,
                status=existing.status,
                reason="callback_data_malformed",
            )
        cb_nonce, cb_index = decoded
        if cb_nonce != existing.nonce:
            return CorrelationReplyResult(
                accepted=False,
                status=existing.status,
                reason="callback_nonce_mismatch",
            )
        if (
            not isinstance(prompt_message_id, int)
            or existing.prompt_message_id != prompt_message_id
        ):
            return CorrelationReplyResult(
                accepted=False,
                status=existing.status,
                reason="callback_prompt_message_mismatch",
            )
        actions = list(existing.inline_actions or [])
        if cb_index >= len(actions):
            return CorrelationReplyResult(
                accepted=False,
                status=existing.status,
                reason="callback_option_index_out_of_range",
            )
        resolved_label = bound_string(redact_string(actions[cb_index]), max_length=512)

        now = datetime.now(timezone.utc)
        settings = get_settings()
        ttl_expires = now + timedelta(seconds=settings.telegram_correlation_ttl_seconds)

        res = self.correlations_col.find_one_and_update(
            {
                "correlation_key": safe_key,
                "status": CorrelationStatus.PENDING.value,
                "lease_id": safe_lease,
                "prompt_message_id": int(prompt_message_id),
            },
            {
                "$set": {
                    "status": CorrelationStatus.REPLIED.value,
                    "reply_text": resolved_label,
                    "reply_kind": "callback",
                    "reply_option_index": int(cb_index),
                    "reply_callback_query_id": safe_cbq_id,
                    "reply_update_id": int(update_id),
                    "reply_message_id": int(prompt_message_id),
                    "replied_at": now,
                    "updated_at": now,
                    "expires_at": ttl_expires,
                }
            },
            return_document=True
            if hasattr(self.correlations_col, "find_one_and_update")
            else False,
        )
        if inspect.isawaitable(res):
            res = await res

        if res:
            return CorrelationReplyResult(
                accepted=True,
                status=CorrelationStatus.REPLIED,
                reply_text=resolved_label,
                reply_kind="callback",
                option_index=int(cb_index),
            )

        # Duplicate delivery (already durably accepted for this update): report
        # idempotently without satisfying another waiter.
        reread = await self.get_correlation(safe_key)
        if (
            reread is not None
            and reread.status in (CorrelationStatus.REPLIED, CorrelationStatus.CONSUMED)
            and reread.reply_kind == "callback"
            and reread.reply_update_id == int(update_id)
        ):
            return CorrelationReplyResult(
                accepted=False,
                status=reread.status,
                reply_text=reread.reply_text,
                reply_kind="callback",
                option_index=reread.reply_option_index,
                duplicate=True,
                reason="duplicate_callback_already_recorded",
            )

        current = reread or existing
        return CorrelationReplyResult(
            accepted=False,
            status=current.status,
            reason=f"cannot_record_callback_in_status_{current.status.value}",
        )

    async def consume_reply(
        self,
        correlation_key: str,
        lease_id: str,
    ) -> CorrelationConsumeResult:
        """Atomically mark reply consumed and clear persisted reply text."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease:
            return CorrelationConsumeResult(consumed=False, reason="invalid_identifiers")

        now = datetime.now(timezone.utc)
        query = {
            "correlation_key": safe_key,
            "status": CorrelationStatus.REPLIED.value,
            "lease_id": safe_lease,
        }
        update = {
            "$set": {
                "status": CorrelationStatus.CONSUMED.value,
                "consumed_at": now,
                "reply_text": None,
                "updated_at": now,
                "lease_id": None,
                "lease_expires_at": None,
            }
        }
        res = self.correlations_col.find_one_and_update(
            query,
            update,
            return_document=False,
        )
        if inspect.isawaitable(res):
            res = await res

        if res:
            reply_text = str(res.get("reply_text") or "")
            return CorrelationConsumeResult(
                consumed=True,
                reply_text=reply_text,
                reply_kind=res.get("reply_kind"),
                reply_option_index=res.get("reply_option_index"),
            )

        existing = await self.get_correlation(safe_key)
        if existing:
            return CorrelationConsumeResult(
                consumed=False,
                reply_kind=existing.reply_kind,
                reply_option_index=existing.reply_option_index,
                reason=f"status_is_{existing.status.value}",
            )
        return CorrelationConsumeResult(consumed=False, reason="correlation_not_found")

    async def mark_timed_out(
        self,
        correlation_key: str,
        lease_id: str,
    ) -> bool:
        """Atomically mark correlation timed out under exact lease."""
        await self.ensure_indexes()
        safe_key = bound_string(correlation_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        if not safe_key or not safe_lease:
            return False

        now = datetime.now(timezone.utc)
        res = self.correlations_col.update_one(
            {
                "correlation_key": safe_key,
                "status": CorrelationStatus.PENDING.value,
                "lease_id": safe_lease,
            },
            {
                "$set": {
                    "status": CorrelationStatus.TIMED_OUT.value,
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    @classmethod
    async def close(cls) -> None:
        """Reset repository lifecycle state."""
        await TelegramPersistenceManager.close()


class NotificationOutboxRepository:
    """Repository for durable notification outbox records in MongoDB."""

    def __init__(self):
        settings = get_settings()
        self.client = TelegramPersistenceManager.get_client()
        self.db = self.client[settings.mongodb_db]
        self.outbox_col = self.db[settings.notification_outbox_collection]

    async def ensure_indexes(self) -> bool:
        """Ensure indexes for outbox records once per process/lifecycle."""
        with TelegramPersistenceManager._lock:
            if TelegramPersistenceManager._outbox_index_succeeded:
                return True
            if TelegramPersistenceManager._outbox_index_error is not None:
                raise TelegramStorageError(TelegramPersistenceManager._outbox_index_error)

            fut = TelegramPersistenceManager._outbox_init_future
            if fut is None:
                fut = concurrent.futures.Future()
                TelegramPersistenceManager._outbox_init_future = fut
                is_initiator = True
            else:
                is_initiator = False

        if is_initiator:
            try:
                if hasattr(self.outbox_col, "create_index"):
                    res1 = self.outbox_col.create_index("idempotency_key", unique=True)
                    if inspect.isawaitable(res1):
                        await res1
                    res2 = self.outbox_col.create_index([("status", 1), ("next_attempt_at", 1)])
                    if inspect.isawaitable(res2):
                        await res2
                    res3 = self.outbox_col.create_index([("lease_expires_at", 1)])
                    if inspect.isawaitable(res3):
                        await res3

                with TelegramPersistenceManager._lock:
                    TelegramPersistenceManager._outbox_index_succeeded = True
                    TelegramPersistenceManager._outbox_index_error = None
                fut.set_result(True)
                return True
            except Exception as e:
                sanitized_err = f"Outbox index initialization failed ({type(e).__name__})"
                with TelegramPersistenceManager._lock:
                    TelegramPersistenceManager._outbox_index_succeeded = False
                    TelegramPersistenceManager._outbox_index_error = sanitized_err
                fut.set_result(False)
                raise TelegramStorageError(sanitized_err)
        else:
            loop = asyncio.get_running_loop()
            await asyncio.wrap_future(fut, loop=loop)
            with TelegramPersistenceManager._lock:
                if TelegramPersistenceManager._outbox_index_succeeded:
                    return True
                raise TelegramStorageError(
                    TelegramPersistenceManager._outbox_index_error
                    or "Outbox index initialization failed"
                )

    async def reconcile_expired_sending_records(self) -> int:
        """Atomically transition expired SENDING records to DELIVERY_UNKNOWN."""
        await self.ensure_indexes()
        now = datetime.now(timezone.utc)
        res = self.outbox_col.update_many(
            {
                "status": OutboxStatus.SENDING.value,
                "lease_expires_at": {"$lt": now},
            },
            {
                "$set": {
                    "status": OutboxStatus.DELIVERY_UNKNOWN.value,
                    "error_reason": "lease_expired_during_send_manual_reconciliation_required",
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        return getattr(res, "modified_count", 0) if res else 0

    async def get_record(self, idempotency_key: str) -> Optional[OutboxRecord]:
        """Fetch outbox record by deterministic idempotency key, failing closed on malformed doc."""
        await self.ensure_indexes()
        safe_key = bound_string(idempotency_key, max_length=160)
        if not safe_key:
            return None
        res = self.outbox_col.find_one({"idempotency_key": safe_key})
        if inspect.isawaitable(res):
            res = await res
        if not res:
            return None
        try:
            return OutboxRecord(**res)
        except Exception as exc:
            raise TelegramStorageError(f"Malformed outbox document ({type(exc).__name__})")

    async def enqueue_notification(
        self,
        idempotency_key: str,
        notification_type: str,
        run_id: str,
        payload_text: str,
        job_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> OutboxEnqueueResult:
        """Atomically enqueue a notification with deterministic chunking and payload validation."""
        await self.ensure_indexes()
        safe_key = bound_string(idempotency_key, max_length=160)
        safe_type = bound_string(notification_type, max_length=64)
        safe_run_id = bound_string(run_id, max_length=64)
        safe_job_id = canonicalize_job_id(job_id) if job_id else None
        safe_payload = redact_string(payload_text or "")
        payload_hash = hashlib.sha256(safe_payload.encode("utf-8")).hexdigest()
        safe_meta = bound_and_redact_metadata(metadata or {})
        now = datetime.now(timezone.utc)
        settings = get_settings()

        existing = await self.get_record(safe_key)
        if existing is not None:
            if (
                existing.notification_type != safe_type
                or existing.run_id != safe_run_id
                or existing.job_id != safe_job_id
                or existing.payload_hash != payload_hash
            ):
                return OutboxEnqueueResult(
                    enqueued=False,
                    already_exists=True,
                    status=existing.status,
                    idempotency_key=safe_key,
                    reason="idempotency_key_payload_conflict",
                )
            return OutboxEnqueueResult(
                enqueued=False,
                already_exists=True,
                status=existing.status,
                idempotency_key=safe_key,
            )

        raw_chunks = (
            [safe_payload[i : i + 4000] for i in range(0, len(safe_payload), 4000)]
            if safe_payload
            else [""]
        )
        chunks_docs = [
            {
                "chunk_index": i,
                "text": chunk_str,
                "sent": False,
                "in_flight": False,
                "message_id": None,
                "sent_at": None,
            }
            for i, chunk_str in enumerate(raw_chunks)
        ]

        new_doc = {
            "idempotency_key": safe_key,
            "notification_type": safe_type,
            "run_id": safe_run_id,
            "job_id": safe_job_id,
            "payload_hash": payload_hash,
            "chunks": chunks_docs,
            "status": OutboxStatus.PENDING.value,
            "attempts": 0,
            "max_attempts": settings.outbox_max_attempts,
            "next_attempt_at": now,
            "lease_id": None,
            "lease_expires_at": None,
            "error_reason": None,
            "created_at": now,
            "updated_at": now,
            "sent_at": None,
            "metadata": safe_meta,
        }

        try:
            res = self.outbox_col.insert_one(new_doc)
            if inspect.isawaitable(res):
                await res
            return OutboxEnqueueResult(
                enqueued=True,
                already_exists=False,
                status=OutboxStatus.PENDING,
                idempotency_key=safe_key,
            )
        except Exception:
            re_read = await self.get_record(safe_key)
            if re_read is not None:
                if (
                    re_read.notification_type != safe_type
                    or re_read.run_id != safe_run_id
                    or re_read.job_id != safe_job_id
                    or re_read.payload_hash != payload_hash
                ):
                    return OutboxEnqueueResult(
                        enqueued=False,
                        already_exists=True,
                        status=re_read.status,
                        idempotency_key=safe_key,
                        reason="idempotency_key_payload_conflict",
                    )
                return OutboxEnqueueResult(
                    enqueued=False,
                    already_exists=True,
                    status=re_read.status,
                    idempotency_key=safe_key,
                )
            raise TelegramStorageError("Failed to enqueue outbox notification during race")

    async def claim_due_records(
        self,
        lease_duration_seconds: Optional[int] = None,
        limit: int = 10,
    ) -> OutboxClaimResult:
        """Atomically claim due pending records using configured lease duration."""
        await self.ensure_indexes()
        await self.reconcile_expired_sending_records()

        now = datetime.now(timezone.utc)
        settings = get_settings()
        lease_dur = lease_duration_seconds or settings.outbox_lease_seconds

        query = {
            "status": OutboxStatus.PENDING.value,
            "lease_id": None,
            "$or": [
                {"next_attempt_at": None},
                {"next_attempt_at": {"$lte": now}},
            ],
        }

        cursor = self.outbox_col.find(query).limit(limit)
        if inspect.isawaitable(cursor):
            cursor = await cursor

        candidates = []
        if hasattr(cursor, "to_list"):
            candidates = await cursor.to_list(length=limit)
        elif hasattr(cursor, "__aiter__"):
            async for doc in cursor:
                candidates.append(doc)
        elif isinstance(cursor, list):
            candidates = cursor

        claimed: list[OutboxRecord] = []
        for doc in candidates:
            doc_key = doc.get("idempotency_key")
            record_lease_id = f"out_{uuid4().hex[:12]}"
            lease_expires = now + timedelta(seconds=max(10, lease_dur))

            claim_query = {
                "idempotency_key": doc_key,
                "status": OutboxStatus.PENDING.value,
                "lease_id": None,
                "$or": [
                    {"next_attempt_at": None},
                    {"next_attempt_at": {"$lte": now}},
                ],
            }
            claim_res = self.outbox_col.find_one_and_update(
                claim_query,
                {
                    "$set": {
                        "status": OutboxStatus.SENDING.value,
                        "lease_id": record_lease_id,
                        "lease_expires_at": lease_expires,
                        "updated_at": now,
                    }
                },
                return_document=True if hasattr(self.outbox_col, "find_one_and_update") else False,
            )
            if inspect.isawaitable(claim_res):
                claim_res = await claim_res
            if claim_res:
                try:
                    claimed.append(OutboxRecord(**claim_res))
                except Exception as exc:
                    raise TelegramStorageError(f"Malformed outbox record ({type(exc).__name__})")

        return OutboxClaimResult(
            claimed=len(claimed) > 0,
            records=claimed,
        )

    async def mark_chunk_in_flight(
        self, idempotency_key: str, chunk_index: int, lease_id: str
    ) -> bool:
        """Mark chunk in-flight before network attempt under exact lease and unsent state."""
        await self.ensure_indexes()
        safe_key = bound_string(idempotency_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        now = datetime.now(timezone.utc)
        res = self.outbox_col.update_one(
            {
                "idempotency_key": safe_key,
                "status": OutboxStatus.SENDING.value,
                "lease_id": safe_lease,
                "chunks": {
                    "$elemMatch": {
                        "chunk_index": chunk_index,
                        "sent": False,
                        "in_flight": False,
                    }
                },
            },
            {"$set": {"chunks.$.in_flight": True, "updated_at": now}},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def mark_chunk_sent(
        self,
        idempotency_key: str,
        chunk_index: int,
        message_id: int,
        lease_id: str,
    ) -> bool:
        """Atomically mark an in-flight chunk as sent under exact lease."""
        await self.ensure_indexes()
        safe_key = bound_string(idempotency_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        now = datetime.now(timezone.utc)
        res = self.outbox_col.update_one(
            {
                "idempotency_key": safe_key,
                "status": OutboxStatus.SENDING.value,
                "lease_id": safe_lease,
                "chunks": {
                    "$elemMatch": {
                        "chunk_index": chunk_index,
                        "sent": False,
                        "in_flight": True,
                    }
                },
            },
            {
                "$set": {
                    "chunks.$.sent": True,
                    "chunks.$.in_flight": False,
                    "chunks.$.message_id": int(message_id),
                    "chunks.$.sent_at": now,
                    "updated_at": now,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def mark_record_sent(
        self,
        idempotency_key: str,
        lease_id: str,
    ) -> bool:
        """Mark the entire outbox record as sent using a strict no-unsent-chunk predicate."""
        await self.ensure_indexes()
        safe_key = bound_string(idempotency_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        now = datetime.now(timezone.utc)

        res = self.outbox_col.update_one(
            {
                "idempotency_key": safe_key,
                "status": OutboxStatus.SENDING.value,
                "lease_id": safe_lease,
                "chunks": {"$not": {"$elemMatch": {"sent": False}}},
            },
            {
                "$set": {
                    "status": OutboxStatus.SENT.value,
                    "sent_at": now,
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def mark_record_failed(
        self,
        idempotency_key: str,
        error_reason: str,
        lease_id: str,
        current_attempts: int,
        reset_in_flight_chunk: Optional[int] = None,
    ) -> bool:
        """Atomically increment attempts with CAS and schedule bounded exponential backoff or terminal failure."""
        await self.ensure_indexes()
        safe_key = bound_string(idempotency_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        safe_reason = bound_string(redact_string(error_reason), max_length=256)
        now = datetime.now(timezone.utc)
        settings = get_settings()

        rec = await self.get_record(safe_key)
        if not rec:
            return False

        next_attempts = current_attempts + 1
        if next_attempts >= rec.max_attempts:
            target_status = OutboxStatus.FAILED.value
            next_attempt_at = None
        else:
            target_status = OutboxStatus.PENDING.value
            backoff_base = min(
                settings.outbox_max_backoff_seconds,
                settings.outbox_initial_backoff_seconds * (2 ** (next_attempts - 1)),
            )
            next_attempt_at = now + timedelta(seconds=backoff_base)

        set_dict: dict[str, Any] = {
            "status": target_status,
            "next_attempt_at": next_attempt_at,
            "error_reason": safe_reason,
            "updated_at": now,
            "lease_id": None,
            "lease_expires_at": None,
        }
        if reset_in_flight_chunk is not None:
            set_dict["chunks.$.in_flight"] = False

        query: dict[str, Any] = {
            "idempotency_key": safe_key,
            "status": OutboxStatus.SENDING.value,
            "lease_id": safe_lease,
            "attempts": current_attempts,
        }
        if reset_in_flight_chunk is not None:
            query["chunks"] = {
                "$elemMatch": {
                    "chunk_index": reset_in_flight_chunk,
                    "sent": False,
                    "in_flight": True,
                }
            }

        res = self.outbox_col.update_one(
            query,
            {
                "$inc": {"attempts": 1},
                "$set": set_dict,
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def mark_delivery_unknown(
        self,
        idempotency_key: str,
        reason: str,
        lease_id: str,
    ) -> bool:
        """Record delivery-unknown state requiring exact SENDING status and lease."""
        await self.ensure_indexes()
        safe_key = bound_string(idempotency_key, max_length=160)
        safe_lease = bound_string(lease_id, max_length=64)
        safe_reason = bound_string(redact_string(reason), max_length=256)
        now = datetime.now(timezone.utc)
        query: dict[str, Any] = {
            "idempotency_key": safe_key,
            "status": OutboxStatus.SENDING.value,
            "lease_id": safe_lease,
        }

        res = self.outbox_col.update_one(
            query,
            {
                "$set": {
                    "status": OutboxStatus.DELIVERY_UNKNOWN.value,
                    "error_reason": safe_reason,
                    "updated_at": now,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = getattr(res, "matched_count", 0) if res else 0
        return matched > 0

    async def get_pending_and_unknown_counts(self) -> tuple[int, int]:
        """Return counts of currently pending/sending and delivery-unknown records without swallowing errors."""
        await self.ensure_indexes()
        res_pending = self.outbox_col.count_documents(
            {"status": {"$in": [OutboxStatus.PENDING.value, OutboxStatus.SENDING.value]}}
        )
        if inspect.isawaitable(res_pending):
            res_pending = await res_pending

        res_unknown = self.outbox_col.count_documents(
            {"status": OutboxStatus.DELIVERY_UNKNOWN.value}
        )
        if inspect.isawaitable(res_unknown):
            res_unknown = await res_unknown

        return int(res_pending or 0), int(res_unknown or 0)

    @classmethod
    async def close(cls) -> None:
        """Reset repository lifecycle state."""
        await TelegramPersistenceManager.close()
