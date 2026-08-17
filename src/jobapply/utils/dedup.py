"""MongoDB-backed deduplication store."""

import asyncio
import concurrent.futures
import hashlib
import inspect
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any, ClassVar, cast

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import UpdateOne

from jobapply.settings import get_settings
from jobapply.utils.redaction import redact_data, redact_string


class DeduplicationStoreError(RuntimeError):
    """Raised when deduplication store invariant or initialization fails."""

    pass


def bound_string(s: Any, max_length: int = 512) -> str:
    """Redact secrets, collapse whitespace, and truncate string to max_length."""
    if not isinstance(s, str):
        s = str(s)
    redacted = redact_string(s)
    collapsed = " ".join(redacted.split())
    if len(collapsed) > max_length:
        return collapsed[:max_length]
    return collapsed


def canonicalize_job_id(job_id: Any) -> str:
    """Canonicalize, redact, and bound a job ID string to <=128 chars.

    If length <= 128, returns the normalized string as-is.
    If length > 128, preserves a readable prefix plus an underscore and a 16-char
    SHA-256 hex digest to ensure distinct long IDs remain distinct and collision-free.
    """
    if not job_id:
        return ""
    if not isinstance(job_id, str):
        job_id = str(job_id)
    redacted = redact_string(job_id)
    collapsed = " ".join(redacted.split())
    if len(collapsed) <= 128:
        return collapsed

    digest = hashlib.sha256(collapsed.encode("utf-8")).hexdigest()[:16]
    prefix = collapsed[:111]
    return f"{prefix}_{digest}"


def _bound_recursive(
    value: Any,
    max_depth: int = 4,
    max_string_len: int = 512,
    max_items: int = 50,
    max_dict_keys: int = 50,
    max_key_len: int = 64,
) -> Any:
    """Recursively redact and bound any data structure to prevent unbounded documents."""
    if max_depth <= 0:
        return "[TRUNCATED_DEPTH]"

    if isinstance(value, str):
        return bound_string(value, max_length=max_string_len)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, datetime):
        return value

    if isinstance(value, (list, tuple, set)):
        bounded_seq = []
        for item in list(value)[:max_items]:
            bounded_seq.append(
                _bound_recursive(
                    item,
                    max_depth=max_depth - 1,
                    max_string_len=max_string_len,
                    max_items=max_items,
                    max_dict_keys=max_dict_keys,
                    max_key_len=max_key_len,
                )
            )
        return bounded_seq

    if isinstance(value, dict):
        bounded_dict = {}
        for k, v in list(value.items())[:max_dict_keys]:
            bounded_k = bound_string(str(k), max_length=max_key_len)
            bounded_dict[bounded_k] = _bound_recursive(
                v,
                max_depth=max_depth - 1,
                max_string_len=max_string_len,
                max_items=max_items,
                max_dict_keys=max_dict_keys,
                max_key_len=max_key_len,
            )
        return bounded_dict

    return bound_string(str(value), max_length=max_string_len)


def bound_and_redact_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Redact secrets and enforce strict recursive bounds on metadata fields before persistence."""
    if not isinstance(meta, dict):
        return {}

    redacted = redact_data(meta)
    bounded: dict[str, Any] = {}

    TOP_LEVEL_STRING_LIMITS = {
        "job_id": 128,
        "status": 64,
        "applied_status": 256,
        "title": 256,
        "company": 256,
        "location": 256,
        "parsed_location": 256,
        "work_type": 128,
        "duration": 128,
        "reason": 1024,
        "qualification_reasoning": 1024,
        "job_summary": 1024,
        "url": 512,
        "description": 16384,
    }

    TOP_LEVEL_LIST_LIMITS = {
        "disallowed_languages": (20, 128),
        "required_languages": (20, 128),
        "key_matches": (50, 256),
        "gaps": (50, 256),
        "responsibilities": (50, 512),
        "requirements": (50, 512),
    }

    for k, v in list(redacted.items())[:50]:
        k_bounded = bound_string(str(k), max_length=64)
        if isinstance(v, str):
            limit = TOP_LEVEL_STRING_LIMITS.get(k, 1024)
            bounded[k_bounded] = bound_string(v, max_length=limit)
        elif isinstance(v, (list, tuple, set)):
            max_items, item_len = TOP_LEVEL_LIST_LIMITS.get(k, (50, 256))
            bounded_list = []
            for item in list(v)[:max_items]:
                bounded_list.append(
                    _bound_recursive(
                        item,
                        max_depth=4,
                        max_string_len=item_len,
                        max_items=max_items,
                    )
                )
            bounded[k_bounded] = bounded_list
        elif isinstance(v, dict):
            bounded[k_bounded] = _bound_recursive(
                v,
                max_depth=4,
                max_string_len=512,
                max_items=50,
                max_dict_keys=50,
            )
        elif isinstance(v, (int, float, bool)) or v is None or isinstance(v, datetime):
            bounded[k_bounded] = v
        else:
            bounded[k_bounded] = bound_string(str(v), max_length=512)

    return bounded


async def _consume_seen_id_cursor(cursor: Any) -> set[str]:
    """Consume job_id values from a Motor async cursor, awaitable result, or iterable mock."""
    if hasattr(cursor, "__aiter__"):
        return {doc["job_id"] async for doc in cursor if isinstance(doc, dict) and "job_id" in doc}
    if inspect.isawaitable(cursor):
        docs = await cursor
        if isinstance(docs, (list, tuple, set)):
            return {doc["job_id"] for doc in docs if isinstance(doc, dict) and "job_id" in doc}
        if hasattr(docs, "__aiter__"):
            return {
                doc["job_id"] async for doc in docs if isinstance(doc, dict) and "job_id" in doc
            }
    if isinstance(cursor, Iterable):
        return {doc["job_id"] for doc in cursor if isinstance(doc, dict) and "job_id" in doc}
    if hasattr(cursor, "__iter__"):
        iterable_cursor = cast(Iterable[Any], cursor)
        return {
            doc["job_id"] for doc in iterable_cursor if isinstance(doc, dict) and "job_id" in doc
        }
    return set()


class DeduplicationStore:
    """Persistent tracking of seen job IDs across sessions."""

    _client: ClassVar[AsyncIOMotorClient | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _init_future: ClassVar[concurrent.futures.Future | None] = None
    _index_succeeded: ClassVar[bool] = False
    _index_error: ClassVar[str | None] = None

    def __init__(self):
        """Initialize deduplication store with MongoDB connection."""
        settings = get_settings()
        if self.__class__._client is None:
            self.__class__._client = AsyncIOMotorClient(settings.mongodb_url)
        self.client = self.__class__._client
        self.collection = self.client[settings.mongodb_db][settings.seen_jobs_collection]

    async def ensure_indexes(self) -> bool:
        """Ensure unique index on job_id once per process/lifecycle, loop-neutrally.

        Raises:
            DeduplicationStoreError: If index initialization fails.
        """
        with self._lock:
            if self._index_succeeded:
                return True
            if self._index_error is not None:
                raise DeduplicationStoreError(self._index_error)

            fut = self._init_future
            if fut is None:
                fut = concurrent.futures.Future()
                self.__class__._init_future = fut
                is_initiator = True
            else:
                is_initiator = False

        if is_initiator:
            try:
                if hasattr(self.collection, "create_index"):
                    res = self.collection.create_index("job_id", unique=True)
                    if inspect.isawaitable(res):
                        await res
                with self._lock:
                    self.__class__._index_succeeded = True
                    self.__class__._index_error = None
                fut.set_result(True)
                return True
            except Exception as e:
                sanitized_err = f"Deduplication index initialization failed ({type(e).__name__})"
                with self._lock:
                    self.__class__._index_succeeded = False
                    self.__class__._index_error = sanitized_err
                fut.set_result(False)
                raise DeduplicationStoreError(sanitized_err)
        else:
            # Concurrent callers share and await the same in-flight initialization
            loop = asyncio.get_running_loop()
            await asyncio.wrap_future(fut, loop=loop)
            with self._lock:
                if self._index_succeeded:
                    return True
                raise DeduplicationStoreError(
                    self._index_error or "Deduplication index initialization failed"
                )

    async def is_seen(self, job_id: str) -> bool:
        """Check if a job ID has been seen before.

        Args:
            job_id: Exact job ID to check.

        Returns:
            True if job has been seen, False otherwise.
        """
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id:
            return False
        return await self.collection.find_one({"job_id": canonical_id}) is not None

    async def find_seen_ids(
        self,
        job_ids: Sequence[str] | set[str] | list[str],
        for_live_run: bool = False,
    ) -> set[str]:
        """Find which of the given job IDs have been seen before using a single batched query.

        Args:
            job_ids: Collection of job IDs to check.
            for_live_run: If True, jobs with status 'qualified' or 'dry_run' are excluded
                          from seen results so they remain eligible for a live run.

        Returns:
            Set of seen job IDs matching the query.
        """
        seen_ids: set[str] = set()
        for jid in job_ids:
            c_id = canonicalize_job_id(jid)
            if c_id:
                seen_ids.add(c_id)

        if not seen_ids:
            return set()

        await self.ensure_indexes()
        query: dict[str, Any] = {"job_id": {"$in": list(seen_ids)}}
        if for_live_run:
            query["status"] = {"$nin": ["qualified", "dry_run"]}

        cursor: Any = self.collection.find(query, {"job_id": 1})
        return await _consume_seen_id_cursor(cursor)

    async def mark_seen(self, job_id: str, metadata: dict) -> None:
        """Mark a single job as seen with metadata. Function job_id parameter is authoritative.

        Args:
            job_id: Exact job ID to mark as seen (authoritative over any metadata['job_id']).
            metadata: Additional metadata to store (title, company, url, qualification data, etc.).
        """
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id:
            return

        bounded_meta = bound_and_redact_metadata(metadata)
        bounded_meta["job_id"] = canonical_id
        now = datetime.now(timezone.utc)
        doc = {
            "job_id": canonical_id,
            **bounded_meta,
            "evaluated_at": bounded_meta.get("evaluated_at", now),
            "updated_at": now,
        }
        res = self.collection.update_one(
            {"job_id": canonical_id},
            {"$set": doc},
            upsert=True,
        )
        if inspect.isawaitable(res):
            await res

    async def mark_seen_many(
        self,
        items: Sequence[dict[str, Any]] | Sequence[tuple[str, dict[str, Any]]],
    ) -> None:
        """Mark multiple jobs as seen in bulk using MongoDB bulk operations.

        Args:
            items: Sequence of metadata dicts containing 'job_id', or (job_id, metadata) pairs.
        """
        if not items:
            return

        await self.ensure_indexes()
        now = datetime.now(timezone.utc)

        # Pre-canonicalize job IDs first, then deduplicate by canonical ID
        deduped: dict[str, dict[str, Any]] = {}
        for item in items:
            if isinstance(item, tuple):
                raw_id, raw_meta = item
                safe = dict(raw_meta) if isinstance(raw_meta, dict) else {}
                canonical_id = canonicalize_job_id(raw_id)
            elif isinstance(item, dict):
                raw_id = item.get("job_id")
                safe = dict(item)
                canonical_id = canonicalize_job_id(raw_id)
            else:
                continue

            if not canonical_id:
                continue

            bounded_meta = bound_and_redact_metadata(safe)
            bounded_meta["job_id"] = canonical_id
            deduped[canonical_id] = bounded_meta

        if not deduped:
            return

        normalized_updates: list[tuple[str, dict[str, Any]]] = []
        operations: list[UpdateOne] = []

        for canonical_id, bounded_meta in deduped.items():
            doc = {
                "job_id": canonical_id,
                **bounded_meta,
                "evaluated_at": bounded_meta.get("evaluated_at", now),
                "updated_at": now,
            }
            normalized_updates.append((canonical_id, doc))
            operations.append(
                UpdateOne(
                    {"job_id": canonical_id},
                    {"$set": doc},
                    upsert=True,
                )
            )

        if hasattr(self.collection, "bulk_write"):
            res = self.collection.bulk_write(operations, ordered=False)
            if inspect.isawaitable(res):
                await res
        elif hasattr(self.collection, "update_one"):
            # Fallback for minimal collection mocks without bulk_write, avoiding private attributes
            for canonical_id, doc in normalized_updates:
                res = self.collection.update_one(
                    {"job_id": canonical_id}, {"$set": doc}, upsert=True
                )
                if inspect.isawaitable(res):
                    await res

    async def load_seen_ids(self, for_live_run: bool = False) -> set[str]:
        """Load all seen job IDs into memory for fast lookup.

        Returns:
            Set of all seen job IDs.
        """
        await self.ensure_indexes()
        query = {"status": {"$nin": ["qualified", "dry_run"]}} if for_live_run else {}
        cursor: Any = self.collection.find(query, {"job_id": 1})
        return await _consume_seen_id_cursor(cursor)

    async def get_daily_count(self) -> int:
        """Count applications submitted or reserved today.

        Returns:
            Number of applications submitted or actively reserved today.
        """
        await self.ensure_indexes()
        today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        submitted_count = await self.collection.count_documents(
            {
                "applied_at": {"$gte": today},
                "status": "submitted",
            }
        )
        try:
            from jobapply.utils.attempts import QuotaRepository

            quota_consumed = await QuotaRepository().get_daily_consumed()
            return max(submitted_count, quota_consumed)
        except Exception:
            return submitted_count

    @classmethod
    async def close(cls) -> None:
        """Close the shared MongoDB client at application shutdown and reset lifecycle state."""
        with cls._lock:
            if cls._client is not None:
                cls._client.close()
                cls._client = None
            cls._init_future = None
            cls._index_succeeded = False
            cls._index_error = None
        try:
            from jobapply.utils.attempts import AttemptRepository, QuotaRepository
            from jobapply.utils.telegram_storage import (
                NotificationOutboxRepository,
                TelegramRepository,
            )

            await AttemptRepository.close()
            await QuotaRepository.close()
            await TelegramRepository.close()
            await NotificationOutboxRepository.close()
        except Exception:
            pass
