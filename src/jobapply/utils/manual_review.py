"""Durable MongoDB-backed manual-review queue with idempotent enqueue and CAS transitions.

The repository follows the existing shared ``MongoClientManager`` lifecycle and
fail-closed index conventions used by the attempts and Telegram repositories.
Queue persistence failures never convert a manual-review outcome into success:
callers wrap enqueues in :func:`safe_enqueue_manual_review`, which degrades any
storage error into a bounded structured log while the outcome itself stays
``needs_manual_review``.

Acknowledgement/resolution are atomic compare-and-set state transitions and can
never increment application counts or alter submission truth. Resolving a
``submission_unknown`` item never marks it submitted; it requires an explicit
safe resolution label and triggers no browser retry.
"""

import asyncio
import concurrent.futures
import hashlib
import inspect
import threading
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

from jobapply.models.manual_review import (
    ManualReviewEnqueueResult,
    ManualReviewItem,
    ManualReviewState,
    ManualReviewTransitionResult,
    ResolutionLabel,
)
from jobapply.settings import get_settings
from jobapply.utils.dedup import bound_and_redact_metadata, bound_string, canonicalize_job_id
from jobapply.utils.mongo import MongoClientManager
from jobapply.utils.observability import log_event
from jobapply.utils.redaction import redact_string


class ManualReviewStorageError(RuntimeError):
    """Raised when manual-review queue storage operations fail."""

    pass


def make_manual_review_idempotency_key(
    run_id: str,
    job_id: Optional[str],
    reason_category: str,
    reason_detail: str = "",
) -> str:
    """Deterministic collision-resistant key stable across resume/replay."""
    safe_run = bound_string(run_id or "unknown", max_length=64)
    safe_job = canonicalize_job_id(job_id) if job_id else "no_job"
    safe_category = bound_string(reason_category or "unknown", max_length=80)
    safe_detail_hash = hashlib.sha256(
        bound_string(redact_string(reason_detail or ""), max_length=256).encode("utf-8")
    ).hexdigest()[:16]
    raw = f"mr:{safe_run}:{safe_job}:{safe_category}:{safe_detail_hash}"
    return f"mr_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"


def sanitize_untrusted_text(value: Any, max_length: int) -> str:
    """Redact secrets, collapse whitespace, and bound one untrusted free-text field.

    Used at the trust boundary before any hostile string reaches a durable
    queue document, checkpoint state, or operator output.
    """
    if value is None:
        return ""
    return bound_string(redact_string(str(value)), max_length=max_length)


def _extract_attempt_fields(
    outcome: dict, extra: Optional[dict] = None
) -> tuple[str | None, str | None, bool]:
    """Pull bounded attempt identifiers/ambiguity from an outcome payload."""
    merged: dict[str, Any] = {}
    nested_extra = outcome.get("extra")
    if isinstance(nested_extra, dict):
        merged.update(nested_extra)
    if isinstance(extra, dict):
        merged.update(extra)
    for key in ("attempt_id", "attempt_status", "ambiguous_submission"):
        if key in outcome and outcome.get(key) is not None:
            merged.setdefault(key, outcome.get(key))
    attempt_id = merged.get("attempt_id")
    attempt_status = merged.get("attempt_status")
    return (
        str(attempt_id) if attempt_id else None,
        str(attempt_status) if attempt_status else None,
        bool(merged.get("ambiguous_submission")),
    )


def manual_review_item_from_outcome(
    run_id: str,
    outcome: dict,
    extra: Optional[dict] = None,
) -> ManualReviewItem:
    """Build the canonical queue item for one needs_manual_review outcome.

    Shared by the live enqueue funnel and by central recovery/reconstruction so
    both paths derive byte-identical deterministic idempotency keys. Every
    untrusted free-text field is redacted and bounded here.
    """
    from jobapply.nodes.outcomes import job_repost_persistence_fields

    attempt_id, attempt_status, ambiguous = _extract_attempt_fields(outcome, extra)
    repost_fields = job_repost_persistence_fields(
        {
            "is_repost": outcome.get("is_repost"),
            "repost_evidence": outcome.get("repost_evidence"),
        }
    )
    return build_manual_review_item(
        run_id=run_id or "default_run",
        job_id=outcome.get("job_id"),
        title=outcome.get("title"),
        company=outcome.get("company"),
        url=outcome.get("url"),
        reason_category=str(outcome.get("reason") or "manual_review"),
        reason_detail=str(outcome.get("error") or "") or None,
        attempt_id=attempt_id,
        attempt_status=attempt_status,
        ambiguous_submission=ambiguous,
        source_outcome={
            key: value
            for key, value in {
                "status": outcome.get("status"),
                "score": outcome.get("score"),
                "qa_count": outcome.get("qa_count"),
                "reason": outcome.get("reason"),
                "dry_run": outcome.get("dry_run"),
                **repost_fields,
            }.items()
            if value is not None
        },
    )


def build_manual_review_item(
    *,
    run_id: str,
    job_id: Optional[str],
    title: Optional[str],
    company: Optional[str],
    url: Optional[str],
    reason_category: str,
    reason_detail: Optional[str] = None,
    attempt_id: Optional[str] = None,
    attempt_status: Optional[str] = None,
    ambiguous_submission: bool = False,
    source_outcome: Optional[dict[str, Any]] = None,
) -> ManualReviewItem:
    """Build a bounded, redacted queue item with its deterministic item ID.

    Every untrusted free-text field (title, company, URL, reason detail,
    attempt fields) is redacted before bounding so credentials or sensitive
    URLs from application errors can never reach Mongo or state. The
    idempotency key is computed after sanitization and stays deterministic.
    """
    safe_run_id = sanitize_untrusted_text(run_id or "unknown", 64)
    idempotency_key = make_manual_review_idempotency_key(
        safe_run_id,
        canonicalize_job_id(job_id) if job_id else None,
        sanitize_untrusted_text(reason_category, 120),
        sanitize_untrusted_text(reason_detail, 256),
    )
    now = datetime.now(timezone.utc)
    return ManualReviewItem(
        item_id=idempotency_key,
        idempotency_key=idempotency_key,
        run_id=safe_run_id,
        job_id=canonicalize_job_id(job_id) if job_id else None,
        title=sanitize_untrusted_text(title, max_length=256) or None,
        company=sanitize_untrusted_text(company, max_length=256) or None,
        url=sanitize_untrusted_text(url, max_length=512) or None,
        reason_category=sanitize_untrusted_text(reason_category, max_length=120),
        reason_detail=sanitize_untrusted_text(reason_detail, max_length=1024) or None,
        attempt_id=sanitize_untrusted_text(attempt_id, max_length=64) or None,
        attempt_status=sanitize_untrusted_text(attempt_status, max_length=64) or None,
        ambiguous_submission=bool(ambiguous_submission),
        created_at=now,
        updated_at=now,
        source_outcome=bound_and_redact_metadata(source_outcome or {}),
    )


class ManualReviewRepository:
    """Repository for the durable manual-review queue in MongoDB."""

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _init_future: ClassVar[Optional[concurrent.futures.Future]] = None
    _index_succeeded: ClassVar[bool] = False
    _index_error: ClassVar[Optional[str]] = None

    def __init__(self):
        settings = get_settings()
        self.client = MongoClientManager.get_client()
        self.collection = self.client[settings.mongodb_db][settings.manual_review_collection]

    async def ensure_indexes(self) -> bool:
        """Ensure unique idempotency/item indexes once per process/lifecycle, loop-neutrally."""
        with self._lock:
            if self._index_succeeded:
                return True
            if self._index_error is not None:
                raise ManualReviewStorageError(self._index_error)

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
                    res1 = self.collection.create_index("idempotency_key", unique=True)
                    if inspect.isawaitable(res1):
                        await res1
                    res2 = self.collection.create_index([("state", 1), ("created_at", -1)])
                    if inspect.isawaitable(res2):
                        await res2
                    res3 = self.collection.create_index([("run_id", 1), ("created_at", -1)])
                    if inspect.isawaitable(res3):
                        await res3
                with self._lock:
                    self.__class__._index_succeeded = True
                    self.__class__._index_error = None
                fut.set_result(True)
                return True
            except Exception as e:
                sanitized_err = f"Manual review index initialization failed ({type(e).__name__})"
                with self._lock:
                    self.__class__._index_succeeded = False
                    self.__class__._index_error = sanitized_err
                fut.set_result(False)
                raise ManualReviewStorageError(sanitized_err)
        else:
            loop = asyncio.get_running_loop()
            await asyncio.wrap_future(fut, loop=loop)
            with self._lock:
                if self._index_succeeded:
                    return True
                raise ManualReviewStorageError(
                    self._index_error or "Manual review index initialization failed"
                )

    @classmethod
    def _reset_state(cls) -> None:
        """Synchronously reset index lifecycle states (registered reset hook)."""
        with cls._lock:
            cls._init_future = None
            cls._index_succeeded = False
            cls._index_error = None

    @classmethod
    async def close(cls) -> None:
        """Close shared MongoDB client and reset all repository lifecycles."""
        await MongoClientManager.close()

    async def enqueue(self, item: ManualReviewItem) -> ManualReviewEnqueueResult:
        """Idempotently insert a queue item keyed by its deterministic idempotency key.

        Duplicate inserts across resume/replay resolve to the existing record
        without creating a second document. Payload conflicts are reported, not
        silently overwritten.
        """
        await self.ensure_indexes()
        doc = item.model_dump()
        try:
            res = self.collection.insert_one(doc)
            if inspect.isawaitable(res):
                await res
            return ManualReviewEnqueueResult(queued=True, item_id=item.item_id)
        except Exception as first_error:
            existing = await self.get_item(item.item_id)
            if existing is not None:
                if existing.idempotency_key != item.idempotency_key:
                    return ManualReviewEnqueueResult(
                        queued=False,
                        already_exists=True,
                        item_id=item.item_id,
                        reason="idempotency_key_conflict",
                    )
                return ManualReviewEnqueueResult(
                    queued=False,
                    already_exists=True,
                    item_id=item.item_id,
                )
            return ManualReviewEnqueueResult(
                queued=False,
                item_id=item.item_id,
                reason=f"insert_failed ({type(first_error).__name__})",
            )

    async def get_item(self, item_id: str) -> Optional[ManualReviewItem]:
        """Fetch one bounded queue item by ID, failing closed on malformed docs."""
        await self.ensure_indexes()
        safe_id = bound_string(item_id, max_length=160)
        if not safe_id:
            return None
        res = self.collection.find_one({"item_id": safe_id})
        if inspect.isawaitable(res):
            res = await res
        if not res:
            return None
        try:
            return ManualReviewItem(**res)
        except Exception as exc:
            raise ManualReviewStorageError(
                f"Malformed manual-review document ({type(exc).__name__})"
            )

    async def list_items(
        self, limit: int = 20, state: Optional[ManualReviewState] = None
    ) -> list[ManualReviewItem]:
        """List bounded queue items newest-first, redacted by construction."""
        await self.ensure_indexes()
        bounded_limit = max(1, min(int(limit or 20), 100))
        query: dict[str, Any] = {}
        if state is not None:
            query["state"] = state.value
        cursor = self.collection.find(query).limit(bounded_limit)
        if inspect.isawaitable(cursor):
            cursor = await cursor
        docs: list[dict[str, Any]] = []
        if hasattr(cursor, "to_list"):
            docs = await cursor.to_list(length=bounded_limit)
        else:
            async for doc in cursor:
                docs.append(doc)
        docs.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
        items: list[ManualReviewItem] = []
        for doc in docs[:bounded_limit]:
            try:
                items.append(ManualReviewItem(**doc))
            except Exception as exc:
                raise ManualReviewStorageError(
                    f"Malformed manual-review document ({type(exc).__name__})"
                )
        return items

    async def acknowledge(
        self, item_id: str, note: Optional[str] = None
    ) -> ManualReviewTransitionResult:
        """Atomically transition open -> acknowledged (CAS under exact prior state)."""
        await self.ensure_indexes()
        safe_id = bound_string(item_id, max_length=160)
        if not safe_id:
            return ManualReviewTransitionResult(
                success=False, item_id=item_id, reason="invalid_item_id"
            )
        now = datetime.now(timezone.utc)
        set_fields: dict[str, Any] = {
            "state": ManualReviewState.ACKNOWLEDGED.value,
            "updated_at": now,
            "acknowledged_at": now,
        }
        if note is not None:
            bounded_note = bound_string(redact_string(note), max_length=512)
            if bounded_note:
                set_fields["operator_note"] = bounded_note

        res = self.collection.find_one_and_update(
            {"item_id": safe_id, "state": ManualReviewState.OPEN.value},
            {"$set": set_fields},
            return_document=True if hasattr(self.collection, "find_one_and_update") else False,
        )
        if inspect.isawaitable(res):
            res = await res
        if res:
            return ManualReviewTransitionResult(
                success=True,
                changed=True,
                item_id=safe_id,
                state=ManualReviewState.ACKNOWLEDGED,
            )
        return await self._transition_rejection(safe_id, expected_from=ManualReviewState.OPEN)

    async def resolve(
        self,
        item_id: str,
        note: Optional[str] = None,
        resolution_label: Optional[ResolutionLabel] = None,
    ) -> ManualReviewTransitionResult:
        """Atomically resolve open/acknowledged items without altering submission truth.

        Items flagged ambiguous_submission (e.g. attempt status submission_unknown)
        require an explicit safe resolution label; resolving never retries a
        browser flow and never increments application counts.
        """
        await self.ensure_indexes()
        safe_id = bound_string(item_id, max_length=160)
        if not safe_id:
            return ManualReviewTransitionResult(
                success=False, item_id=item_id, reason="invalid_item_id"
            )

        existing = await self.get_item(safe_id)
        if existing is None:
            return ManualReviewTransitionResult(
                success=False, not_found=True, item_id=safe_id, reason="not_found"
            )

        if existing.state == ManualReviewState.RESOLVED:
            same_label = (
                resolution_label is None
                or existing.resolution_label is None
                or existing.resolution_label == resolution_label
            )
            return ManualReviewTransitionResult(
                success=same_label,
                no_op=True,
                conflict=not same_label,
                item_id=safe_id,
                state=existing.state,
                reason=None if same_label else "already_resolved_with_different_label",
            )

        if existing.ambiguous_submission and resolution_label is None:
            return ManualReviewTransitionResult(
                success=False,
                conflict=True,
                item_id=safe_id,
                state=existing.state,
                reason="submission_unknown_requires_explicit_resolution_label",
            )

        now = datetime.now(timezone.utc)
        set_fields: dict[str, Any] = {
            "state": ManualReviewState.RESOLVED.value,
            "updated_at": now,
            "resolved_at": now,
        }
        if resolution_label is not None:
            set_fields["resolution_label"] = resolution_label.value
        if note is not None:
            bounded_note = bound_string(redact_string(note), max_length=512)
            if bounded_note:
                set_fields["operator_note"] = bounded_note

        res = self.collection.find_one_and_update(
            {
                "item_id": safe_id,
                "state": {
                    "$in": [ManualReviewState.OPEN.value, ManualReviewState.ACKNOWLEDGED.value]
                },
            },
            {"$set": set_fields},
            return_document=True if hasattr(self.collection, "find_one_and_update") else False,
        )
        if inspect.isawaitable(res):
            res = await res
        if res:
            return ManualReviewTransitionResult(
                success=True,
                changed=True,
                item_id=safe_id,
                state=ManualReviewState.RESOLVED,
            )
        # Lost race against a concurrent transition.
        reread = await self.get_item(safe_id)
        if reread is not None and reread.state == ManualReviewState.RESOLVED:
            same_label = (
                resolution_label is None
                or reread.resolution_label is None
                or reread.resolution_label == resolution_label
            )
            return ManualReviewTransitionResult(
                success=same_label,
                no_op=True,
                conflict=not same_label,
                item_id=safe_id,
                state=reread.state,
                reason=None if same_label else "already_resolved_with_different_label",
            )
        return ManualReviewTransitionResult(
            success=False,
            conflict=True,
            item_id=safe_id,
            state=reread.state if reread else None,
            reason="concurrent_transition_lost",
        )

    async def _transition_rejection(
        self, safe_id: str, *, expected_from: ManualReviewState
    ) -> ManualReviewTransitionResult:
        existing = await self.get_item(safe_id)
        if existing is None:
            return ManualReviewTransitionResult(
                success=False, not_found=True, item_id=safe_id, reason="not_found"
            )
        if existing.state == expected_from:
            return ManualReviewTransitionResult(
                success=False,
                conflict=True,
                item_id=safe_id,
                state=existing.state,
                reason="concurrent_transition_lost",
            )
        if existing.state == ManualReviewState.RESOLVED:
            return ManualReviewTransitionResult(
                success=True, no_op=True, item_id=safe_id, state=existing.state
            )
        return ManualReviewTransitionResult(
            success=False,
            no_op=True,
            item_id=safe_id,
            state=existing.state,
            reason=f"item_already_{existing.state.value}",
        )


MongoClientManager.register_reset_hook(ManualReviewRepository._reset_state)

# Hard cap on pending queue items preserved in checkpoint state so a prolonged
# outage can never grow graph state unboundedly.
MAX_PENDING_MANUAL_REVIEW_ITEMS = 50


def serialize_pending_manual_review_item(item: ManualReviewItem) -> dict:
    """Bounded, redacted serialized form for durable checkpoint-state retention."""
    return {
        "item_id": sanitize_untrusted_text(item.item_id, 160),
        "idempotency_key": sanitize_untrusted_text(item.idempotency_key, 64),
        "run_id": sanitize_untrusted_text(item.run_id, 64),
        "job_id": canonicalize_job_id(item.job_id) if item.job_id else None,
        "title": sanitize_untrusted_text(item.title, 256) or None,
        "company": sanitize_untrusted_text(item.company, 256) or None,
        "url": sanitize_untrusted_text(item.url, 512) or None,
        "reason_category": sanitize_untrusted_text(item.reason_category, 120),
        "reason_detail": sanitize_untrusted_text(item.reason_detail, 1024) or None,
        "attempt_id": sanitize_untrusted_text(item.attempt_id, 64) or None,
        "attempt_status": sanitize_untrusted_text(item.attempt_status, 64) or None,
        "ambiguous_submission": bool(item.ambiguous_submission),
        "state": item.state.value,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


async def safe_enqueue_manual_review(item: ManualReviewItem) -> bool:
    """Best-effort durable enqueue that never raises into outcome recording.

    A storage failure here must never convert a manual-review outcome into a
    success; the caller's outcome keeps its needs_manual_review status and this
    helper records one bounded structured warning for operator follow-up.
    Callers retain the serialized item in checkpoint state (see
    :func:`flush_manual_review_queue`) so temporary outages stay retryable.
    """
    try:
        repo = ManualReviewRepository()
        result = await repo.enqueue(item)
        if not result.queued and not result.already_exists:
            log_event(
                "warning",
                "manual_review.enqueue_failed",
                "Manual-review queue persistence failed; outcome remains needs_manual_review.",
                run_id=item.run_id,
                node="manual_review_queue",
                details={"reason": bound_string(result.reason or "unknown", max_length=120)},
            )
            return False
        if result.already_exists:
            log_event(
                "debug",
                "manual_review.enqueue_duplicate",
                "Manual-review item already queued (idempotent replay).",
                run_id=item.run_id,
                node="manual_review_queue",
            )
        return True
    except Exception as exc:
        log_event(
            "warning",
            "manual_review.enqueue_exception",
            "Manual-review queue persistence raised; outcome remains needs_manual_review.",
            run_id=item.run_id,
            node="manual_review_queue",
            exc=exc,
        )
        return False


async def flush_manual_review_queue(state: Any) -> dict:
    """Retry pending entries and reconstruct evicted items from authoritative outcomes.

    Central safe boundary used before the final notification and on resumed
    execution. Two deterministic sources feed idempotent enqueues:

    1. explicit pending entries retained in checkpoint state (including
       construction-failure fallbacks, which parse into full items);
    2. every authoritative ``needs_manual_review`` application outcome,
       reconstructed through :func:`manual_review_item_from_outcome` — this is
       the real recovery path for entries evicted by bounded-pending overflow,
       because the outcome list itself is never truncated.

    Both sources share the same deterministic idempotency key, so an item is
    durably enqueued exactly once regardless of how many times it appears.
    Successfully handled pending entries are removed from state; reconstruction
    failures stay retryable via the authoritative outcomes on the next flush.
    Never raises; never touches submission truth or quotas.
    """
    raw_pending = list((state or {}).get("manual_review_queue_pending") or [])
    remaining: list[dict] = []
    seen_keys: set[str] = set()

    for raw in raw_pending[:MAX_PENDING_MANUAL_REVIEW_ITEMS]:
        try:
            item = ManualReviewItem(**raw)
        except Exception:
            # Malformed pending entries stay visible rather than disappearing.
            remaining.append(raw if isinstance(raw, dict) else {"raw": bound_string(str(raw), 200)})
            continue
        enqueued = await safe_enqueue_manual_review(item)
        if not enqueued:
            remaining.append(serialize_pending_manual_review_item(item))
        else:
            seen_keys.add(item.idempotency_key)

    # Reconstruction pass from authoritative outcomes (overflow recovery).
    outcomes = [
        outcome
        for outcome in ((state or {}).get("application_outcomes") or [])
        if isinstance(outcome, dict) and outcome.get("status") == "needs_manual_review"
    ]
    run_id = str((state or {}).get("run_id") or "default_run")
    for outcome in outcomes:
        try:
            item = manual_review_item_from_outcome(run_id, outcome)
        except Exception as exc:
            log_event(
                "warning",
                "manual_review.reconstruction_failed",
                "A manual-review outcome could not be reconstructed; it remains "
                "authoritative in application_outcomes for the next flush.",
                run_id=run_id or None,
                node="manual_review_queue",
                exc=exc,
            )
            continue
        if item.idempotency_key in seen_keys:
            continue  # already durably enqueued via pending path this flush
        enqueued = await safe_enqueue_manual_review(item)
        if not enqueued:
            seen_keys.discard(item.idempotency_key)

    if len(raw_pending) > MAX_PENDING_MANUAL_REVIEW_ITEMS:
        log_event(
            "warning",
            "manual_review.pending_overflow_dropped",
            "Pending manual-review items exceeded the durable cap; excess dropped.",
            node="manual_review_queue",
        )
    return {"manual_review_queue_pending": remaining}
