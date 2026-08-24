"""Idempotent live-application attempts and atomic quota reservation."""

import asyncio
import concurrent.futures
import hashlib
import inspect
import threading
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional
from uuid import uuid4

from pydantic import BaseModel

from jobapply.models.application import (
    LEGAL_ATTEMPT_TRANSITIONS,
    AttemptPreflightResult,
    AttemptStatus,
    AttemptTransitionResult,
)
from jobapply.settings import get_settings
from jobapply.utils.dedup import (
    bound_and_redact_metadata,
    bound_string,
    canonicalize_job_id,
)
from jobapply.utils.mongo import MongoClientManager


class AttemptError(RuntimeError):
    """Base error for application attempt operations."""

    pass


class AttemptTransitionError(AttemptError):
    """Raised when an illegal attempt state transition is attempted."""

    pass


class AttemptClaimError(AttemptError):
    """Raised when claiming an application attempt fails."""

    pass


class QuotaExceededError(AttemptError):
    """Raised when daily or session quota reservation fails."""

    pass


def make_safe_field_key(key: Any, prefix: str = "k_") -> str:
    """Generate a safe, collision-resistant Mongo field key with no '.' or '$' chars."""
    raw = str(key or "").strip()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}{digest}"


def get_utc_date_str(dt: Optional[datetime] = None) -> str:
    """Get UTC date string YYYY-MM-DD."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


class AttemptClaimResult(BaseModel):
    """Result of attempting to claim a live application attempt."""

    claimed: bool
    status: AttemptStatus
    attempt_id: str
    reason: Optional[str] = None
    existing_run_id: Optional[str] = None


class QuotaReservationResult(BaseModel):
    """Result of reserving daily and session quota."""

    success: bool
    already_reserved: bool = False
    daily_reserved: int = 0
    session_reserved: int = 0
    reason: Optional[str] = None


class QuotaReleaseResult(BaseModel):
    """Result of releasing a quota reservation."""

    success: bool
    changed: bool = False
    already_released: bool = False
    reason: Optional[str] = None


class QuotaCommitResult(BaseModel):
    """Result of committing a quota reservation to submitted."""

    success: bool
    changed: bool = False
    already_committed: bool = False
    reason: Optional[str] = None


def validate_attempt_transition(
    current_status: AttemptStatus,
    target_status: AttemptStatus,
) -> None:
    """Validate that transitioning from current_status to target_status is legal.

    Raises:
        AttemptTransitionError: If the transition is illegal or constitutes a regression.
    """
    if current_status == target_status:
        return

    allowed = LEGAL_ATTEMPT_TRANSITIONS.get(current_status, set())
    if target_status not in allowed:
        raise AttemptTransitionError(
            f"Illegal attempt transition from '{current_status.value}' to '{target_status.value}'"
        )


class AttemptRepository:
    """Repository for managing idempotent application attempts in MongoDB."""

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _init_future: ClassVar[Optional[concurrent.futures.Future]] = None
    _index_succeeded: ClassVar[bool] = False
    _index_error: ClassVar[Optional[str]] = None

    def __init__(self):
        """Initialize attempt repository with MongoDB connection."""
        settings = get_settings()
        self.client = MongoClientManager.get_client()
        self.collection = self.client[settings.mongodb_db][settings.application_attempts_collection]

    async def ensure_indexes(self) -> bool:
        """Ensure unique index on job_id once per process/lifecycle, loop-neutrally."""
        with self._lock:
            if self._index_succeeded:
                return True
            if self._index_error is not None:
                raise AttemptError(self._index_error)

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
                sanitized_err = f"Attempt index initialization failed ({type(e).__name__})"
                with self._lock:
                    self.__class__._index_succeeded = False
                    self.__class__._index_error = sanitized_err
                fut.set_result(False)
                raise AttemptError(sanitized_err)
        else:
            loop = asyncio.get_running_loop()
            await asyncio.wrap_future(fut, loop=loop)
            with self._lock:
                if self._index_succeeded:
                    return True
                raise AttemptError(self._index_error or "Attempt index initialization failed")

    async def get_attempt(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get the current attempt record for a job ID."""
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id:
            return None
        res = self.collection.find_one({"job_id": canonical_id})
        if inspect.isawaitable(res):
            return await res
        return res

    async def preflight_check(self, job_id: str) -> AttemptPreflightResult:
        """Read-only preflight check at execution entry without claiming an attempt."""
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id:
            return AttemptPreflightResult(can_proceed=False, reason="invalid_job_id")

        existing = await self.get_attempt(canonical_id)
        if existing is None:
            return AttemptPreflightResult(can_proceed=True)

        status_str = str(existing.get("status", AttemptStatus.CREATED.value))
        try:
            status = AttemptStatus(status_str)
        except ValueError:
            status = AttemptStatus.SUBMISSION_UNKNOWN

        attempt_id = str(existing.get("attempt_id", ""))

        if status == AttemptStatus.SUBMITTED:
            return AttemptPreflightResult(
                can_proceed=False,
                status=AttemptStatus.SUBMITTED,
                reason="already_submitted",
                existing_attempt_id=attempt_id,
            )
        if status == AttemptStatus.SUBMISSION_UNKNOWN:
            return AttemptPreflightResult(
                can_proceed=False,
                status=AttemptStatus.SUBMISSION_UNKNOWN,
                reason="submission_unknown_prior_attempt",
                existing_attempt_id=attempt_id,
            )
        if status in (AttemptStatus.CREATED, AttemptStatus.QUOTA_RESERVED):
            return AttemptPreflightResult(
                can_proceed=False,
                status=status,
                reason="in_progress_attempt_exists",
                existing_attempt_id=attempt_id,
            )
        if status == AttemptStatus.RELEASED:
            return AttemptPreflightResult(
                can_proceed=True,
                status=AttemptStatus.RELEASED,
                existing_attempt_id=attempt_id,
            )

        return AttemptPreflightResult(can_proceed=True)

    async def begin_attempt(
        self,
        job_id: str,
        run_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AttemptClaimResult:
        """Claim ownership of an attempt only at final live-submit boundary.

        Every claim allocates a fresh attempt_id token. Existing in-progress or terminal
        attempts always block new callers, even from the same run ID.

        Args:
            job_id: Canonical job identifier.
            run_id: Current session identifier.
            metadata: Optional bounded initial metadata.

        Returns:
            AttemptClaimResult with claim decision, status, and attempt_id token.
        """
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id:
            raise AttemptClaimError("Invalid empty job_id")

        safe_run_id = bound_string(run_id, max_length=64)
        safe_meta = bound_and_redact_metadata(metadata or {})
        now = datetime.now(timezone.utc)
        new_attempt_id = f"att_{uuid4().hex[:16]}"

        existing = await self.get_attempt(canonical_id)
        if existing is not None:
            existing_status_str = str(existing.get("status", AttemptStatus.CREATED.value))
            try:
                existing_status = AttemptStatus(existing_status_str)
            except ValueError:
                existing_status = AttemptStatus.SUBMISSION_UNKNOWN

            existing_run_id = str(existing.get("run_id", ""))
            existing_attempt_id = str(existing.get("attempt_id", ""))

            if existing_status == AttemptStatus.SUBMITTED:
                return AttemptClaimResult(
                    claimed=False,
                    status=AttemptStatus.SUBMITTED,
                    attempt_id=existing_attempt_id,
                    reason="already_submitted",
                    existing_run_id=existing_run_id,
                )

            if existing_status == AttemptStatus.SUBMISSION_UNKNOWN:
                return AttemptClaimResult(
                    claimed=False,
                    status=AttemptStatus.SUBMISSION_UNKNOWN,
                    attempt_id=existing_attempt_id,
                    reason="submission_unknown_manual_review_required",
                    existing_run_id=existing_run_id,
                )

            if existing_status in (AttemptStatus.CREATED, AttemptStatus.QUOTA_RESERVED):
                return AttemptClaimResult(
                    claimed=False,
                    status=existing_status,
                    attempt_id=existing_attempt_id,
                    reason="concurrent_worker_active",
                    existing_run_id=existing_run_id,
                )

            if existing_status == AttemptStatus.RELEASED:
                update_query = {
                    "job_id": canonical_id,
                    "status": AttemptStatus.RELEASED.value,
                }
                update_doc = {
                    "$set": {
                        "status": AttemptStatus.CREATED.value,
                        "run_id": safe_run_id,
                        "attempt_id": new_attempt_id,
                        "updated_at": now,
                        "released_at": None,
                        "release_reason": None,
                        **({"metadata": safe_meta} if safe_meta else {}),
                    }
                }
                res = self.collection.update_one(update_query, update_doc)
                if inspect.isawaitable(res):
                    res = await res
                matched = (
                    getattr(res, "matched_count", 0) or getattr(res, "modified_count", 0)
                    if res
                    else 0
                )
                if matched > 0:
                    return AttemptClaimResult(
                        claimed=True,
                        status=AttemptStatus.CREATED,
                        attempt_id=new_attempt_id,
                        existing_run_id=safe_run_id,
                    )
                # Bounded re-read on race
                re_read = await self.get_attempt(canonical_id)
                if not re_read:
                    raise AttemptClaimError(
                        "Failed to claim attempt during concurrent race (document absent)"
                    )
                try:
                    re_status = AttemptStatus(
                        str(re_read.get("status", AttemptStatus.SUBMISSION_UNKNOWN.value))
                    )
                except ValueError:
                    re_status = AttemptStatus.SUBMISSION_UNKNOWN
                return AttemptClaimResult(
                    claimed=False,
                    status=re_status,
                    attempt_id=str(re_read.get("attempt_id", "")),
                    reason="concurrent_worker_active",
                    existing_run_id=str(re_read.get("run_id", "")),
                )

        new_doc = {
            "job_id": canonical_id,
            "attempt_id": new_attempt_id,
            "run_id": safe_run_id,
            "status": AttemptStatus.CREATED.value,
            "created_at": now,
            "updated_at": now,
            "metadata": safe_meta,
        }

        try:
            res = self.collection.insert_one(new_doc)
            if inspect.isawaitable(res):
                await res
            return AttemptClaimResult(
                claimed=True,
                status=AttemptStatus.CREATED,
                attempt_id=new_attempt_id,
                existing_run_id=safe_run_id,
            )
        except Exception:
            # Duplicate key error on concurrent insert: single re-read
            re_read = await self.get_attempt(canonical_id)
            if not re_read:
                raise AttemptClaimError(
                    "Failed to claim attempt during concurrent race (document absent)"
                )
            try:
                re_status = AttemptStatus(
                    str(re_read.get("status", AttemptStatus.SUBMISSION_UNKNOWN.value))
                )
            except ValueError:
                re_status = AttemptStatus.SUBMISSION_UNKNOWN
            return AttemptClaimResult(
                claimed=False,
                status=re_status,
                attempt_id=str(re_read.get("attempt_id", "")),
                reason="concurrent_worker_active",
                existing_run_id=str(re_read.get("run_id", "")),
            )

    async def reserve_quota_state(
        self,
        job_id: str,
        attempt_id: str,
    ) -> bool:
        """Transition attempt state to QUOTA_RESERVED for the exact attempt_id token."""
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id or not attempt_id:
            return False

        now = datetime.now(timezone.utc)
        res = self.collection.update_one(
            {
                "job_id": canonical_id,
                "attempt_id": attempt_id,
                "status": AttemptStatus.CREATED.value,
            },
            {
                "$set": {
                    "status": AttemptStatus.QUOTA_RESERVED.value,
                    "reserved_at": now,
                    "updated_at": now,
                }
            },
        )
        if inspect.isawaitable(res):
            res = await res
        matched = (
            getattr(res, "matched_count", 0) or getattr(res, "modified_count", 0) if res else 0
        )
        return matched > 0

    async def mark_unknown(
        self,
        job_id: str,
        attempt_id: str,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Durable transition to SUBMISSION_UNKNOWN immediately before clicking submit.

        Requires attempt to be in QUOTA_RESERVED state for the exact attempt_id token.
        """
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id or not attempt_id:
            return False

        now = datetime.now(timezone.utc)
        safe_err = bound_string(error, max_length=512) if error else None
        safe_meta = bound_and_redact_metadata(metadata or {})

        update_set: dict[str, Any] = {
            "status": AttemptStatus.SUBMISSION_UNKNOWN.value,
            "unknown_at": now,
            "updated_at": now,
        }
        if safe_err:
            update_set["error"] = safe_err
        if safe_meta:
            for k, v in safe_meta.items():
                update_set[f"metadata.{k}"] = v

        res = self.collection.update_one(
            {
                "job_id": canonical_id,
                "attempt_id": attempt_id,
                "status": AttemptStatus.QUOTA_RESERVED.value,
            },
            {"$set": update_set},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = (
            getattr(res, "matched_count", 0) or getattr(res, "modified_count", 0) if res else 0
        )
        return matched > 0

    async def mark_submitted(
        self,
        job_id: str,
        attempt_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AttemptTransitionResult:
        """Transition from SUBMISSION_UNKNOWN to SUBMITTED upon explicit LinkedIn confirmation."""
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id or not attempt_id:
            return AttemptTransitionResult(
                success=False,
                changed=False,
                status=AttemptStatus.SUBMISSION_UNKNOWN,
                attempt_id=attempt_id,
                reason="invalid_identifiers",
            )

        existing = await self.get_attempt(canonical_id)
        if existing and existing.get("status") == AttemptStatus.SUBMITTED.value:
            if existing.get("attempt_id") == attempt_id:
                return AttemptTransitionResult(
                    success=True,
                    changed=False,
                    status=AttemptStatus.SUBMITTED,
                    attempt_id=attempt_id,
                )
            return AttemptTransitionResult(
                success=False,
                changed=False,
                status=AttemptStatus.SUBMITTED,
                attempt_id=attempt_id,
                reason="attempt_token_mismatch",
            )

        now = datetime.now(timezone.utc)
        safe_meta = bound_and_redact_metadata(metadata or {})

        update_set: dict[str, Any] = {
            "status": AttemptStatus.SUBMITTED.value,
            "submitted_at": now,
            "updated_at": now,
        }
        if safe_meta:
            for k, v in safe_meta.items():
                update_set[f"metadata.{k}"] = v

        res = self.collection.update_one(
            {
                "job_id": canonical_id,
                "attempt_id": attempt_id,
                "status": AttemptStatus.SUBMISSION_UNKNOWN.value,
            },
            {"$set": update_set},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = (
            getattr(res, "matched_count", 0) or getattr(res, "modified_count", 0) if res else 0
        )

        if matched > 0:
            return AttemptTransitionResult(
                success=True,
                changed=True,
                status=AttemptStatus.SUBMITTED,
                attempt_id=attempt_id,
            )

        # Check if already submitted in race
        re_check = await self.get_attempt(canonical_id)
        if re_check and re_check.get("status") == AttemptStatus.SUBMITTED.value:
            if re_check.get("attempt_id") == attempt_id:
                return AttemptTransitionResult(
                    success=True,
                    changed=False,
                    status=AttemptStatus.SUBMITTED,
                    attempt_id=attempt_id,
                )
            return AttemptTransitionResult(
                success=False,
                changed=False,
                status=AttemptStatus.SUBMITTED,
                attempt_id=attempt_id,
                reason="attempt_token_mismatch",
            )

        return AttemptTransitionResult(
            success=False,
            changed=False,
            status=AttemptStatus.SUBMISSION_UNKNOWN,
            attempt_id=attempt_id,
            reason="transition_update_unmatched",
        )

    async def mark_released(
        self,
        job_id: str,
        attempt_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Transition a definitely pre-click attempt to RELEASED for the exact attempt_id token."""
        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        if not canonical_id or not attempt_id:
            return False

        existing = await self.get_attempt(canonical_id)
        if existing and existing.get("status") == AttemptStatus.RELEASED.value:
            return existing.get("attempt_id") == attempt_id

        now = datetime.now(timezone.utc)
        safe_reason = bound_string(reason, max_length=512) if reason else None

        update_set: dict[str, Any] = {
            "status": AttemptStatus.RELEASED.value,
            "released_at": now,
            "updated_at": now,
        }
        if safe_reason:
            update_set["release_reason"] = safe_reason

        res = self.collection.update_one(
            {
                "job_id": canonical_id,
                "attempt_id": attempt_id,
                "status": {
                    "$in": [
                        AttemptStatus.CREATED.value,
                        AttemptStatus.QUOTA_RESERVED.value,
                    ]
                },
            },
            {"$set": update_set},
        )
        if inspect.isawaitable(res):
            res = await res
        matched = (
            getattr(res, "matched_count", 0) or getattr(res, "modified_count", 0) if res else 0
        )
        return matched > 0

    @classmethod
    def _reset_state(cls) -> None:
        """Synchronously reset index lifecycle state."""
        with cls._lock:
            cls._init_future = None
            cls._index_succeeded = False
            cls._index_error = None

    @classmethod
    async def close(cls) -> None:
        """Close shared MongoDB client and reset all repository lifecycles."""
        await MongoClientManager.close()


MongoClientManager.register_reset_hook(AttemptRepository._reset_state)


class QuotaRepository:
    """Repository for atomic conditional daily and per-session quota reservations."""

    _lock: ClassVar[threading.Lock] = threading.Lock()
    _init_future: ClassVar[Optional[concurrent.futures.Future]] = None
    _index_succeeded: ClassVar[bool] = False
    _index_error: ClassVar[Optional[str]] = None

    def __init__(self):
        """Initialize quota repository with MongoDB connection."""
        settings = get_settings()
        self.client = MongoClientManager.get_client()
        self.collection = self.client[settings.mongodb_db][settings.application_quotas_collection]

    async def ensure_indexes(self) -> bool:
        """Ensure unique index on date field once per process/lifecycle."""
        with self._lock:
            if self._index_succeeded:
                return True
            if self._index_error is not None:
                raise AttemptError(self._index_error)

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
                    res = self.collection.create_index("date", unique=True)
                    if inspect.isawaitable(res):
                        await res
                with self._lock:
                    self.__class__._index_succeeded = True
                    self.__class__._index_error = None
                fut.set_result(True)
                return True
            except Exception as e:
                sanitized_err = f"Quota index initialization failed ({type(e).__name__})"
                with self._lock:
                    self.__class__._index_succeeded = False
                    self.__class__._index_error = sanitized_err
                fut.set_result(False)
                raise AttemptError(sanitized_err)
        else:
            loop = asyncio.get_running_loop()
            await asyncio.wrap_future(fut, loop=loop)
            with self._lock:
                if self._index_succeeded:
                    return True
                raise AttemptError(self._index_error or "Quota index initialization failed")

    async def _ensure_day_doc(self, date_key: str) -> None:
        """Ensure base UTC-day document exists without cap predicates to avoid first-day insert races."""
        now = datetime.now(timezone.utc)
        try:
            res = self.collection.update_one(
                {"date": date_key},
                {
                    "$setOnInsert": {
                        "date": date_key,
                        "daily_reserved": 0,
                        "daily_submitted": 0,
                        "sessions": {},
                        "reservations": {},
                        "created_at": now,
                        "updated_at": now,
                    }
                },
                upsert=True,
            )
            if inspect.isawaitable(res):
                await res
        except Exception as exc:
            # Re-read to determine if document now exists due to harmless concurrent race
            try:
                doc = await self._get_day_doc(date_key)
            except Exception as re_err:
                raise AttemptError(
                    f"Quota day initialization failed ({type(re_err).__name__})"
                ) from re_err
            if not doc:
                raise AttemptError(
                    f"Quota day initialization failed ({type(exc).__name__})"
                ) from exc

    async def reserve_quota(
        self,
        attempt_id: str,
        job_id: str,
        run_id: str,
        daily_cap: int,
        session_cap: int,
        date_str: Optional[str] = None,
    ) -> QuotaReservationResult:
        """Atomically reserve daily and session quota keyed by attempt_id token.

        Args:
            attempt_id: Unique attempt identifier token.
            job_id: Canonical job identifier.
            run_id: Session identifier.
            daily_cap: Maximum allowed daily applications.
            session_cap: Maximum allowed session applications.
            date_str: Optional UTC date string YYYY-MM-DD (defaults to today UTC).

        Returns:
            QuotaReservationResult with success, already_reserved, and counts.
        """
        if daily_cap <= 0 or session_cap <= 0:
            return QuotaReservationResult(
                success=False,
                reason="zero_cap",
            )

        await self.ensure_indexes()
        canonical_id = canonicalize_job_id(job_id)
        date_key = date_str or get_utc_date_str()
        safe_run_id = make_safe_field_key(run_id, prefix="r_")
        safe_attempt_key = make_safe_field_key(attempt_id, prefix="a_")
        now = datetime.now(timezone.utc)

        # 1. Ensure day document exists without cap predicates
        await self._ensure_day_doc(date_key)

        # 2. Pre-read check for existing reservation and owner matching
        doc = await self._get_day_doc(date_key)
        if doc:
            reservations = doc.get("reservations", {})
            existing_res = reservations.get(safe_attempt_key)
            if existing_res:
                if existing_res.get("run_id") != safe_run_id:
                    return QuotaReservationResult(
                        success=False,
                        reason="reservation_owner_mismatch",
                    )
                if existing_res.get("status") in ("reserved", "committed"):
                    daily_reserved = int(doc.get("daily_reserved", 0))
                    session_reserved = int(
                        doc.get("sessions", {}).get(safe_run_id, {}).get("reserved", 0)
                    )
                    return QuotaReservationResult(
                        success=True,
                        already_reserved=True,
                        daily_reserved=daily_reserved,
                        session_reserved=session_reserved,
                    )

            daily_reserved = int(doc.get("daily_reserved", 0))
            session_reserved = int(doc.get("sessions", {}).get(safe_run_id, {}).get("reserved", 0))
            if daily_reserved >= daily_cap:
                return QuotaReservationResult(
                    success=False,
                    daily_reserved=daily_reserved,
                    session_reserved=session_reserved,
                    reason="daily_cap_reached",
                )
            if session_reserved >= session_cap:
                return QuotaReservationResult(
                    success=False,
                    daily_reserved=daily_reserved,
                    session_reserved=session_reserved,
                    reason="session_cap_reached",
                )

        # 3. Atomic conditional single-document update with upsert=False
        filter_query = {
            "date": date_key,
            "$and": [
                {
                    "$or": [
                        {"daily_reserved": {"$exists": False}},
                        {"daily_reserved": {"$lt": daily_cap}},
                    ]
                },
                {
                    "$or": [
                        {f"sessions.{safe_run_id}.reserved": {"$exists": False}},
                        {f"sessions.{safe_run_id}.reserved": {"$lt": session_cap}},
                    ]
                },
            ],
            f"reservations.{safe_attempt_key}.status": {"$nin": ["reserved", "committed"]},
        }

        update_op = {
            "$inc": {
                "daily_reserved": 1,
                f"sessions.{safe_run_id}.reserved": 1,
            },
            "$set": {
                f"reservations.{safe_attempt_key}": {
                    "attempt_id": attempt_id,
                    "job_id": canonical_id,
                    "run_id": safe_run_id,
                    "status": "reserved",
                    "reserved_at": now,
                    "updated_at": now,
                },
                "updated_at": now,
            },
        }

        try:
            res = self.collection.update_one(filter_query, update_op, upsert=False)
            if inspect.isawaitable(res):
                res = await res

            matched = getattr(res, "matched_count", 0)
            modified = getattr(res, "modified_count", 0)

            if matched > 0 or modified > 0:
                doc_after = await self._get_day_doc(date_key)
                d_cnt = int(doc_after.get("daily_reserved", 1)) if doc_after else 1
                s_cnt = (
                    int(doc_after.get("sessions", {}).get(safe_run_id, {}).get("reserved", 1))
                    if doc_after
                    else 1
                )
                return QuotaReservationResult(
                    success=True,
                    already_reserved=False,
                    daily_reserved=d_cnt,
                    session_reserved=s_cnt,
                )
        except Exception as e:
            # Re-read to check if our reservation was committed despite network error
            doc_err = await self._get_day_doc(date_key)
            if doc_err:
                resv = doc_err.get("reservations", {}).get(safe_attempt_key)
                if (
                    resv
                    and resv.get("run_id") == safe_run_id
                    and resv.get("status") in ("reserved", "committed")
                ):
                    return QuotaReservationResult(
                        success=True,
                        already_reserved=True,
                        daily_reserved=int(doc_err.get("daily_reserved", 0)),
                        session_reserved=int(
                            doc_err.get("sessions", {}).get(safe_run_id, {}).get("reserved", 0)
                        ),
                    )
            raise QuotaExceededError(
                f"Quota reservation failed due to update error ({type(e).__name__})"
            ) from e

        # 4. Check why update did not match
        doc_final = await self._get_day_doc(date_key)
        if doc_final:
            resv = doc_final.get("reservations", {}).get(safe_attempt_key)
            if resv:
                if resv.get("run_id") != safe_run_id:
                    return QuotaReservationResult(
                        success=False,
                        reason="reservation_owner_mismatch",
                    )
                if resv.get("status") in ("reserved", "committed"):
                    return QuotaReservationResult(
                        success=True,
                        already_reserved=True,
                        daily_reserved=int(doc_final.get("daily_reserved", 0)),
                        session_reserved=int(
                            doc_final.get("sessions", {}).get(safe_run_id, {}).get("reserved", 0)
                        ),
                    )
            d_cnt = int(doc_final.get("daily_reserved", 0))
            s_cnt = int(doc_final.get("sessions", {}).get(safe_run_id, {}).get("reserved", 0))
            if d_cnt >= daily_cap:
                return QuotaReservationResult(
                    success=False,
                    daily_reserved=d_cnt,
                    session_reserved=s_cnt,
                    reason="daily_cap_reached",
                )
            if s_cnt >= session_cap:
                return QuotaReservationResult(
                    success=False,
                    daily_reserved=d_cnt,
                    session_reserved=s_cnt,
                    reason="session_cap_reached",
                )

        return QuotaReservationResult(
            success=False,
            reason="quota_reservation_failed",
        )

    async def release_quota(
        self,
        attempt_id: str,
        run_id: str,
        date_str: Optional[str] = None,
    ) -> QuotaReleaseResult:
        """Release a previously reserved quota slot for the exact attempt_id and run_id."""
        await self.ensure_indexes()
        date_key = date_str or get_utc_date_str()
        safe_run_id = make_safe_field_key(run_id, prefix="r_")
        safe_attempt_key = make_safe_field_key(attempt_id, prefix="a_")
        now = datetime.now(timezone.utc)

        # Check if already released (with owner check)
        doc = await self._get_day_doc(date_key)
        if doc:
            resv = doc.get("reservations", {}).get(safe_attempt_key)
            if resv:
                if resv.get("run_id") != safe_run_id:
                    return QuotaReleaseResult(
                        success=False,
                        changed=False,
                        reason="not_reserved_or_wrong_owner",
                    )
                if resv.get("status") == "released":
                    return QuotaReleaseResult(
                        success=True,
                        changed=False,
                        already_released=True,
                    )

        filter_query = {
            "date": date_key,
            f"reservations.{safe_attempt_key}.status": "reserved",
            f"reservations.{safe_attempt_key}.run_id": safe_run_id,
            "daily_reserved": {"$gt": 0},
            f"sessions.{safe_run_id}.reserved": {"$gt": 0},
        }
        update_op = {
            "$inc": {
                "daily_reserved": -1,
                f"sessions.{safe_run_id}.reserved": -1,
            },
            "$set": {
                f"reservations.{safe_attempt_key}.status": "released",
                f"reservations.{safe_attempt_key}.released_at": now,
                f"reservations.{safe_attempt_key}.updated_at": now,
                "updated_at": now,
            },
        }

        res = self.collection.update_one(filter_query, update_op)
        if inspect.isawaitable(res):
            res = await res
        matched = (
            getattr(res, "matched_count", 0) or getattr(res, "modified_count", 0) if res else 0
        )

        if matched > 0:
            return QuotaReleaseResult(success=True, changed=True)

        doc_after = await self._get_day_doc(date_key)
        if doc_after:
            resv = doc_after.get("reservations", {}).get(safe_attempt_key)
            if resv and resv.get("run_id") == safe_run_id and resv.get("status") == "released":
                return QuotaReleaseResult(
                    success=True,
                    changed=False,
                    already_released=True,
                )

        return QuotaReleaseResult(
            success=False,
            changed=False,
            reason="not_reserved_or_wrong_owner",
        )

    async def commit_quota(
        self,
        attempt_id: str,
        run_id: str,
        date_str: Optional[str] = None,
    ) -> QuotaCommitResult:
        """Mark a reserved quota slot as confirmed submitted for exact attempt_id and run_id."""
        await self.ensure_indexes()
        date_key = date_str or get_utc_date_str()
        safe_run_id = make_safe_field_key(run_id, prefix="r_")
        safe_attempt_key = make_safe_field_key(attempt_id, prefix="a_")
        now = datetime.now(timezone.utc)

        # Check if already committed (with owner check)
        doc = await self._get_day_doc(date_key)
        if doc:
            resv = doc.get("reservations", {}).get(safe_attempt_key)
            if resv:
                if resv.get("run_id") != safe_run_id:
                    return QuotaCommitResult(
                        success=False,
                        changed=False,
                        reason="not_reserved_or_wrong_owner",
                    )
                if resv.get("status") == "committed":
                    return QuotaCommitResult(
                        success=True,
                        changed=False,
                        already_committed=True,
                    )

        filter_query = {
            "date": date_key,
            f"reservations.{safe_attempt_key}.status": "reserved",
            f"reservations.{safe_attempt_key}.run_id": safe_run_id,
        }
        update_op = {
            "$inc": {
                "daily_submitted": 1,
                f"sessions.{safe_run_id}.submitted": 1,
            },
            "$set": {
                f"reservations.{safe_attempt_key}.status": "committed",
                f"reservations.{safe_attempt_key}.committed_at": now,
                f"reservations.{safe_attempt_key}.updated_at": now,
                "updated_at": now,
            },
        }

        res = self.collection.update_one(filter_query, update_op)
        if inspect.isawaitable(res):
            res = await res
        matched = (
            getattr(res, "matched_count", 0) or getattr(res, "modified_count", 0) if res else 0
        )

        if matched > 0:
            return QuotaCommitResult(success=True, changed=True)

        doc_after = await self._get_day_doc(date_key)
        if doc_after:
            resv = doc_after.get("reservations", {}).get(safe_attempt_key)
            if resv and resv.get("run_id") == safe_run_id and resv.get("status") == "committed":
                return QuotaCommitResult(
                    success=True,
                    changed=False,
                    already_committed=True,
                )

        return QuotaCommitResult(
            success=False,
            changed=False,
            reason="not_reserved_or_wrong_owner",
        )

    async def get_daily_consumed(self, date_str: Optional[str] = None) -> int:
        """Get authoritative daily consumed capacity (daily_reserved = reserved + unknown + committed)."""
        await self.ensure_indexes()
        date_key = date_str or get_utc_date_str()
        doc = await self._get_day_doc(date_key)
        if not doc:
            return 0
        return max(0, int(doc.get("daily_reserved", 0)))

    async def get_session_consumed(self, run_id: str, date_str: Optional[str] = None) -> int:
        """Get authoritative session consumed capacity."""
        await self.ensure_indexes()
        date_key = date_str or get_utc_date_str()
        safe_run_id = make_safe_field_key(run_id, prefix="r_")
        doc = await self._get_day_doc(date_key)
        if not doc:
            return 0
        return max(
            0,
            int(doc.get("sessions", {}).get(safe_run_id, {}).get("reserved", 0)),
        )

    async def _get_day_doc(self, date_key: str) -> Optional[dict[str, Any]]:
        """Fetch the day document for date_key."""
        res = self.collection.find_one({"date": date_key})
        if inspect.isawaitable(res):
            return await res
        return res

    @classmethod
    def _reset_state(cls) -> None:
        """Synchronously reset index lifecycle state."""
        with cls._lock:
            cls._init_future = None
            cls._index_succeeded = False
            cls._index_error = None

    @classmethod
    async def close(cls) -> None:
        """Close shared MongoDB client and reset all repository lifecycles."""
        await MongoClientManager.close()


MongoClientManager.register_reset_hook(QuotaRepository._reset_state)
