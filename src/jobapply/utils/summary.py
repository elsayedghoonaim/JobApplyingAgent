"""Deterministic, typed run-summary construction and atomic publication.

The summary is derived exclusively from authoritative graph state (per-job
application outcomes and state counters). Log messages and ProgressTracker
counters are never used as sources of truth. Publication is atomic: content is
written to a contained temporary sibling and published with ``os.replace``, so
re-running or resuming a run ID safely replaces the prior summary.
"""

import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from jobapply.utils.account_safety import (
    normalize_safety_log_payload,
    sanitize_evidence_string,
)
from jobapply.utils.paths import get_run_output_dir, sanitize_component
from jobapply.utils.redaction import redact_string

SUMMARY_SCHEMA_VERSION = 1
SUMMARY_FILENAME = "summary.json"

TERMINAL_STATUS_COMPLETED = "completed"
TERMINAL_STATUS_PAUSED = "paused"
TERMINAL_STATUS_INTERRUPTED = "interrupted"
TERMINAL_STATUS_FAILED = "failed"
TERMINAL_STATUSES = (
    TERMINAL_STATUS_COMPLETED,
    TERMINAL_STATUS_PAUSED,
    TERMINAL_STATUS_INTERRUPTED,
    TERMINAL_STATUS_FAILED,
)

STATUS_SUBMITTED = "submitted"
STATUS_DRY_RUN = "dry_run"
STATUS_SKIPPED = "skipped"
STATUS_MANUAL_REVIEW = "needs_manual_review"
STATUS_FAILED = "failed"

MAX_SUMMARY_ERRORS = 10
MAX_SUMMARY_ERROR_LENGTH = 200
MAX_SKIPPED_REASON_BUCKETS = 20
MAX_SKIPPED_REASON_LENGTH = 80
MAX_ARTIFACT_REFS = 20


def _iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_error(text: str) -> str:
    return redact_string(str(text))[:MAX_SUMMARY_ERROR_LENGTH]


def _skipped_reason_buckets(outcomes: list[dict]) -> dict[str, int]:
    """Group skipped outcomes by bounded reason, capped to a bounded bucket count."""
    buckets: dict[str, int] = {}
    for outcome in outcomes:
        reason = (
            sanitize_evidence_string(
                str(outcome.get("reason") or "unspecified"),
                max_length=MAX_SKIPPED_REASON_LENGTH,
            ).strip()
            or "unspecified"
        )
        buckets[reason] = buckets.get(reason, 0) + 1
    if len(buckets) <= MAX_SKIPPED_REASON_BUCKETS:
        return buckets
    merged: dict[str, int] = {}
    overflow = 0
    for index, (reason, count) in enumerate(buckets.items()):
        if index < MAX_SKIPPED_REASON_BUCKETS - 1:
            merged[reason] = count
        else:
            overflow += count
    if overflow:
        merged["other"] = overflow
    return merged


def _submission_unknown_count(manual_review_outcomes: list[dict]) -> int:
    """Count ambiguous manual-review outcomes without ever counting submissions."""
    return sum(
        1
        for outcome in manual_review_outcomes
        if bool(outcome.get("ambiguous_submission"))
        or outcome.get("attempt_status") == "submission_unknown"
    )


def _account_safety_pause_summary(run_id: str, state: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return bounded, sanitized pause evidence when the run is safety-paused."""
    if not state.get("account_safety_paused"):
        return None
    norm = normalize_safety_log_payload(
        run_id=run_id,
        barrier_type=str(state.get("account_safety_barrier_type") or "SECURITY BARRIER"),
        stage=str(state.get("account_safety_stage") or "unknown"),
        reason=str(state.get("account_safety_reason") or "Account safety challenge detected"),
        url=str(state.get("account_safety_url") or "https://www.linkedin.com"),
        resume_instructions=str(state.get("account_safety_resume_instructions") or ""),
    )
    detected_at_raw = str(state.get("account_safety_detected_at") or "")
    evidence = sanitize_evidence_string(
        str(state.get("account_safety_evidence") or ""), max_length=120
    )
    return {
        "barrier_type": norm["barrier_type"],
        "stage": norm["stage"],
        "reason": norm["reason"],
        "url": norm["url"],
        "detected_at": redact_string(detected_at_raw)[:64] or None,
        "evidence": evidence or None,
        "resume_instructions": norm["resume_instructions"],
    }


def _default_artifact_candidates(state: Mapping[str, Any]) -> list[Any]:
    """Default generated-artifact candidates taken from authoritative state."""
    return [state.get("resume_path"), state.get("cover_letter_path")]


def _contained_artifact_refs(artifacts: Iterable[Any], run_dir: Path) -> list[str]:
    """Keep only artifact references safely contained under the run directory."""
    refs: list[str] = []
    seen: set[str] = set()
    for raw in artifacts:
        if not raw:
            continue
        try:
            candidate = Path(str(raw)).resolve()
            candidate.relative_to(run_dir)
        except Exception:
            continue
        if not candidate.is_file():
            continue
        ref = candidate.relative_to(run_dir).as_posix()
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs[:MAX_ARTIFACT_REFS]


def build_summary(
    *,
    run_id: str,
    terminal_status: str,
    dry_run: bool,
    started_at: datetime,
    finished_at: datetime | None = None,
    state: Mapping[str, Any] | None = None,
    artifacts: Iterable[Any] | None = None,
    failure: BaseException | None = None,
    cleanup_errors: Iterable[str] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic summary dict from authoritative run state.

    Args:
        run_id: Run identifier; canonicalized to the safe run-directory name.
        terminal_status: One of completed, paused, interrupted, failed.
        dry_run: Whether the session ran in dry-run mode.
        started_at: Session start timestamp.
        finished_at: Terminal timestamp; defaults to now (UTC).
        state: Authoritative final/checkpoint graph state.
        artifacts: Candidate artifact paths; only contained files are kept.
        failure: Primary workflow exception for failed runs.
        cleanup_errors: Bounded descriptions of finalization cleanup failures.
        base_dir: Optional alternate outputs root (used by tests).

    Returns:
        Ordered summary dict with stable field ordering. The payload's
        ``run_id`` always equals the actual sanitized run-directory name.

    Raises:
        ValueError: If ``terminal_status`` is not a recognized terminal status.
    """
    if terminal_status not in TERMINAL_STATUSES:
        raise ValueError(
            f"Invalid terminal_status '{terminal_status}'; expected one of {list(TERMINAL_STATUSES)}"
        )

    finished_at = finished_at or datetime.now(timezone.utc)
    duration_seconds = round(max(0.0, (finished_at - started_at).total_seconds()), 3)

    # One canonical run ID shared by path resolution and the payload itself so
    # the recorded run_id always equals the actual safe run-directory name.
    canonical_run_id = sanitize_component(run_id, default="default_run")

    safe_state: Mapping[str, Any] = state or {}
    artifact_candidates = (
        list(artifacts) if artifacts is not None else _default_artifact_candidates(safe_state)
    )
    outcomes: list[dict] = [
        outcome
        for outcome in (safe_state.get("application_outcomes") or [])
        if isinstance(outcome, dict)
    ]
    submitted = [item for item in outcomes if item.get("status") == STATUS_SUBMITTED]
    dry_runs = [item for item in outcomes if item.get("status") == STATUS_DRY_RUN]
    skipped = [item for item in outcomes if item.get("status") == STATUS_SKIPPED]
    manual_review = [item for item in outcomes if item.get("status") == STATUS_MANUAL_REVIEW]
    failed_outcomes = [item for item in outcomes if item.get("status") == STATUS_FAILED]

    raw_errors = [str(error) for error in (safe_state.get("errors") or [])]
    all_errors = list(raw_errors)
    if failure is not None:
        all_errors.append(f"{type(failure).__name__}: {failure}")
    if cleanup_errors is not None:
        all_errors.extend(str(error) for error in cleanup_errors)

    run_dir = get_run_output_dir(run_id, base_dir)

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": canonical_run_id,
        "terminal_status": terminal_status,
        "dry_run": bool(dry_run),
        "started_at": _iso_utc(started_at),
        "finished_at": _iso_utc(finished_at),
        "duration_seconds": duration_seconds,
        "jobs_evaluated": int(safe_state.get("jobs_evaluated_count") or 0),
        "jobs_qualified": int(safe_state.get("qualified_jobs_count") or 0),
        "jobs_not_qualified": int(safe_state.get("not_qualified_jobs_count") or 0),
        "confirmed_submissions": len(submitted),
        "dry_run_ready": len(dry_runs),
        "skipped_count": len(skipped),
        "skipped_by_reason": _skipped_reason_buckets(skipped),
        "needs_manual_review_count": len(manual_review),
        "submission_unknown_count": _submission_unknown_count(manual_review),
        "failed_count": len(failed_outcomes),
        "error_count": len(all_errors),
        "errors": [_bounded_error(error) for error in all_errors[:MAX_SUMMARY_ERRORS]],
        "account_safety_pause": _account_safety_pause_summary(canonical_run_id, safe_state),
        "artifacts": _contained_artifact_refs(artifact_candidates, run_dir),
    }


def summary_path(run_id: str, base_dir: str | Path | None = None) -> Path:
    """Resolve the contained destination path outputs/<run-id>/summary.json."""
    run_dir = get_run_output_dir(run_id, base_dir)
    destination = (run_dir / SUMMARY_FILENAME).resolve()
    try:
        destination.relative_to(run_dir)
    except ValueError as err:
        raise RuntimeError("Summary path containment violation") from err
    return destination


def write_summary(
    summary: Mapping[str, Any], *, run_id: str, base_dir: str | Path | None = None
) -> Path:
    """Atomically publish the summary for a run, replacing any prior version.

    The payload is written to a unique temporary sibling inside the same run
    directory, flushed, then published with ``os.replace``. On failure the
    temporary file is removed and any existing valid summary is preserved.

    Args:
        summary: Summary payload produced by :func:`build_summary`.
        run_id: Run identifier used to resolve the contained destination.
        base_dir: Optional alternate outputs root (used by tests).

    Returns:
        Final published summary path.

    Raises:
        RuntimeError: On path containment violations or write failures after
            temporary-file cleanup.
    """
    destination = summary_path(run_id, base_dir)
    run_dir = destination.parent
    payload = json.dumps(dict(summary), ensure_ascii=False, indent=2) + "\n"
    temp_path = run_dir / f".{SUMMARY_FILENAME}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination
