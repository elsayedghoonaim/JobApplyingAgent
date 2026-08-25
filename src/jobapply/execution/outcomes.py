"""Execution outcome construction, receipts, and state transition builders."""

from typing import Optional

from jobapply.models.application import ApplicationStatus
from jobapply.models.telegram import OutboxDeliveryResult
from jobapply.nodes.outcomes import (
    append_application_outcome,
    build_application_outcome,
    state_list,
)
from jobapply.state import JobApplyState
from jobapply.utils.account_safety import (
    AccountSafetyDetection,
    normalize_safety_log_payload,
    sanitize_evidence_string,
)
from jobapply.utils.dedup import canonicalize_job_id

# Bounded redaction helper reused for any hostile free-text that must survive
# into pending state even when queue-item construction itself fails.
from jobapply.utils.manual_review import (
    MAX_PENDING_MANUAL_REVIEW_ITEMS,
    manual_review_item_from_outcome,
    safe_enqueue_manual_review,
    sanitize_untrusted_text,
    serialize_pending_manual_review_item,
)
from jobapply.utils.observability import log_event
from jobapply.utils.telegram import TelegramClient


def _extract_attempt_fields(
    outcome: dict, extra: dict | None
) -> tuple[str | None, str | None, bool]:
    """Pull bounded attempt identifiers/ambiguity from an outcome payload."""
    merged: dict = {}
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


def _fallback_pending_entry(state: JobApplyState, outcome: dict, extra: dict | None) -> dict:
    """Deterministic bounded pending entry built without the pydantic model.

    Used only when queue-item construction itself fails, so a hostile or
    malformed outcome can never silently drop retry state. Every field is
    redacted and bounded; the entry stays reconstructable/retryable via its
    deterministic idempotency key.
    """
    from datetime import datetime, timezone

    from jobapply.utils.manual_review import make_manual_review_idempotency_key

    run_id = sanitize_untrusted_text(str(state.get("run_id") or "default_run"), 64)
    job_id = canonicalize_job_id(outcome.get("job_id")) if outcome.get("job_id") else None
    reason_category = sanitize_untrusted_text(str(outcome.get("reason") or "manual_review"), 120)
    reason_detail = sanitize_untrusted_text(str(outcome.get("error") or ""), 256)
    key = make_manual_review_idempotency_key(run_id, job_id, reason_category, reason_detail)
    attempt_id, attempt_status, ambiguous = _extract_attempt_fields(outcome, extra)
    return {
        "item_id": key,
        "idempotency_key": key,
        "run_id": run_id or "default_run",
        "job_id": job_id,
        "title": sanitize_untrusted_text(str(outcome.get("title") or ""), 256) or None,
        "company": sanitize_untrusted_text(str(outcome.get("company") or ""), 256) or None,
        "url": sanitize_untrusted_text(str(outcome.get("url") or ""), 512) or None,
        "reason_category": reason_category,
        "reason_detail": sanitize_untrusted_text(str(outcome.get("error") or ""), 1024) or None,
        "attempt_id": sanitize_untrusted_text(attempt_id or "", 64) or None,
        "attempt_status": sanitize_untrusted_text(attempt_status or "", 64) or None,
        "ambiguous_submission": bool(ambiguous),
        "state": "open",
        # Diagnostic marker only; ignored by the pydantic model on re-parse.
        "construction_failed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def _queue_manual_review_for_outcome(
    state: JobApplyState,
    outcome: dict,
    extra: dict | None = None,
) -> tuple[bool, Optional[dict]]:
    """Durably enqueue one needs_manual_review outcome; never raises.

    Returns ``(queued, pending_entry)`` where ``pending_entry`` is the bounded
    redacted serialized item to retain in checkpoint state when queueing did
    not durably land. Construction failures fall back to a deterministic
    bounded entry so retry state is never silently lost; the outcome itself is
    never converted into a success.
    """
    try:
        item = manual_review_item_from_outcome(
            str(state.get("run_id") or "default_run"), outcome, extra
        )
        queued = await safe_enqueue_manual_review(item)
        if queued:
            return True, None
        return False, serialize_pending_manual_review_item(item)
    except Exception as exc:
        # Construction itself failed (e.g. hostile field shapes). Preserve a
        # deterministic bounded pending entry rather than losing the item.
        log_event(
            "warning",
            "manual_review.construction_failed",
            "Manual-review item construction failed; retaining bounded fallback entry.",
            run_id=str(state.get("run_id") or "") or None,
            node="manual_review_queue",
            exc=exc,
        )
        return False, _fallback_pending_entry(state, outcome, extra)


def _pending_state_update(state: JobApplyState, pending_entry: Optional[dict]) -> dict:
    """Build bounded pending-list state when a durable queue write failed.

    When the bounded list is already full, NO existing entry is ever evicted:
    an evicted legacy/orphan record might have no authoritative application
    outcome from which it could be reconstructed. Instead the overflow counter
    is incremented and recovery of the NEWEST failure relies on the
    ``needs_manual_review`` outcome appended atomically in this same graph
    update, from which central flush deterministically rebuilds the queue
    item. Returned updates therefore always pair the new application outcome
    with its overflow accounting in a single atomic state transition.
    """
    if not pending_entry:
        return {}

    pending = list(state.get("manual_review_queue_pending") or [])
    overflow = int(state.get("manual_review_queue_overflow") or 0)

    if len(pending) >= MAX_PENDING_MANUAL_REVIEW_ITEMS:
        overflow += 1
        log_event(
            "warning",
            "manual_review.pending_overflow",
            "Pending manual-review state full; existing entries retained and "
            "newest failure deferred to authoritative-outcome reconstruction.",
            node="manual_review_queue",
            details={"deferred_reason": str(pending_entry.get("reason_category") or "")[:64]},
        )
        return {
            "manual_review_queue_overflow": overflow,
            "manual_review_queue_pending": pending,
        }

    pending.append(pending_entry)
    return {
        "manual_review_queue_pending": pending[:MAX_PENDING_MANUAL_REVIEW_ITEMS],
        "manual_review_queue_overflow": overflow,
    }


def skipped_update(
    state: JobApplyState,
    reason: str,
    error: str,
    form_qa_exchanges: list[dict],
    extra: dict | None = None,
) -> dict:
    """Build a skipped outcome update."""
    outcome = build_application_outcome(
        state,
        "skipped",
        reason=reason,
        error=error,
        qa_count=len(form_qa_exchanges),
        extra=extra,
    )
    return {
        "application_status": "skipped",
        "application_error": error,
        "form_qa_exchanges": form_qa_exchanges,
        "skipped_jobs": state_list(state, "skipped_jobs") + [outcome],
        "application_outcomes": append_application_outcome(state, outcome),
    }


async def manual_review_update(
    state: JobApplyState,
    reason: str,
    error: str,
    form_qa_exchanges: list[dict],
    extra: dict | None = None,
) -> dict:
    """Build a manual-review outcome update and durably enqueue it idempotently."""
    outcome = build_application_outcome(
        state,
        "needs_manual_review",
        reason=reason,
        error=error,
        qa_count=len(form_qa_exchanges),
        extra=extra,
    )
    queued, pending_entry = await _queue_manual_review_for_outcome(state, outcome, extra)
    updates = {
        "application_status": "needs_manual_review",
        "application_error": error,
        "form_qa_exchanges": form_qa_exchanges,
        "skipped_jobs": state_list(state, "skipped_jobs") + [outcome],
        "application_outcomes": append_application_outcome(state, outcome),
    }
    if not queued:
        updates.update(_pending_state_update(state, pending_entry))
    return updates


async def account_safety_execution_update(
    state: JobApplyState,
    detection: AccountSafetyDetection,
    current_job: dict,
    form_qa_exchanges: list[dict],
    *,
    pre_submit: bool = True,
) -> dict:
    """Build an account safety pause update for execution node without incrementing counters."""
    norm = normalize_safety_log_payload(
        run_id=str(state.get("run_id") or "unknown"),
        barrier_type=detection.barrier_type.value if detection.barrier_type else None,
        stage=detection.stage,
        reason=detection.reason,
        url=detection.url,
        resume_instructions=detection.resume_instructions,
    )
    btype_val = detection.barrier_type.value if detection.barrier_type else "unknown"
    safe_evidence = (
        sanitize_evidence_string(detection.evidence, max_length=120) if detection.evidence else None
    )
    if pre_submit:
        reason = f"account_safety_paused:{btype_val}"
        error_msg = norm["reason"]
        extra = {
            "account_safety_barrier": {
                "barrier_type": btype_val,
                "reason": norm["reason"],
                "stage": norm["stage"],
                "url": norm["url"],
            },
            "ambiguous_submission": False,
        }
    else:
        reason = f"ambiguous_post_submit_barrier:{btype_val}"
        error_msg = f"Submission status ambiguous: safety barrier or inspection unavailable ({norm['reason']})"
        extra = {
            "account_safety_barrier": {
                "barrier_type": btype_val,
                "reason": norm["reason"],
                "stage": norm["stage"],
                "url": norm["url"],
            },
            "ambiguous_submission": True,
        }

    outcome = build_application_outcome(
        state,
        ApplicationStatus.NEEDS_MANUAL_REVIEW.value,
        reason=reason,
        error=error_msg,
        qa_count=len(form_qa_exchanges),
        extra=extra,
    )
    queued, pending_entry = await _queue_manual_review_for_outcome(state, outcome, extra)
    updates = {
        "account_safety_paused": True,
        "account_safety_barrier_type": btype_val,
        "account_safety_reason": norm["reason"],
        "account_safety_stage": norm["stage"],
        "account_safety_url": norm["url"],
        "account_safety_detected_at": detection.detected_at,
        "account_safety_evidence": safe_evidence,
        "account_safety_resume_instructions": norm["resume_instructions"],
        "application_status": ApplicationStatus.NEEDS_MANUAL_REVIEW.value,
        "application_error": error_msg,
        "form_qa_exchanges": form_qa_exchanges,
        "application_outcomes": append_application_outcome(state, outcome),
        "logs": state["logs"] + [f"🛑 Account-safety pause ({btype_val}): {error_msg}"],
    }
    if not queued:
        updates.update(_pending_state_update(state, pending_entry))
    return updates


def failed_update(state: JobApplyState, error: str, form_qa_exchanges: list[dict]) -> dict:
    """Build a failed outcome update."""
    outcome = build_application_outcome(
        state,
        "failed",
        error=error,
        qa_count=len(form_qa_exchanges),
    )
    return {
        "application_status": "failed",
        "application_error": error,
        "form_qa_exchanges": form_qa_exchanges,
        "application_outcomes": append_application_outcome(state, outcome),
        "errors": list(state.get("errors") or []) + [error],
    }


def applied_update(
    state: JobApplyState,
    status: str,
    form_qa_exchanges: list[dict],
    increment_counters: bool = True,
) -> dict:
    """Build a submitted or dry-run-ready outcome update."""
    outcome = build_application_outcome(
        state,
        status,
        qa_count=len(form_qa_exchanges),
        extra={"dry_run": status == "dry_run"},
    )
    applied_jobs = state_list(state, "applied_jobs") + [outcome]
    update = {
        "application_status": status,
        "application_error": None,
        "form_qa_exchanges": form_qa_exchanges,
        "applied_jobs": applied_jobs,
        "application_outcomes": append_application_outcome(state, outcome),
    }

    if status == "submitted" and increment_counters:
        update["applications_count"] = state["applications_count"] + 1
        update["daily_applications_count"] = state["daily_applications_count"] + 1

    return update


def format_application_receipt(outcome: dict) -> str:
    """Build a plain-text Telegram receipt with unambiguous status wording."""
    status = outcome.get("status")
    already_applied = outcome.get("reason") == "already_applied"
    if status == "submitted":
        header = "✅ APPLICATION SUBMITTED"
        status_detail = "LinkedIn confirmed the application was submitted."
    elif already_applied:
        header = "☑️ ALREADY APPLIED"
        status_detail = "LinkedIn shows that this application was submitted previously."
    else:
        header = "🧪 DRY RUN COMPLETE — NOT SUBMITTED"
        status_detail = "Reached the final Review/Submit step. Submit was not clicked."
    score = float(outcome.get("score") or 0.0)
    lines = [
        header,
        "",
        f"Job: {outcome.get('title', 'Unknown title')}",
        f"Company: {outcome.get('company', 'Unknown company')}",
        f"Status: {status_detail}",
        f"Fit score: {score:.0%}",
        f"Questions answered: {outcome.get('qa_count', 0)}",
        f"Resume: {'Edited' if outcome.get('resume_edited') else 'Base resume'}",
    ]
    if outcome.get("url"):
        lines.append(f"Job link: {outcome['url']}")
    if outcome.get("timestamp"):
        lines.append(f"Recorded at (UTC): {outcome['timestamp']}")
    return "\n".join(lines)


async def send_application_receipt(
    telegram: TelegramClient,
    outcome: dict,
    run_id: str | None = None,
) -> str | None:
    """Send a receipt via durable outbox without changing a successful application outcome on failure."""
    try:
        safe_run_id = str(run_id or outcome.get("run_id") or "default_run")
        job_id = canonicalize_job_id(outcome.get("job_id", "unknown")) or "unknown"
        status = str(outcome.get("status") or "unknown")
        idempotency_key = f"receipt:{safe_run_id}:{job_id}:{status}"
        text = format_application_receipt(outcome)

        res = await telegram.enqueue_and_deliver(
            idempotency_key=idempotency_key,
            notification_type="application_receipt",
            run_id=safe_run_id,
            text=text,
            job_id=job_id,
        )
        if isinstance(res, OutboxDeliveryResult) and not res.success:
            return (
                f"Telegram application receipt outbox delivery pending/failed ({res.status.value})"
            )
        return None
    except Exception as exc:
        return f"Telegram application receipt delivery exception ({type(exc).__name__})"
