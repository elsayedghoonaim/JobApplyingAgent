"""Execution outcome construction, receipts, and state transition builders."""

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
from jobapply.utils.telegram import TelegramClient


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


def manual_review_update(
    state: JobApplyState,
    reason: str,
    error: str,
    form_qa_exchanges: list[dict],
    extra: dict | None = None,
) -> dict:
    """Build a manual-review outcome update."""
    outcome = build_application_outcome(
        state,
        "needs_manual_review",
        reason=reason,
        error=error,
        qa_count=len(form_qa_exchanges),
        extra=extra,
    )
    return {
        "application_status": "needs_manual_review",
        "application_error": error,
        "form_qa_exchanges": form_qa_exchanges,
        "skipped_jobs": state_list(state, "skipped_jobs") + [outcome],
        "application_outcomes": append_application_outcome(state, outcome),
    }


def account_safety_execution_update(
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

    return {
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
