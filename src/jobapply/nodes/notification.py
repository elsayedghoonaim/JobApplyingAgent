"""Send a clear plain-text Telegram summary at the end of a session."""

from langsmith.run_helpers import trace

from jobapply.models.telegram import OutboxStatus
from jobapply.state import JobApplyState
from jobapply.utils.observability import log_event
from jobapply.utils.telegram import TelegramClient


def _job_details(index: int, outcome: dict) -> list[str]:
    """Format one job outcome as an easy-to-scan block."""
    score = float(outcome.get("score") or 0.0)
    lines = [
        f"{index}. {outcome.get('title', 'Unknown title')}",
        f"   Company: {outcome.get('company', 'Unknown company')}",
        f"   Fit score: {score:.0%}",
        f"   Questions answered: {outcome.get('qa_count', 0)}",
    ]
    if outcome.get("url"):
        lines.append(f"   Link: {outcome['url']}")
    return lines


def format_session_summary(state: JobApplyState) -> str:
    """Build a structured summary that never confuses dry runs with submissions."""
    outcomes = list(state.get("application_outcomes") or [])
    submitted = [item for item in outcomes if item.get("status") == "submitted"]
    dry_runs = [item for item in outcomes if item.get("status") == "dry_run"]
    skipped = [item for item in outcomes if item.get("status") == "skipped"]
    manual = [item for item in outcomes if item.get("status") == "needs_manual_review"]
    failed = [item for item in outcomes if item.get("status") == "failed"]

    from jobapply.utils.account_safety import (
        normalize_safety_log_payload,
        sanitize_evidence_string,
    )

    if state.get("account_safety_paused"):
        norm = normalize_safety_log_payload(
            run_id=str(state.get("run_id") or "unknown"),
            barrier_type=str(state.get("account_safety_barrier_type") or "SECURITY BARRIER"),
            stage=str(state.get("account_safety_stage") or "unknown"),
            reason=str(state.get("account_safety_reason") or "Account safety challenge detected"),
            url=str(state.get("account_safety_url") or "https://www.linkedin.com"),
            resume_instructions=str(state.get("account_safety_resume_instructions") or ""),
        )
        run_id_str = norm["run_id"]
    else:
        norm = None
        run_id_str = sanitize_evidence_string(str(state.get("run_id") or "unknown"), max_length=64)

    lines = [
        "📊 JOB APPLY SESSION SUMMARY",
        f"Run ID: {run_id_str}",
    ]

    if norm:
        lines.extend(
            [
                "",
                "🛑 ACCOUNT SAFETY PAUSED — HUMAN ACTION REQUIRED",
                f"Barrier Class: {norm['barrier_type']}",
                f"Stage: {norm['stage']}",
                f"URL: {norm['url']}",
                f"Reason: {norm['reason']}",
                f"Instructions: {norm['resume_instructions']}",
                "⚠️ SECURITY NOTICE: Never share OTPs, 2FA codes, passwords, or CAPTCHA answers.",
            ]
        )

    lines.extend(
        [
            "",
            "TOTALS",
            f"✅ Confirmed submitted: {len(submitted)}",
            f"🧪 Dry run only (not submitted): {len(dry_runs)}",
            f"⏭ Skipped: {len(skipped)}",
            f"⚠️ Needs manual review: {len(manual)}",
            f"❌ Failed: {len(failed)}",
        ]
    )

    if submitted:
        lines.extend(["", "CONFIRMED SUBMISSIONS"])
        for index, outcome in enumerate(submitted[:10], 1):
            lines.extend(_job_details(index, outcome))

    if dry_runs:
        lines.extend(["", "DRY RUNS — SUBMIT WAS NOT CLICKED"])
        for index, outcome in enumerate(dry_runs[:10], 1):
            lines.extend(_job_details(index, outcome))

    if skipped:
        lines.extend(["", "SKIPPED"])
        for index, outcome in enumerate(skipped[:10], 1):
            reason = outcome.get("reason") or outcome.get("error") or "Unspecified"
            lines.append(f"{index}. {outcome.get('title', 'Unknown title')} — {reason}")

    if manual:
        lines.extend(["", "NEEDS MANUAL REVIEW"])
        for index, outcome in enumerate(manual[:10], 1):
            reason = outcome.get("reason") or outcome.get("error") or "Incomplete form"
            extra = outcome.get("extra") or {}
            if (
                extra.get("ambiguous_submission")
                or extra.get("attempt_status") == "submission_unknown"
            ):
                reason = f"[AMBIGUOUS SUBMISSION] {reason}"
            lines.append(f"{index}. {outcome.get('title', 'Unknown title')} — {reason}")

    if failed:
        lines.extend(["", "FAILED"])
        for index, outcome in enumerate(failed[:10], 1):
            error = str(outcome.get("error") or "Unknown error")
            lines.append(f"{index}. {outcome.get('title', 'Unknown title')} — {error[:120]}")

    queries = list(state.get("search_queries") or [])
    if queries:
        completed = min(int(state.get("current_query_index", 0)) + 1, len(queries))
        lines.extend(["", f"Search queries reached: {completed}/{len(queries)}"])

    if dry_runs:
        lines.extend(
            [
                "",
                "Important: Dry-run entries are preparation checks, not submitted applications.",
            ]
        )

    message = "\n".join(lines)
    return message if len(message) <= 4000 else message[:3997] + "..."


async def notification_node(state: JobApplyState) -> dict:
    """Send the final structured session summary via durable outbox."""
    message = format_session_summary(state)
    outcomes = list(state.get("application_outcomes") or [])
    run_id = str(state.get("run_id") or "default_run")
    idempotency_key = f"summary:{run_id}"

    telegram = TelegramClient()

    try:
        async with trace(
            "telegram_send_summary",
            run_type="tool",
            metadata={
                "submitted_count": sum(item.get("status") == "submitted" for item in outcomes),
                "dry_run_count": sum(item.get("status") == "dry_run" for item in outcomes),
                "run_id": run_id,
                "account_safety_paused": bool(state.get("account_safety_paused")),
            },
        ):
            result = await telegram.enqueue_and_deliver(
                idempotency_key=idempotency_key,
                notification_type="session_summary",
                run_id=run_id,
                text=message,
                metadata={"account_safety_paused": bool(state.get("account_safety_paused"))},
            )

            (
                pending_count,
                unknown_count,
            ) = await telegram.outbox_repo.get_pending_and_unknown_counts()

            if result.success and result.status == OutboxStatus.SENT:
                return {
                    "notification_sent": True,
                    "outbox_pending_count": pending_count,
                    "outbox_unknown_count": unknown_count,
                }
            else:
                err_str = f"Summary notification queued/retryable ({result.status.value}): {result.reason or 'delivery not confirmed'}"
                return {
                    "notification_sent": False,
                    "outbox_pending_count": pending_count,
                    "outbox_unknown_count": unknown_count,
                    "errors": list(state.get("errors") or []) + [err_str],
                }

    except Exception as exc:
        from jobapply.utils.account_safety import sanitize_evidence_string

        err_str = sanitize_evidence_string(
            f"Telegram summary notification persistence failure: {type(exc).__name__}"
        )
        log_event(
            "warning",
            "notification.summary_persistence_failed",
            f"⚠️ {err_str}",
            run_id=str(state.get("run_id") or "") or None,
            node="notification_node",
            exc=exc,
        )
        return {
            "notification_sent": False,
            "errors": list(state.get("errors") or []) + [err_str],
        }
