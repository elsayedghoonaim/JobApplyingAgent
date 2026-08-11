"""Send a clear plain-text Telegram summary at the end of a session."""

from langsmith.run_helpers import trace

from jobapply.state import JobApplyState
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

    lines = [
        "📊 JOB APPLY SESSION SUMMARY",
        f"Run ID: {state.get('run_id', 'unknown')}",
        "",
        "TOTALS",
        f"✅ Confirmed submitted: {len(submitted)}",
        f"🧪 Dry run only (not submitted): {len(dry_runs)}",
        f"⏭ Skipped: {len(skipped)}",
        f"⚠️ Needs manual review: {len(manual)}",
        f"❌ Failed: {len(failed)}",
    ]

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
            lines.append(
                f"{index}. {outcome.get('title', 'Unknown title')} — {reason}"
            )

    if manual:
        lines.extend(["", "NEEDS MANUAL REVIEW"])
        for index, outcome in enumerate(manual[:10], 1):
            reason = outcome.get("reason") or outcome.get("error") or "Incomplete form"
            lines.append(
                f"{index}. {outcome.get('title', 'Unknown title')} — {reason}"
            )

    if failed:
        lines.extend(["", "FAILED"])
        for index, outcome in enumerate(failed[:10], 1):
            error = str(outcome.get("error") or "Unknown error")
            lines.append(
                f"{index}. {outcome.get('title', 'Unknown title')} — {error[:120]}"
            )

    queries = list(state.get("search_queries") or [])
    if queries:
        completed = min(int(state.get("current_query_index", 0)) + 1, len(queries))
        lines.extend(["", f"Search queries reached: {completed}/{len(queries)}"])

    if dry_runs:
        lines.extend([
            "",
            "Important: Dry-run entries are preparation checks, not submitted applications.",
        ])

    message = "\n".join(lines)
    return message if len(message) <= 4000 else message[:3997] + "..."


async def notification_node(state: JobApplyState) -> dict:
    """Send the final structured session summary."""
    message = format_session_summary(state)
    outcomes = list(state.get("application_outcomes") or [])
    async with trace(
        "telegram_send_summary",
        run_type="tool",
        metadata={
            "submitted_count": sum(item.get("status") == "submitted" for item in outcomes),
            "dry_run_count": sum(item.get("status") == "dry_run" for item in outcomes),
            "run_id": state.get("run_id"),
        },
    ):
        await TelegramClient().send_message(message)
    return {"notification_sent": True}
