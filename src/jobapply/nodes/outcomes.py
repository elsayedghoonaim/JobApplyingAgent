"""Shared helpers for per-application outcomes."""

import os
from datetime import datetime, timezone
from typing import Any

from jobapply.state import JobApplyState

EDITED_RESUME_PREFIX = "edited_resume_"


def state_list(state: JobApplyState, key: str) -> list[dict]:
    """Return a mutable copy of a list-valued state key."""
    return list(state.get(key) or [])


def resume_was_edited(state: JobApplyState) -> bool:
    """True only when an approved edited resume PDF exists and is selected."""
    resume_path = state.get("resume_path")
    return (
        state.get("approval_status") == "approved"
        and bool(resume_path)
        and os.path.exists(resume_path)
        and os.path.basename(resume_path).startswith(EDITED_RESUME_PREFIX)
    )


def build_application_outcome(
    state: JobApplyState,
    status: str,
    *,
    reason: str | None = None,
    error: str | None = None,
    qa_count: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Build the canonical per-job outcome stored by tail nodes."""
    current_job = state.get("current_job") or {}
    qualification_result = state.get("qualification_result") or {}

    outcome = {
        "job_id": current_job.get("job_id"),
        "title": current_job.get("title", "Unknown title"),
        "company": current_job.get("company", "Unknown company"),
        "url": current_job.get("url"),
        "score": qualification_result.get("score", 0.0),
        "status": status,
        "resume_path": state.get("resume_path"),
        "cover_letter_path": state.get("cover_letter_path"),
        "resume_edited": resume_was_edited(state),
        "qa_count": qa_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if reason:
        outcome["reason"] = reason
    if error:
        outcome["error"] = error
    if extra:
        outcome.update({key: value for key, value in extra.items() if value is not None})

    return outcome


def append_application_outcome(state: JobApplyState, outcome: dict) -> list[dict]:
    """Append an outcome without mutating the incoming state object."""
    return state_list(state, "application_outcomes") + [outcome]
