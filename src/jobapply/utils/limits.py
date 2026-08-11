"""Shared application-cap calculations."""

from jobapply.state import JobApplyState


def session_application_count(state: JobApplyState) -> int:
    """Count live submissions, or completed dry-run forms in dry-run mode."""
    count = state.get("applications_count", 0)
    if state.get("dry_run"):
        count += sum(
            outcome.get("status") == "dry_run" for outcome in state.get("application_outcomes", [])
        )
    return count


def caps_reached(state: JobApplyState) -> bool:
    """Return whether another application attempt is forbidden."""
    max_applications = state.get("max_applications")
    session_cap = (
        max_applications is not None and session_application_count(state) >= max_applications
    )
    daily_application_cap = state.get("daily_application_cap")
    daily_cap = (
        not state.get("dry_run")
        and daily_application_cap is not None
        and state.get("daily_applications_count", 0) >= daily_application_cap
    )
    return session_cap or daily_cap
