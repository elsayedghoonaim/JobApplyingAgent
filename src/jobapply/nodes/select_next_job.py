"""Select next job node - iterator/router logic."""

from jobapply.state import JobApplyState


async def select_next_job_node(state: JobApplyState) -> dict:
    """Pick the next unseen job from current listings.

    Advances current_job_index past any already-seen jobs.
    If no unseen jobs remain, the router function handles pagination/query advancement.

    Args:
        state: Current graph state.

    Returns:
        State updates dict with current_job and current_job_index, or current_job=None if exhausted.
    """
    if state.get("account_safety_paused"):
        return {"current_job": None}

    max_jobs = state.get("max_jobs_to_evaluate")
    if max_jobs is not None and state.get("jobs_evaluated_count", 0) >= max_jobs:
        return {"current_job": None}

    seen = state["seen_job_ids"]
    listings = state["job_listings"]

    # Scan for next unseen job
    for i in range(state["current_job_index"], len(listings)):
        job = listings[i]
        job_id = job.get("job_id")

        if job_id not in seen:
            print(f"👉 Processing job: {job.get('title')} at {job.get('company')}")
            return {
                "current_job_index": i + 1,  # advance past this job
                "current_job": job,
            }
        else:
            print(
                f"⏭️  Skipping already-seen job {job_id}: {job.get('title')} at {job.get('company')}"
            )

    # No unseen jobs remain — router will decide next action
    return {"current_job": None}
