"""LangSmith tracing utilities with privacy-safe metadata helpers."""

from typing import Any, Optional


def get_safe_job_metadata(job: dict, include_description: bool = False) -> dict[str, Any]:
    """Extract safe metadata from a job dict for tracing.

    Args:
        job: Job dictionary with job_id, title, company, etc.
        include_description: Whether to include job description (default False for privacy).

    Returns:
        Safe metadata dict suitable for LangSmith traces.
    """
    metadata = {
        "job_id": job.get("job_id", "unknown"),
        "job_title": job.get("title", "unknown"),
        "company": job.get("company", "unknown"),
        "location": job.get("location", "unknown"),
        "url": job.get("url", ""),
    }

    if include_description:
        # Only include if explicitly requested (not recommended for privacy)
        metadata["description_length"] = len(job.get("description", ""))

    return metadata


def get_safe_error_metadata(error: Exception) -> dict[str, Any]:
    """Extract safe metadata from an exception for tracing.

    Args:
        error: Exception instance.

    Returns:
        Safe metadata dict with error type and message.
    """
    return {
        "error_type": type(error).__name__,
        "error_message": str(error)[:200],  # truncate long messages
    }


def get_search_metadata(query: str, page: int, result_count: int) -> dict[str, Any]:
    """Get metadata for search operations.

    Args:
        query: Search query string.
        page: Page number.
        result_count: Number of results found.

    Returns:
        Metadata dict for search traces.
    """
    return {
        "search_query": query,
        "page": page,
        "result_count": result_count,
    }


def get_browser_metadata(
    port: int, connected: bool, contexts_count: Optional[int] = None, error: Optional[str] = None
) -> dict[str, Any]:
    """Get metadata for browser/CDP operations.

    Args:
        port: CDP debug port.
        connected: Whether connection succeeded.
        contexts_count: Number of browser contexts (if connected).
        error: Error message if connection failed.

    Returns:
        Metadata dict for browser traces.
    """
    metadata = {
        "cdp_port": port,
        "connected": connected,
    }

    if contexts_count is not None:
        metadata["contexts_count"] = contexts_count

    if error:
        metadata["error"] = error[:200]

    return metadata


def get_telegram_metadata(
    nonce: Optional[str] = None,
    timeout: Optional[int] = None,
    timed_out: bool = False,
    message_sent: bool = False,
) -> dict[str, Any]:
    """Get metadata for Telegram operations.

    Args:
        nonce: Correlation nonce for Q&A (if applicable).
        timeout: Timeout in seconds (if applicable).
        timed_out: Whether the operation timed out.
        message_sent: Whether message was sent successfully.

    Returns:
        Metadata dict for Telegram traces.
    """
    metadata: dict[str, Any] = {
        "message_sent": message_sent,
    }

    if nonce:
        metadata["nonce"] = nonce

    if timeout is not None:
        metadata["timeout_seconds"] = timeout

    if timed_out:
        metadata["timed_out"] = True

    return metadata


def get_qualification_metadata(
    qualified: bool, score: float, threshold: float, key_matches_count: int, gaps_count: int
) -> dict[str, Any]:
    """Get metadata for qualification results.

    Args:
        qualified: Whether job is qualified.
        score: Qualification score (0.0-1.0).
        threshold: Qualification threshold.
        key_matches_count: Number of key matches.
        gaps_count: Number of gaps.

    Returns:
        Metadata dict for qualification traces.
    """
    return {
        "qualified": qualified,
        "score": score,
        "threshold": threshold,
        "key_matches_count": key_matches_count,
        "gaps_count": gaps_count,
    }


def get_execution_metadata(
    dry_run: bool, status: str, qa_count: int = 0, form_steps: int = 0, timed_out: bool = False
) -> dict[str, Any]:
    """Get metadata for Easy Apply execution.

    Args:
        dry_run: Whether this is a dry run.
        status: Application status (submitted, skipped, failed, dry_run).
        qa_count: Number of Q&A exchanges.
        form_steps: Number of form steps processed.
        timed_out: Whether Q&A timed out.

    Returns:
        Metadata dict for execution traces.
    """
    return {
        "dry_run": dry_run,
        "status": status,
        "qa_count": qa_count,
        "form_steps": form_steps,
        "timed_out": timed_out,
    }


def get_session_metadata(
    run_id: str, dry_run: bool, max_applications: int, queries_count: int, edge_port: int
) -> dict[str, Any]:
    """Get metadata for entire session/run.

    Args:
        run_id: Unique run identifier.
        dry_run: Whether this is a dry run.
        max_applications: Session application cap.
        queries_count: Number of search queries.
        edge_port: Edge CDP debug port.

    Returns:
        Metadata dict for session-level traces.
    """
    return {
        "run_id": run_id,
        "dry_run": dry_run,
        "max_applications": max_applications,
        "queries_count": queries_count,
        "edge_debug_port": edge_port,
    }
