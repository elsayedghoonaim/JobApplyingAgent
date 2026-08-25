"""Validated search location/recency configuration and safe LinkedIn URL construction.

Environment settings are defaults; explicit CLI values win. Location is trimmed,
bounded, nonempty, and always URL-encoded through ``urlencode`` — never string
interpolation. Recency is a bounded positive number of days or disabled (``0`` /
empty / ``None``) with a deterministic mapping to LinkedIn's ``f_TPR``
time-range parameter.
"""

from typing import Any, Optional, Union
from urllib.parse import urlencode

MAX_LOCATION_LENGTH = 200
MIN_RECENCY_DAYS = 1
MAX_RECENCY_DAYS = 30
SECONDS_PER_DAY = 86400
DEFAULT_SEARCH_LOCATION = "Worldwide"


def normalize_search_location(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Validate a search location string.

    Returns:
        ``(location, error)``. Exactly one element is non-None. The location is
        trimmed, bounded to 200 characters, and guaranteed nonempty.

    A ``raw`` value of ``None`` means "not provided" and yields ``(None, None)``
    so callers can fall back to configured defaults.
    """
    if raw is None:
        return None, None
    if not isinstance(raw, str):
        return None, "Location must be a string."
    trimmed = raw.strip()
    if not trimmed:
        return None, "Location cannot be empty."
    if len(trimmed) > MAX_LOCATION_LENGTH:
        return None, f"Location must be at most {MAX_LOCATION_LENGTH} characters."
    return trimmed, None


def normalize_recency_days(
    raw: Optional[Union[int, str]],
) -> tuple[Optional[int], Optional[str]]:
    """Validate a recency-days value.

    Returns:
        ``(days, error)``. Exactly one element is non-None.

    ``0``, empty strings, and ``None`` all mean "disabled" and normalize to
    ``None``. Otherwise the value must be an integer between 1 and 30.
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None, None
        try:
            raw = int(stripped)
        except ValueError:
            return None, "Recency days must be a whole number of days."
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, "Recency days must be a whole number of days."
    if raw == 0:
        return None, None
    if raw < MIN_RECENCY_DAYS or raw > MAX_RECENCY_DAYS:
        return (
            None,
            f"Recency days must be between {MIN_RECENCY_DAYS} and {MAX_RECENCY_DAYS}, or 0 to disable.",
        )
    return raw, None


def recency_days_to_tpr(days: int) -> str:
    """Deterministically map bounded recency days to LinkedIn's f_TPR parameter."""
    normalized, error = normalize_recency_days(days)
    if error or normalized is None:
        raise ValueError(error or "invalid_recency_days")
    return f"r{normalized * SECONDS_PER_DAY}"


def build_linkedin_search_url(
    base_url: str,
    query: str,
    page_num: int,
    *,
    location: Optional[str] = None,
    recency_days: Optional[int] = None,
) -> str:
    """Build a correctly encoded LinkedIn Easy Apply search URL.

    Args:
        base_url: LinkedIn base URL (trailing slash tolerated).
        query: Search keywords query.
        page_num: 1-based page number; 25 results per page.
        location: Validated location filter; defaults to Worldwide when None.
        recency_days: Bounded recency in days; None disables the time filter.

    Returns:
        Fully URL-encoded search URL built with ``urlencode``.
    """
    params = {
        "keywords": query,
        "f_AL": "true",
        "location": location if location else DEFAULT_SEARCH_LOCATION,
        "start": (page_num - 1) * 25,
    }
    if recency_days:
        params["f_TPR"] = recency_days_to_tpr(recency_days)
    return f"{base_url.rstrip('/')}/jobs/search/?{urlencode(params)}"


def effective_search_params(
    cli_location: Optional[str],
    cli_recency_days: Optional[int],
    env_location: str,
    env_recency_days: Optional[int],
) -> tuple[str, Optional[int], Optional[str]]:
    """Resolve CLI-over-environment search parameters with validation.

    Environment values act as defaults; explicitly provided CLI values win.

    Returns:
        ``(location, recency_days, error)``. On validation failure the first two
        elements are empty placeholders and ``error`` describes the problem.
    """
    location, loc_err = normalize_search_location(cli_location)
    if loc_err:
        return "", None, loc_err

    recency, rec_err = normalize_recency_days(cli_recency_days)
    if rec_err:
        return "", None, rec_err

    if location is None:
        location, loc_err = normalize_search_location(env_location)
        if loc_err or location is None:
            return "", None, loc_err or "Configured search location is empty."

    if recency is None and cli_recency_days is None:
        recency, rec_err = normalize_recency_days(env_recency_days)
        if rec_err:
            return "", None, rec_err

    return location, recency, None


def resolve_state_search_params(state: Any, settings: Any) -> tuple[str, Optional[int]]:
    """Resolve effective search parameters from checkpoint state.

    Checkpoint keys are authoritative whenever present — including an
    explicitly stored ``None`` recency (filter disabled). Only a genuinely
    missing key from a legacy checkpoint falls back to configured environment
    settings, so a run saved with recency disabled can never silently resume
    under a newly configured filter.
    """
    state_is_mapping = isinstance(state, dict)

    if state_is_mapping and "search_location" in state:
        location = str(state.get("search_location") or DEFAULT_SEARCH_LOCATION)
    else:
        location = str(getattr(settings, "search_location", "") or DEFAULT_SEARCH_LOCATION)

    if state_is_mapping and "search_recency_days" in state:
        raw_recency = state.get("search_recency_days")
    else:
        raw_recency = getattr(settings, "search_recency_days", None)
    days, _error = normalize_recency_days(raw_recency)
    return location, days
