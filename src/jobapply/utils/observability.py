"""Centralized structured logging with recursive redaction and hard bounds.

Every file record is exactly one valid JSON object (JSON Lines) with stable
bounded fields. Console output remains concise and human-readable. Hostile
DOM/LLM/error content cannot create enormous logs because strings, mapping
keys, collection sizes, and nesting depth are all capped before serialization.
Sanitization is allocation-safe: hostile mappings/sequences never have more
than MAX+1 entries consumed per container, so oversized containers cannot
exhaust memory. A shared per-call budget additionally caps the TOTAL number of
entries visited across all nesting, so exponentially branching structures stay
linear. Set serialization is deterministic across processes.
"""

import hashlib
import json
import logging
import math
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from jobapply.utils.redaction import is_sensitive_key, redact_string

LOGGER_NAME = "jobapply"

# ── Bounds so hostile content cannot create enormous logs ──
MAX_MESSAGE_LENGTH = 500
MAX_CONSOLE_MESSAGE_LENGTH = 500
MAX_FIELD_LENGTH = 128
MAX_EVENT_LENGTH = 80
MAX_STAGE_LENGTH = 60
MAX_STATUS_LENGTH = 40
MAX_DETAIL_STRING_LENGTH = 300
MAX_DETAIL_DEPTH = 4
MAX_DETAIL_ITEMS = 50
MAX_DETAIL_KEYS = 50
MAX_DETAIL_KEY_LENGTH = 64
# Global per-call budget covering every mapping entry and sequence element
# across all nesting levels (secondary to nothing; per-container limits and
# depth remain as additional secondary bounds).
MAX_TOTAL_DETAIL_ENTRIES = 400

# LogRecord-reserved attribute names that must never be shadowed via ``extra``.
_RESERVED_LOG_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_LEVEL_NAMES = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Prevent lastResort stderr noise when no handlers are configured yet.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def bound_text(value: Any, max_length: int = MAX_MESSAGE_LENGTH) -> str:
    """Return a redacted, length-bounded string representation."""
    text = redact_string(str(value))
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def fingerprint(value: Any) -> str:
    """Stable short hash identifying sensitive text without revealing it."""
    normalized = str(value).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:16]


def exception_summary(exc: BaseException) -> dict[str, str]:
    """Represent an exception as its safe type plus bounded redacted message."""
    return {
        "type": type(exc).__name__,
        "message": bound_text(exc, MAX_DETAIL_STRING_LENGTH),
    }


def _safe_len(container: Any) -> int | None:
    """Return len() when safely available, otherwise None (never raises)."""
    try:
        size = len(container)
    except Exception:
        return None
    return size if isinstance(size, int) and size >= 0 else None


class _SanitizeBudget:
    """Fresh-per-call budget shared across every nesting level.

    A new instance is created for each top-level ``sanitize_log_value`` call
    (never a mutable default), so each record/value starts with a full budget.
    """

    __slots__ = ("remaining",)

    def __init__(self, total: int = MAX_TOTAL_DETAIL_ENTRIES) -> None:
        self.remaining = total


def _next_entry(iterator: Any, budget: _SanitizeBudget) -> tuple[bool, Any, bool]:
    """Fetch the next entry under the strict global budget.

    The budget is checked BEFORE calling ``next()``; nothing is inspected once
    it is exhausted. Every successfully fetched or probed entry is charged to
    the shared budget. Returns ``(has_entry_or_probe, item, charged)`` where
    ``has_entry_or_probe`` is False on clean StopIteration.
    """
    if budget.remaining <= 0:
        return False, None, False
    try:
        item = next(iterator)
    except StopIteration:
        return False, None, False
    budget.remaining -= 1
    return True, item, True


def _container_overflow(container: Any, limit: int, iterator: Any, budget: _SanitizeBudget) -> bool:
    """Decide whether a filled container has more entries beyond its limit.

    Sized containers are decided from their cardinality without iteration.
    Unsized containers get exactly one extra probe, which is charged to the
    shared budget like any other inspected entry.
    """
    total = _safe_len(container)
    if total is not None:
        return total > limit
    has_probe, _, _ = _next_entry(iterator, budget)
    return has_probe


def _sanitize_mapping(mapping: Any, depth: int, budget: _SanitizeBudget) -> Any:
    """Redact and bound a mapping consuming at most MAX_DETAIL_KEYS (+1 probe) entries."""
    try:
        entries = mapping.items()
        iterator = iter(entries)
    except Exception:
        return "[UNSERIALIZABLE_MAPPING]"

    kept: dict[str, Any] = {}
    consumed = 0
    overflow = False
    exhausted = False
    while True:
        if consumed >= MAX_DETAIL_KEYS:
            # Per-container limit reached; decide overflow without further
            # inspection when possible (probe, if used, consumes budget).
            overflow = _container_overflow(mapping, MAX_DETAIL_KEYS, iterator, budget)
            break
        # Strict global budget check BEFORE pulling the next entry.
        if budget.remaining <= 0:
            exhausted = True
            break
        has_entry, entry, _charged = _next_entry(iterator, budget)
        if not has_entry:
            break
        raw_key, raw_item = entry
        consumed += 1
        try:
            key_text = str(raw_key)
        except Exception:
            key_text = "_"
        key = redact_string(key_text)[:MAX_DETAIL_KEY_LENGTH].strip() or "_"
        if is_sensitive_key(key_text):
            kept[key] = "[REDACTED]"
        else:
            kept[key] = _sanitize_value(raw_item, depth + 1, budget)

    if exhausted:
        kept["[truncated_budget]"] = "[TRUNCATED_BUDGET]"
    if overflow:
        total_keys = _safe_len(mapping)
        remaining = "unknown" if total_keys is None else str(total_keys - MAX_DETAIL_KEYS)
        kept["[truncated_keys]"] = remaining
    return kept


def _sanitize_sequence(sequence: Any, depth: int, budget: _SanitizeBudget) -> Any:
    """Redact and bound a sequence/iterable consuming at most MAX+1 entries."""
    try:
        iterator = iter(sequence)
    except Exception:
        return "[UNSERIALIZABLE_SEQUENCE]"

    bounded_items: list[Any] = []
    consumed = 0
    overflow = False
    exhausted = False
    while True:
        if consumed >= MAX_DETAIL_ITEMS:
            overflow = _container_overflow(sequence, MAX_DETAIL_ITEMS, iterator, budget)
            break
        # Strict global budget check BEFORE pulling the next element.
        if budget.remaining <= 0:
            exhausted = True
            break
        has_entry, item, _charged = _next_entry(iterator, budget)
        if not has_entry:
            break
        consumed += 1
        bounded_items.append(_sanitize_value(item, depth + 1, budget))

    if exhausted:
        bounded_items.append("[TRUNCATED_BUDGET]")
    if overflow:
        total_items = _safe_len(sequence)
        remaining = "unknown" if total_items is None else str(total_items - MAX_DETAIL_ITEMS)
        bounded_items.append(f"[TRUNCATED_ITEMS {remaining}]")
    return bounded_items


def _sanitize_set(values: Any, depth: int, budget: _SanitizeBudget) -> Any:
    """Deterministically serialize a set without iterating oversized sets.

    Built-in sets/frozensets always report a safe cardinality. Anything larger
    than the item limit collapses to a cardinality-only marker, so output never
    depends on hash iteration order. Smaller sets are fully sanitized and then
    sorted, which is also deterministic. The shared budget gates every pull.
    """
    total_items = _safe_len(values)
    if total_items is None:
        return "[UNSERIALIZABLE_SET]"
    if total_items > MAX_DETAIL_ITEMS:
        return f"[TRUNCATED_SET {total_items}]"

    try:
        iterator = iter(values)
    except Exception:
        return "[UNSERIALIZABLE_SET]"

    rendered: list[str] = []
    exhausted = False
    while True:
        # Strict global budget check BEFORE pulling the next element.
        if budget.remaining <= 0:
            exhausted = True
            break
        has_entry, item, _charged = _next_entry(iterator, budget)
        if not has_entry:
            break
        rendered.append(str(_sanitize_value(item, depth + 1, budget)))

    if exhausted:
        rendered.append("[TRUNCATED_BUDGET]")
    rendered.sort()
    return rendered


def sanitize_log_value(value: Any) -> Any:
    """Recursively redact and bound a value into JSON-serializable primitives.

    A fresh global budget is created per call. Hostile containers are consumed
    lazily with at most MAX+1 entries touched per container and at most
    ``MAX_TOTAL_DETAIL_ENTRIES`` entries visited overall, so unbounded or
    exponentially branching inputs cannot exhaust memory. Iteration/repr
    failures degrade to bounded markers instead of dropping the whole event.
    """
    return _sanitize_value(value, 0, _SanitizeBudget())


def _sanitize_value(value: Any, depth: int, budget: _SanitizeBudget) -> Any:
    try:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else bound_text(value)
        if isinstance(value, str):
            return bound_text(value, MAX_DETAIL_STRING_LENGTH)
        if isinstance(value, (bytes, bytearray)):
            return bound_text(repr(bytes(value)), MAX_DETAIL_STRING_LENGTH)
        if isinstance(value, BaseException):
            return exception_summary(value)
        if depth >= MAX_DETAIL_DEPTH:
            return "[TRUNCATED_DEPTH]"
        if isinstance(value, Mapping):
            return _sanitize_mapping(value, depth, budget)
        if isinstance(value, (set, frozenset)):
            return _sanitize_set(value, depth, budget)
        if isinstance(value, (list, tuple)):
            return _sanitize_sequence(value, depth, budget)
        if isinstance(value, Iterable):
            return _sanitize_sequence(value, depth, budget)
        return bound_text(repr(value), MAX_DETAIL_STRING_LENGTH)
    except Exception:
        return "[UNSERIALIZABLE_VALUE]"


def optional_str(value: Any) -> str | None:
    """Return a bounded string for truthy values, otherwise None."""
    if value is None or value == "":
        return None
    return bound_text(value, MAX_FIELD_LENGTH)


class StructuredJsonlFormatter(logging.Formatter):
    """Serialize each record as exactly one valid, bounded JSON object line."""

    def format(self, record: logging.LogRecord) -> str:
        details = getattr(record, "details", None)
        raw_exception = getattr(record, "exception_summary", None)
        payload: dict[str, Any] = {
            "timestamp": (
                datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            ),
            "level": record.levelname,
            "event": bound_text(getattr(record, "event", "log"), MAX_EVENT_LENGTH),
            "message": bound_text(record.getMessage(), MAX_MESSAGE_LENGTH),
            "run_id": optional_str(getattr(record, "run_id", None)),
            "job_id": optional_str(getattr(record, "job_id", None)),
            "attempt_id": optional_str(getattr(record, "attempt_id", None)),
            "node": optional_str(getattr(record, "node", None)),
            "stage": optional_str(getattr(record, "stage", None)),
            "outcome": optional_str(getattr(record, "outcome", None)),
            "status": optional_str(getattr(record, "status", None)),
            # Fresh budget per field: externally supplied values can never
            # bypass redaction/bounds, including the exception summary.
            "details": sanitize_log_value(details) if details is not None else None,
            "exception": sanitize_log_value(raw_exception) if raw_exception is not None else None,
        }
        try:
            # All payload values are already redacted/bounded primitives or
            # marker strings, so no unredacted default serializer is needed.
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            fallback = {
                "timestamp": payload.get("timestamp"),
                "level": record.levelname,
                "event": "log.serialization_fallback",
                "message": "A log record could not be serialized; details dropped.",
                "run_id": payload.get("run_id"),
                "job_id": None,
                "attempt_id": None,
                "node": None,
                "stage": None,
                "outcome": None,
                "status": None,
                "details": None,
                "exception": None,
            }
            return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))


def log_event(
    level: int | str,
    event: str,
    message: str | None = None,
    *,
    run_id: str | None = None,
    job_id: str | None = None,
    attempt_id: str | None = None,
    node: str | None = None,
    stage: str | None = None,
    outcome: str | None = None,
    status: str | None = None,
    details: Mapping[str, Any] | None = None,
    exc: BaseException | None = None,
) -> None:
    """Emit one structured, recursively redacted event through the jobapply logger.

    This is the single centralized helper for workflow logging. Callers never
    hand-build JSON. Logging failures are swallowed so they can never alter
    workflow control flow.
    """
    try:
        if isinstance(level, str):
            log_level = _LEVEL_NAMES.get(level.strip().lower(), logging.INFO)
        else:
            log_level = int(level)

        effective_logger = logging.getLogger(LOGGER_NAME)
        resolved_message = bound_text(
            message if message is not None else event.replace("_", " "),
            MAX_MESSAGE_LENGTH,
        )

        extras: dict[str, Any] = {
            "event": bound_text(event, MAX_EVENT_LENGTH),
            "run_id": optional_str(run_id),
            "job_id": optional_str(job_id),
            "attempt_id": optional_str(attempt_id),
            "node": optional_str(node) or optional_str(stage),
            "stage": optional_str(stage),
            "outcome": optional_str(outcome),
            "status": optional_str(status) or optional_str(outcome),
            "details": sanitize_log_value(details) if details is not None else None,
            "exception_summary": exception_summary(exc) if exc is not None else None,
        }
        # Defensive: never shadow reserved LogRecord attributes via ``extra``.
        safe_extras = {k: v for k, v in extras.items() if k not in _RESERVED_LOG_RECORD_ATTRS}

        effective_logger.log(log_level, resolved_message, extra=safe_extras)
    except Exception:
        pass
