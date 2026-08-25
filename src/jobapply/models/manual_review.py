"""Typed models for the durable manual-review queue.

All free-text fields carry explicit trust-boundary bounds so malformed or
hostile stored documents fail closed at parse time instead of flowing
unbounded data into state, summaries, or operator output.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ManualReviewState(str, Enum):
    """Lifecycle state of a manual-review queue item."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ResolutionLabel(str, Enum):
    """Explicit safe resolution labels; resolving never alters submission truth."""

    CONFIRMED_NOT_SUBMITTED = "confirmed_not_submitted"
    CONFIRMED_SUBMITTED = "confirmed_submitted"
    REQUIRES_FOLLOWUP = "requires_followup"


class ManualReviewItem(BaseModel):
    """One durable, bounded, redacted manual-review queue record."""

    item_id: str = Field(min_length=1, max_length=160)
    idempotency_key: str = Field(min_length=1, max_length=64)
    run_id: str = Field(min_length=1, max_length=64)
    job_id: Optional[str] = Field(default=None, max_length=128)
    title: Optional[str] = Field(default=None, max_length=256)
    company: Optional[str] = Field(default=None, max_length=256)
    url: Optional[str] = Field(default=None, max_length=512)
    reason_category: str = Field(min_length=1, max_length=120)
    reason_detail: Optional[str] = Field(default=None, max_length=1024)
    attempt_id: Optional[str] = Field(default=None, max_length=64)
    attempt_status: Optional[str] = Field(default=None, max_length=64)
    ambiguous_submission: bool = False
    state: ManualReviewState = ManualReviewState.OPEN
    operator_note: Optional[str] = Field(default=None, max_length=512)
    resolution_label: Optional[ResolutionLabel] = None
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    source_outcome: dict[str, Any] = Field(default_factory=dict)


class ManualReviewEnqueueResult(BaseModel):
    """Result of idempotently enqueueing a manual-review item."""

    queued: bool
    already_exists: bool = False
    item_id: str = Field(min_length=1, max_length=160)
    reason: Optional[str] = Field(default=None, max_length=120)


class ManualReviewTransitionResult(BaseModel):
    """Result of an atomic compare-and-set acknowledgement/resolution."""

    success: bool
    changed: bool = False
    not_found: bool = False
    conflict: bool = False
    no_op: bool = False
    item_id: str = Field(default="", max_length=160)
    state: Optional[ManualReviewState] = None
    reason: Optional[str] = Field(default=None, max_length=120)
