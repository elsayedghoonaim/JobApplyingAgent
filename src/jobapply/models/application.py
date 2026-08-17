"""Application-related data models."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ApplicationStatus(str, Enum):
    """Status of a job application."""

    SUBMITTED = "submitted"
    DRY_RUN = "dry_run"
    SKIPPED = "skipped"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"
    FAILED = "failed"


class AttemptStatus(str, Enum):
    """Lifecycle status for a live job application attempt."""

    CREATED = "created"
    QUOTA_RESERVED = "quota_reserved"
    SUBMISSION_UNKNOWN = "submission_unknown"
    SUBMITTED = "submitted"
    RELEASED = "released"


LEGAL_ATTEMPT_TRANSITIONS: dict[AttemptStatus, set[AttemptStatus]] = {
    AttemptStatus.CREATED: {AttemptStatus.QUOTA_RESERVED, AttemptStatus.RELEASED},
    AttemptStatus.QUOTA_RESERVED: {AttemptStatus.SUBMISSION_UNKNOWN, AttemptStatus.RELEASED},
    AttemptStatus.SUBMISSION_UNKNOWN: {AttemptStatus.SUBMITTED},
    AttemptStatus.SUBMITTED: set(),
    AttemptStatus.RELEASED: {AttemptStatus.CREATED},
}


class AttemptPreflightResult(BaseModel):
    """Result of a read-only attempt preflight check at execution entry."""

    can_proceed: bool
    status: Optional[AttemptStatus] = None
    reason: Optional[str] = None
    existing_attempt_id: Optional[str] = None


class AttemptTransitionResult(BaseModel):
    """Result of a state transition on an application attempt."""

    success: bool
    changed: bool = False
    status: AttemptStatus
    attempt_id: str
    reason: Optional[str] = None


class ApplicationAttempt(BaseModel):
    """Persistent live application attempt record."""

    attempt_id: str
    job_id: str
    run_id: str
    status: AttemptStatus = AttemptStatus.CREATED
    created_at: datetime
    updated_at: datetime
    reserved_at: Optional[datetime] = None
    unknown_at: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    release_reason: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class DocumentBundle(BaseModel):
    """Bundle of documents for an application."""

    resume_path: str
    cover_letter_path: Optional[str] = None
    was_edited: bool = False


class ApplicationResult(BaseModel):
    """Result of a job application attempt."""

    status: ApplicationStatus
    error: Optional[str] = None
    timestamp: str
    job_id: str
    job_url: str
    job_title: str
    company: str
    score: float
    resume_edited: bool = False
    safety_issues: Optional[list[str]] = None
