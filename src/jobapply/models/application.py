"""Application-related data models."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ApplicationStatus(str, Enum):
    """Status of a job application."""

    SUBMITTED = "submitted"
    DRY_RUN = "dry_run"
    SKIPPED = "skipped"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"
    FAILED = "failed"


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
