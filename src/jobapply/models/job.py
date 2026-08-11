"""Job-related data models."""

from pydantic import BaseModel, Field
from typing import Optional


class JobPosting(BaseModel):
    """LinkedIn job posting details."""

    job_id: str  # extracted from LinkedIn URL
    title: str
    company: str
    location: str
    url: str
    description: str
    posted_date: Optional[str] = None
    has_easy_apply: bool = True


class QualificationResult(BaseModel):
    """Result of job qualification evaluation."""

    qualified: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_matches: list[str]
    gaps: list[str]
    job_summary: str  # Summary of what you'll be working on
