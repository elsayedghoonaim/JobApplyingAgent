"""Job-related data models."""

from typing import Optional

from pydantic import BaseModel, Field


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


class CombinedJobAnalysis(BaseModel):
    """Combined job qualification scoring and structured description extraction."""

    qualified: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_matches: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    job_summary: str
    parsed_location: Optional[str] = None
    duration: Optional[str] = None
    work_type: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    required_languages: list[str] = Field(default_factory=list)
    clean_description: Optional[str] = None
