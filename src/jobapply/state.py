"""State schema for LangGraph."""

from typing import Optional, TypedDict


class JobApplyState(TypedDict):
    """State schema for the job application graph."""

    # ── Search iteration ──
    search_queries: list[str]
    search_location: str  # effective LinkedIn location filter (CLI > env > Worldwide)
    search_recency_days: Optional[int]  # bounded recency days; None = disabled
    current_query_index: int  # which query we're on
    current_page: int  # pagination within a query
    pages_per_query: int  # max pages to fetch per query
    job_listings: list[dict]  # jobs fetched for current query+page
    current_job_index: int  # pointer into job_listings
    current_job: Optional[dict]  # active job: {job_id, title, company, location, url, description}
    search_failed: bool  # infrastructure failure; stop instead of paginating repeatedly
    query_exhausted: Optional[bool]  # LinkedIn search returned no results for this query

    # ── Deduplication ──
    seen_job_ids: set[str]  # in-memory set (also persisted to MongoDB)

    # ── Qualification ──
    qualification_result: Optional[dict]  # {qualified, score, reasoning, matches, gaps}

    # ── Generation ──
    edits_urgent: bool
    proposed_edits: Optional[str]
    edit_reasoning: Optional[str]
    cover_letter_text: Optional[str]
    resume_path: Optional[str]  # base PDF or edited PDF
    cover_letter_path: Optional[str]

    # ── Approval (rare path) ──
    approval_status: Optional[str]  # "approved" | "use_base" | "skip" | None
    approval_nonce: Optional[str]  # unique ID for Telegram correlation

    # ── Execution ──
    application_status: Optional[str]  # ApplicationStatus value
    application_error: Optional[str]
    form_qa_exchanges: Optional[list[dict]]  # [{question, answer, timed_out}] from Telegram Q&A

    # ── Notification ──
    notification_sent: bool
    outbox_pending_count: Optional[int]
    outbox_unknown_count: Optional[int]
    # Bounded serialized queue items whose durable enqueue failed and are
    # retried idempotently at central safe boundaries. The overflow counter
    # explicitly records entries evicted when the bounded list was full.
    manual_review_queue_pending: list[dict]
    manual_review_queue_overflow: int

    # ── Account safety ──
    account_safety_paused: bool
    account_safety_barrier_type: Optional[str]
    account_safety_reason: Optional[str]
    account_safety_stage: Optional[str]
    account_safety_url: Optional[str]
    account_safety_detected_at: Optional[str]
    account_safety_evidence: Optional[str]
    account_safety_resume_instructions: Optional[str]

    # ── Session tracking ──
    run_id: str  # also used as LangGraph thread_id
    dry_run: bool
    applications_count: int  # this session
    daily_applications_count: int  # today's total (loaded from MongoDB)
    max_applications: int  # session cap
    daily_application_cap: int  # daily cap
    max_jobs_to_evaluate: Optional[int]  # limit total jobs to evaluate (None = unlimited)
    jobs_evaluated_count: int  # how many jobs evaluated so far
    qualified_jobs_count: int  # how many qualified jobs in this session
    not_qualified_jobs_count: int  # how many not qualified jobs in this session
    application_outcomes: list[dict]  # canonical per-qualified-job outcomes
    applied_jobs: list[dict]  # per-job summaries
    skipped_jobs: list[dict]  # includes Q&A timeout skips
    logs: list[str]
    errors: list[str]
