"""Settings module for jobapply using Pydantic Settings."""

import os
from functools import lru_cache
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file explicitly
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="JOBAPPLY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields in .env
    )

    # User Data
    data_dir: str = "user-data"

    def resolve_data_path(self, *parts: str) -> str:
        """Resolve a path under the configured data directory safely working from any CWD."""
        path = self.data_dir
        if not os.path.isabs(path):
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(repo_root, path)
        return os.path.abspath(os.path.join(path, *parts))

    # MongoDB
    mongodb_url: str = Field(default="mongodb://localhost:27017")
    mongodb_db: str = "jobapply"

    # Gemini uses the native generateContent API; OpenRouter uses its
    # OpenAI-compatible chat completions API.
    llm_provider: Literal["gemini", "openrouter"] = "gemini"
    llm_model: str = "gemma-4-31b-it"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    llm_fallback_models: str = "openrouter/free"

    @property
    def llm_fallback_models_list(self) -> list[str]:
        """Return configured OpenRouter fallback model IDs in priority order."""
        return [model.strip() for model in self.llm_fallback_models.split(",") if model.strip()]

    # Telegram
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    telegram_bot_name: str = "Magdy"
    telegram_user_title: str = "Boss"
    telegram_question_language: str = "English"
    report_timezone: str = "Africa/Cairo"

    # LinkedIn
    linkedin_base_url: str = "https://www.linkedin.com"
    linkedin_signin_timeout_seconds: int = Field(default=600, gt=0)
    linkedin_navigation_timeout_ms: int = Field(default=60_000, ge=10_000, le=180_000)

    # Agent behavior
    max_applications: int = Field(default=50, ge=0)
    max_applications_per_session: Optional[int] = Field(default=None, ge=0)
    daily_application_cap: Optional[int] = Field(default=None, ge=0)
    qualification_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    urgent_edit_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    action_delay_ms: int = Field(default=2000, ge=0)
    delay_jitter_percent: int = Field(default=30, ge=0, le=100)
    job_load_timeout_ms: int = Field(default=3000, gt=0)
    edge_debug_port: int = Field(default=9222, ge=1, le=65535)
    edge_auto_launch: bool = True
    edge_executable_path: str = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    edge_user_data_dir: str = r"%LOCALAPPDATA%\JobApply\EdgeProfile"
    dry_run: bool = True
    headless: bool = False
    pages_per_query: int = Field(default=100, ge=1, le=100)
    approval_timeout_seconds: int = Field(default=300, gt=0)
    form_qa_timeout_seconds: int = Field(default=300, gt=0)
    auto_accept_application_terms: bool = False

    @property
    def effective_max_applications(self) -> int:
        """Return configured max applications, preferring session override if set."""
        if self.max_applications_per_session is not None:
            return self.max_applications_per_session
        return self.max_applications

    # Persistence
    seen_jobs_collection: str = "seen_jobs"
    application_attempts_collection: str = "application_attempts"
    application_quotas_collection: str = "application_quotas"
    telegram_correlations_collection: str = "telegram_correlations"
    telegram_cursors_collection: str = "telegram_cursors"
    notification_outbox_collection: str = "notification_outbox"
    candidate_facts_collection: str = "candidate_facts"
    candidate_fact_validity_days: int = Field(default=365, ge=1, le=3650)
    candidate_fact_history_limit: int = Field(default=20, ge=1, le=100)
    telegram_correlation_ttl_seconds: int = Field(default=604800, ge=3600)
    telegram_poll_lease_seconds: int = Field(default=30, ge=5)
    outbox_max_attempts: int = Field(default=3, ge=1)
    outbox_initial_backoff_seconds: int = Field(default=5, ge=1)
    outbox_max_backoff_seconds: int = Field(default=300, ge=1)
    outbox_lease_seconds: int = Field(default=60, ge=5)

    # LLM Prompt Bounding
    max_job_description_chars: int = Field(default=8000, ge=500, le=50000)
    max_clean_description_chars: int = Field(default=3500, ge=500, le=10000)
    max_profile_context_chars: int = Field(default=6000, ge=500, le=50000)
    max_resume_context_chars: int = Field(default=8000, ge=500, le=50000)

    # Persistence
    manual_review_collection: str = "manual_review_queue"

    # Search & Card Stability Loader
    search_queries: str = "Machine Learning Engineer,AI Engineer"
    target_title_keywords: str = "Machine Learning,ML,MLOps,Deep Learning"
    exclude_senior_titles: bool = True
    allowed_languages: str = "Arabic,English"
    excluded_locations: str = (
        "Palestine,Palastin,Palastine,State of Palestine,Palestinian Territory,Palestinian Territories,Israel,Israeli,West Bank,"
        "Gaza,Gaza Strip,Tel Aviv,Jerusalem,Haifa,Beer Sheva,Beersheba,Ashdod,"
        "Netanya,Petah Tikva,Rishon LeZion,Ramat Gan,Herzliya,فلسطين,إسرائيل,ישראל"
    )
    search_location: str = Field(default="Worldwide", min_length=1, max_length=200)
    # Bounded positive number of days; unset/empty or 0 disables the recency filter.
    search_recency_days: Optional[int] = Field(default=None, ge=0, le=30)

    @field_validator("search_recency_days", mode="before")
    @classmethod
    def _empty_recency_disables_filter(cls, value: object) -> object:
        """Treat an empty/unset environment value as 'recency disabled'."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    search_max_scroll_rounds: int = Field(default=5, ge=1, le=20)
    search_card_stability_rounds: int = Field(default=2, ge=1, le=10)
    search_scroll_delay_seconds: float = Field(default=0.3, ge=0.05, le=5.0)

    @property
    def search_queries_list(self) -> list[str]:
        """Parse comma-separated search queries into a list."""
        return [q.strip() for q in self.search_queries.split(",") if q.strip()]

    @property
    def target_title_keywords_list(self) -> list[str]:
        """Parse the title phrases that make a job eligible."""
        return [value.strip() for value in self.target_title_keywords.split(",") if value.strip()]

    @property
    def allowed_languages_list(self) -> list[str]:
        """Parse languages the candidate accepts as job requirements."""
        return [value.strip() for value in self.allowed_languages.split(",") if value.strip()]

    @property
    def excluded_locations_list(self) -> list[str]:
        """Parse locations where jobs must never be processed."""
        return [value.strip() for value in self.excluded_locations.split(",") if value.strip()]

    @field_validator(
        "search_queries",
        "target_title_keywords",
        "allowed_languages",
        "excluded_locations",
    )
    @classmethod
    def _comma_list_must_not_be_empty(cls, value: str) -> str:
        """Reject empty comma-separated personalization lists."""
        if not any(part.strip() for part in value.split(",")):
            raise ValueError("must contain at least one non-empty value")
        return value


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
