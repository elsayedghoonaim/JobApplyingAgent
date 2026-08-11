"""Settings module for jobapply using Pydantic Settings."""

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
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

    # Single LLM provider: Gemma through the native Google generateContent API.
    llm_model: Literal["gemma-4-31b-it"] = "gemma-4-31b-it"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Telegram
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    telegram_bot_name: str = "Magdy"
    telegram_user_title: str = "Boss"
    telegram_question_language: str = "English"

    # LinkedIn
    linkedin_base_url: str = "https://www.linkedin.com"
    linkedin_signin_timeout_seconds: int = Field(default=600, gt=0)

    # Agent behavior
    max_applications_per_session: int = Field(default=25, ge=0)
    daily_application_cap: int = Field(default=50, ge=0)
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
    pages_per_query: int = Field(default=3, ge=1, le=20)
    approval_timeout_seconds: int = Field(default=300, gt=0)
    form_qa_timeout_seconds: int = Field(default=300, gt=0)
    auto_accept_application_terms: bool = False

    # Persistence
    seen_jobs_collection: str = "seen_jobs"

    # Search
    search_queries: str = "Machine Learning Engineer,AI Engineer"

    @property
    def search_queries_list(self) -> list[str]:
        """Parse comma-separated search queries into a list."""
        return [q.strip() for q in self.search_queries.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
