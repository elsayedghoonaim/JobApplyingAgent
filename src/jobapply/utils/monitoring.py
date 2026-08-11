"""Enhanced logging and monitoring utilities."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(run_id: str, log_dir: str = "outputs") -> logging.Logger:
    """Setup logging to both file and console.

    Args:
        run_id: Current run ID for log file naming.
        log_dir: Directory to store log files.

    Returns:
        Configured logger instance.
    """
    # Windows terminals may default to CP1252; workflow output contains Unicode.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")

    # Create log directory
    log_path = Path(log_dir) / run_id
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("jobapply")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    # File handler (detailed logs)
    log_file = log_path / f"jobapply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (important messages only)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ProgressTracker:
    """Track and display progress of job application process."""

    def __init__(self, logger: logging.Logger):
        """Initialize progress tracker.

        Args:
            logger: Logger instance for progress updates.
        """
        self.logger = logger
        self.queries_completed = 0
        self.jobs_processed = 0
        self.jobs_qualified = 0
        self.applications_submitted = 0
        self.applications_skipped = 0
        self.errors = 0
        self.start_time = datetime.now()

    def update_query(self, query: str, page: int):
        """Update current query being processed."""
        self.logger.info(f"🔍 Searching: '{query}' (page {page})")

    def found_jobs(self, count: int):
        """Log number of jobs found."""
        self.logger.info(f"   Found {count} jobs")

    def evaluating_job(self, title: str, company: str):
        """Log job being evaluated."""
        self.jobs_processed += 1
        self.logger.info(f"\n📋 [{self.jobs_processed}] Evaluating: {title} at {company}")

    def job_qualified(self, score: float):
        """Log qualified job."""
        self.jobs_qualified += 1
        self.logger.info(f"   ✅ Qualified (score: {score:.2f})")

    def job_not_qualified(self, score: float, reason: str):
        """Log not qualified job."""
        self.logger.info(f"   ⏭️  Not qualified (score: {score:.2f}) - {reason[:80]}")

    def generating_documents(self):
        """Log document generation."""
        self.logger.info("   📝 Generating cover letter...")

    def urgent_edit_detected(self):
        """Log urgent edit detection."""
        self.logger.info("   🔴 Urgent edit recommended - awaiting approval...")

    def applying(self, dry_run: bool = False):
        """Log application submission."""
        mode = "DRY RUN" if dry_run else "LIVE"
        self.logger.info(f"   📤 Submitting application ({mode})...")

    def application_success(self):
        """Log successful application."""
        self.applications_submitted += 1
        self.logger.info("   ✅ Application submitted!")

    def application_skipped(self, reason: str):
        """Log skipped application."""
        self.applications_skipped += 1
        self.logger.info(f"   ⏭️  Skipped - {reason}")

    def error(self, error: str):
        """Log error."""
        self.errors += 1
        self.logger.error(f"   ❌ Error: {error}")

    def telegram_question(self, question: str):
        """Log Telegram Q&A."""
        self.logger.info(f"   ❓ Telegram Q&A: {question[:60]}...")

    def print_summary(self):
        """Print session summary."""
        duration = datetime.now() - self.start_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)

        # Calculate not qualified count
        not_qualified = self.jobs_processed - self.jobs_qualified

        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 SESSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"⏱️  Duration: {hours}h {minutes}m")
        self.logger.info(f"🔍 Jobs Processed: {self.jobs_processed}")
        self.logger.info(f"✅ Qualified: {self.jobs_qualified}")
        self.logger.info(f"❌ Not Qualified: {not_qualified}")
        self.logger.info(f"📤 Applications Submitted: {self.applications_submitted}")
        self.logger.info(f"⏭️  Skipped: {self.applications_skipped}")
        self.logger.info(f"❌ Errors: {self.errors}")
        self.logger.info("=" * 60)
