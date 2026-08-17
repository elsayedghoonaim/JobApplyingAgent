"""Main entry point for jobapply CLI."""

import argparse
import asyncio
from datetime import datetime
from typing import Optional
from uuid import uuid4

from langchain_core.tracers.context import tracing_v2_enabled

from jobapply.graph import compile_graph, make_graph_config
from jobapply.settings import get_settings
from jobapply.utils.account_safety import (
    normalize_safety_log_payload,
)
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.llm import close_llm_client
from jobapply.utils.monitoring import ProgressTracker, setup_logging
from jobapply.utils.paths import get_run_output_dir, validate_run_id
from jobapply.utils.redaction import redact_string
from jobapply.utils.tracing import get_session_metadata


def resolve_graph_input(resume_requested, snapshot, initial_state):
    """Choose new input, checkpoint continuation, or a completed saved result."""
    if not resume_requested or not snapshot or not snapshot.values:
        return initial_state, None
    if snapshot.next:
        return None, None
    return None, dict(snapshot.values)


async def run(
    run_id: Optional[str] = None,
    dry_run: Optional[bool] = None,
    max_applications: Optional[int] = None,
    max_jobs: Optional[int] = None,
):
    """Main entry point for the job application agent.

    Args:
        run_id: Reuse to resume a crashed run. None = fresh run (uuid4).
        dry_run: Override .env setting.
        max_applications: Override .env setting.
    """
    settings = get_settings()
    effective_dry_run = dry_run if dry_run is not None else settings.dry_run
    resume_requested = run_id is not None
    if resume_requested:
        run_id = validate_run_id(run_id)
    else:
        run_id = str(uuid4())

    # Setup safe logging directory
    output_dir = get_run_output_dir(run_id)
    logger = setup_logging(run_id)
    logger.info(f"🚀 Starting job application agent (run_id: {run_id})")
    logger.info(f"📁 Logs: {output_dir}")

    # Load daily count from MongoDB
    dedup = DeduplicationStore()
    daily_count = await dedup.get_daily_count()

    logger.info(f"📊 Application count today: {daily_count}")

    # Initialize state
    initial_state = {
        "search_queries": settings.search_queries_list,
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": settings.pages_per_query,
        "job_listings": [],
        "current_job_index": 0,
        "current_job": None,
        "search_failed": False,
        "seen_job_ids": set(),
        "qualification_result": None,
        "edits_urgent": False,
        "proposed_edits": None,
        "edit_reasoning": None,
        "cover_letter_text": None,
        "resume_path": None,
        "cover_letter_path": None,
        "approval_status": None,
        "approval_nonce": None,
        "application_status": None,
        "application_error": None,
        "form_qa_exchanges": None,
        "notification_sent": False,
        "outbox_pending_count": 0,
        "outbox_unknown_count": 0,
        "account_safety_paused": False,
        "account_safety_barrier_type": None,
        "account_safety_reason": None,
        "account_safety_stage": None,
        "account_safety_url": None,
        "account_safety_detected_at": None,
        "account_safety_evidence": None,
        "account_safety_resume_instructions": None,
        "run_id": run_id,
        "dry_run": effective_dry_run,
        "applications_count": 0,
        "daily_applications_count": daily_count,
        "max_applications": (
            max_applications
            if max_applications is not None
            else settings.max_applications_per_session
        ),
        "daily_application_cap": settings.daily_application_cap,
        "max_jobs_to_evaluate": max_jobs,  # New: limit total jobs to evaluate
        "jobs_evaluated_count": 0,  # New: track how many jobs evaluated
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
    }

    if initial_state["dry_run"]:
        logger.warning("⚠️  DRY RUN MODE - No applications will be submitted")

    # Compile graph
    logger.info("🔧 Compiling graph...")
    graph = compile_graph()

    # Prepare session metadata for tracing
    session_metadata = get_session_metadata(
        run_id=run_id,
        dry_run=initial_state["dry_run"],
        max_applications=initial_state["max_applications"],
        queries_count=len(initial_state["search_queries"]),
        edge_port=settings.edge_debug_port,
    )

    # Prepare config with tracing metadata and tags
    config = make_graph_config(
        run_id=run_id,
        tags=["jobapply", "dry-run" if initial_state["dry_run"] else "live", f"run:{run_id}"],
        metadata=session_metadata,
    )

    # Create progress tracker
    tracker = ProgressTracker(logger)

    # Run graph with LangSmith tracing
    logger.info("▶️  Starting execution...\n")
    try:
        with tracing_v2_enabled(
            project_name=None,  # uses LANGSMITH_PROJECT from env
            tags=["jobapply-session"],
        ) as tracer:
            snapshot = None
            if resume_requested:
                snapshot = await graph.aget_state(config)
            graph_input, saved_result = resolve_graph_input(
                resume_requested,
                snapshot,
                initial_state,
            )
            if resume_requested and snapshot and snapshot.values:
                if snapshot.values.get("account_safety_paused"):
                    norm = normalize_safety_log_payload(
                        run_id=str(run_id or "unknown"),
                        barrier_type=str(
                            snapshot.values.get("account_safety_barrier_type") or "SECURITY BARRIER"
                        ),
                        stage=str(snapshot.values.get("account_safety_stage") or "unknown"),
                        reason=str(
                            snapshot.values.get("account_safety_reason")
                            or "Account safety challenge detected"
                        ),
                        url=str(
                            snapshot.values.get("account_safety_url") or "https://www.linkedin.com"
                        ),
                        resume_instructions=str(
                            snapshot.values.get("account_safety_resume_instructions") or ""
                        ),
                    )
                    logger.warning(
                        f"🛑 Saved run '{norm['run_id']}' was paused due to an account-safety barrier: {norm['reason']}. "
                        "Resolve manually in Microsoft Edge, then start a NEW session (do not reuse the paused run ID)."
                    )
                else:
                    logger.info(
                        "Resuming from the saved checkpoint"
                        if snapshot.next
                        else "Run is already complete; using its saved result"
                    )

            result = (
                saved_result
                if saved_result is not None
                else await graph.ainvoke(graph_input, config)
            )

            # Get and log LangSmith trace URL
            try:
                trace_url = tracer.get_run_url()
                logger.info(f"🔗 LangSmith trace: {trace_url}")
            except Exception:
                pass  # Ignore if we can't get the URL

            logger.info("\n✅ Execution complete!")

            # Get accurate counts from the final state of the graph
            jobs_processed = result.get("jobs_evaluated_count", 0)
            qualified_count = result.get("qualified_jobs_count", 0)
            not_qualified_count = result.get("not_qualified_jobs_count", 0)
            submitted_count = result.get("applications_count", 0)
            dry_run_count = sum(
                1
                for outcome in result.get("application_outcomes", [])
                if outcome.get("status") == "dry_run"
            )

            # Calculate duration
            duration_seconds = (datetime.now() - tracker.start_time).total_seconds()
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)

            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("📊 SESSION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"⏱️  Duration: {hours}h {minutes}m")
            logger.info(f"🔍 Jobs Evaluated: {jobs_processed}")
            logger.info(f"✅ Qualified: {qualified_count}")
            logger.info(f"❌ Not Qualified: {not_qualified_count}")
            logger.info(f"📤 Applications Submitted: {submitted_count}")
            logger.info(f"🧪 Dry Run Ready: {dry_run_count}")
            logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user. State saved to checkpoint.")
        logger.info(f"   Resume with: jobapply --run-id {run_id}")
    except Exception as e:
        logger.error(f"❌ Error: {redact_string(str(e))}")
        logger.info(f"   Resume with: jobapply --run-id {run_id}")
        raise
    finally:
        await close_llm_client()
        await DeduplicationStore.close()


def main():
    """CLI entry point."""

    def valid_run_id_arg(value: str) -> str:
        try:
            return validate_run_id(value)
        except ValueError as err:
            raise argparse.ArgumentTypeError(str(err)) from err

    def non_negative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("must be zero or greater")
        return parsed

    parser = argparse.ArgumentParser(
        description="Autonomous job-hunting agent powered by LangGraph"
    )
    parser.add_argument(
        "--run-id",
        type=valid_run_id_arg,
        help="Reuse run ID to resume from checkpoint",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Dry run mode - no actual submissions",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Disable dry run mode - submit actual applications",
    )
    parser.add_argument(
        "--max-applications",
        type=non_negative_int,
        help="Override max applications per session",
    )
    parser.add_argument(
        "--max-jobs",
        type=non_negative_int,
        help="Maximum number of jobs to evaluate (will fetch multiple pages if needed)",
    )

    args = parser.parse_args()

    asyncio.run(
        run(
            run_id=args.run_id,
            dry_run=args.dry_run,
            max_applications=args.max_applications,
            max_jobs=args.max_jobs,
        )
    )


if __name__ == "__main__":
    main()
