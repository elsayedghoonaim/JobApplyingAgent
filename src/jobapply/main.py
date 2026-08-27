"""Main entry point for jobapply CLI."""

import argparse
import asyncio
import inspect
import logging
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import tracing_v2_enabled

from jobapply.doctor import run_doctor_cli
from jobapply.graph import compile_graph, make_graph_config
from jobapply.review_cli import run_review_cli
from jobapply.settings import get_settings
from jobapply.utils.account_safety import (
    normalize_safety_log_payload,
)
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.llm import close_llm_client
from jobapply.utils.monitoring import ProgressTracker, setup_logging, shutdown_logging
from jobapply.utils.observability import log_event
from jobapply.utils.paths import get_run_output_dir, validate_run_id
from jobapply.utils.redaction import redact_string
from jobapply.utils.search_config import effective_search_params
from jobapply.utils.summary import (
    TERMINAL_STATUS_COMPLETED,
    TERMINAL_STATUS_FAILED,
    TERMINAL_STATUS_INTERRUPTED,
    TERMINAL_STATUS_PAUSED,
    build_summary,
    write_summary,
)
from jobapply.utils.tracing import get_session_metadata


def resolve_graph_input(resume_requested, snapshot, initial_state):
    """Choose new input, checkpoint continuation, or a completed saved result."""
    if not resume_requested or not snapshot or not snapshot.values:
        return initial_state, None
    if snapshot.next:
        return None, None
    return None, dict(snapshot.values)


def _artifact_candidates(state: Mapping[str, Any] | None) -> list[Any]:
    """Collect candidate artifact paths from state; containment is validated later."""
    if not state:
        return []
    return [state.get("resume_path"), state.get("cover_letter_path")]


async def _attempt_cleanup(label: str, operation) -> BaseException | None:
    """Run one cleanup step, returning a bounded exception instead of raising.

    Cancellation raised by a cleanup await is recorded like any other cleanup
    failure so finalization always completes every step.
    """
    try:
        outcome = operation()
        if inspect.isawaitable(outcome):
            await outcome
        return None
    except (Exception, asyncio.CancelledError) as exc:
        log_event(
            "warning",
            f"cleanup.{label}_failed",
            f"{label} cleanup failed during run finalization.",
            node="main",
            exc=exc,
        )
        return exc


async def _finalize_run(
    *,
    run_id: str,
    dry_run: bool,
    started_at: datetime,
    state: Mapping[str, Any] | None,
    workflow_error: BaseException | None,
    interrupted: bool,
    cancelled: asyncio.CancelledError | None = None,
) -> None:
    """Centralized finalization: cleanup, single summary publication, shutdown.

    Cleanup attempts never mask the primary failure. The terminal summary is
    published exactly once, after terminal status and cleanup results are
    known. Logging is always shut down exactly once before anything is raised.

    Escape precedence:
      0. an original asyncio.CancelledError always wins and is re-raised;
      a. otherwise the original workflow exception wins;
      b. otherwise the first cleanup failure wins;
      c. otherwise a summary publication failure is raised, so a run can never
         silently succeed without its durable terminal summary.
    User-interrupted runs keep the existing non-raising KeyboardInterrupt
    behavior when finalization succeeds, but a failed durable publication is
    still propagated so callers cannot believe finalization succeeded.
    """
    llm_close_error = await _attempt_cleanup("llm_client", close_llm_client)
    mongo_close_error = await _attempt_cleanup("mongo", DeduplicationStore.close)
    cleanup_errors = [err for err in (llm_close_error, mongo_close_error) if err is not None]
    terminal_state = state

    if workflow_error is not None:
        terminal_status = TERMINAL_STATUS_FAILED
    elif interrupted:
        terminal_status = TERMINAL_STATUS_INTERRUPTED
    elif cleanup_errors:
        # A successful workflow whose required cleanup failed is not completed.
        terminal_status = TERMINAL_STATUS_FAILED
    elif terminal_state and terminal_state.get("account_safety_paused"):
        terminal_status = TERMINAL_STATUS_PAUSED
    else:
        terminal_status = TERMINAL_STATUS_COMPLETED

    # Single guarded publication attempt; its failure is captured, never
    # silently swallowed for successful runs.
    publication_error: BaseException | None = None
    try:
        summary = build_summary(
            run_id=run_id,
            terminal_status=terminal_status,
            dry_run=dry_run,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            state=terminal_state,
            artifacts=_artifact_candidates(terminal_state),
            failure=None if cancelled is not None else workflow_error,
            cleanup_errors=[f"cleanup {type(err).__name__}" for err in cleanup_errors],
        )
        written = write_summary(summary, run_id=run_id)
        log_event(
            "info",
            "summary.published",
            f"Run summary published ({terminal_status}).",
            run_id=run_id,
            node="main",
            status=terminal_status,
        )
        logging.getLogger("jobapply").debug(f"Summary written to {written}")
    except Exception as exc:
        publication_error = exc
        log_event(
            "error",
            "summary.publish_failed",
            "Run summary could not be published.",
            run_id=run_id,
            node="main",
            status=terminal_status,
            exc=exc,
        )

    # Logging is always shut down exactly once, before anything is raised.
    shutdown_logging()

    if cancelled is not None:
        # Cancellation semantics win over cleanup/publication failures.
        raise cancelled
    if workflow_error is not None:
        raise workflow_error
    if cleanup_errors and not interrupted:
        raise cleanup_errors[0]
    if publication_error is not None:
        raise publication_error


async def _read_checkpoint_state(graph, config) -> dict[str, Any] | None:
    """Best-effort read of the checkpointed graph state for interrupted/failed runs.

    Tolerates in-flight cancellation so finalization can always continue.
    """
    try:
        snapshot = await graph.aget_state(config)
        if snapshot and snapshot.values:
            return dict(snapshot.values)
    except (Exception, asyncio.CancelledError) as exc:
        log_event(
            "debug",
            "summary.checkpoint_read_failed",
            "Could not read checkpoint state for the terminal summary.",
            node="main",
            exc=exc,
        )
    return None


async def run(
    run_id: Optional[str] = None,
    dry_run: Optional[bool] = None,
    max_applications: Optional[int] = None,
    max_jobs: Optional[int] = None,
    location: Optional[str] = None,
    recency_days: Optional[int] = None,
):
    """Main entry point for the job application agent.

    Args:
        run_id: Reuse to resume a crashed run. None = fresh run (uuid4).
        dry_run: Override .env setting.
        max_applications: Override .env setting.
        location: Override configured LinkedIn search location.
        recency_days: Override configured search recency window in days.
    """
    settings = get_settings()
    effective_dry_run = dry_run if dry_run is not None else settings.dry_run
    resume_requested = run_id is not None

    # Validate explicitly supplied CLI values eagerly (these are user input and
    # must never wait for a checkpoint). Environment search defaults are only
    # resolved for FRESH runs: on a resumed run the checkpoint keys are
    # authoritative — including an explicitly stored None — so changed
    # environment defaults are deliberately left unresolved.
    from jobapply.utils.search_config import (
        DEFAULT_SEARCH_LOCATION,
        normalize_recency_days,
        normalize_search_location,
    )

    cli_location, cli_location_error = normalize_search_location(location)
    if cli_location_error:
        raise ValueError(f"Invalid search configuration: {cli_location_error}")
    cli_recency_days, cli_recency_error = normalize_recency_days(recency_days)
    if cli_recency_error:
        raise ValueError(f"Invalid search configuration: {cli_recency_error}")

    if resume_requested:
        # Placeholders only: LangGraph resumes from the checkpoint, which holds
        # the authoritative effective values; explicit CLI overrides (when
        # given) are durably applied to the checkpoint before execution.
        search_location = cli_location if cli_location is not None else DEFAULT_SEARCH_LOCATION
        search_recency_days = cli_recency_days
    else:
        search_location, search_recency_days, search_config_error = effective_search_params(
            location, recency_days, settings.search_location, settings.search_recency_days
        )
        if search_config_error:
            raise ValueError(f"Invalid search configuration: {search_config_error}")

    started_at = datetime.now(timezone.utc)
    if resume_requested:
        run_id = validate_run_id(run_id)
    else:
        run_id = str(uuid4())

    # Setup safe logging directory
    output_dir = get_run_output_dir(run_id)
    logger = setup_logging(run_id)
    logger.info(f"🚀 Starting job application agent (run_id: {run_id})")
    logger.info(f"📁 Logs: {output_dir}")

    # Everything operational lives inside the protected lifecycle so any
    # failure produces a failed summary and full cleanup without masking.
    workflow_error: BaseException | None = None
    cancelled_error: asyncio.CancelledError | None = None
    interrupted = False
    terminal_state: Mapping[str, Any] | None = None
    graph = None
    config: Optional[RunnableConfig] = None
    tracker = ProgressTracker(logger)

    try:
        # Mongo repository creation and daily-count lookup
        dedup = DeduplicationStore()
        daily_count = await dedup.get_daily_count()
        logger.info(f"📊 Application count today: {daily_count}")

        # Initialize state
        initial_state = {
            "search_queries": settings.search_queries_list,
            "search_location": search_location,
            "search_recency_days": search_recency_days,
            "current_query_index": 0,
            "current_page": 1,
            "pages_per_query": settings.pages_per_query,
            "job_listings": [],
            "current_job_index": 0,
            "current_job": None,
            "search_failed": False,
            "query_exhausted": False,
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
            "manual_review_queue_pending": [],
            "manual_review_queue_overflow": 0,
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
                else settings.effective_max_applications
            ),
            "daily_application_cap": (
                settings.daily_application_cap
                if settings.daily_application_cap is not None
                else (
                    max_applications
                    if max_applications is not None
                    else settings.effective_max_applications
                )
            ),
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

        # Compile graph and prepare configuration
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

        # Run graph with LangSmith tracing
        logger.info("▶️  Starting execution...\n")
        with tracing_v2_enabled(
            project_name=None,  # uses LANGSMITH_PROJECT from env
            tags=["jobapply-session"],
        ) as tracer:
            snapshot = None
            if resume_requested:
                snapshot = await graph.aget_state(config)
            checkpoint_is_real = bool(snapshot and snapshot.values)
            graph_input, saved_result = resolve_graph_input(
                resume_requested,
                snapshot,
                initial_state,
            )

            if resume_requested and not checkpoint_is_real:
                # A requested run ID with a missing/empty checkpoint behaves
                # exactly like a fresh run: resolve CLI > environment >
                # defaults now so the placeholder values chosen earlier never
                # reach the graph. An EXISTING checkpoint keeps its own keys
                # authoritative (including recency=None), and legacy
                # checkpoints missing individual keys fall back per-key inside
                # the search node.
                fresh_location, fresh_recency, fresh_error = effective_search_params(
                    location, recency_days, settings.search_location, settings.search_recency_days
                )
                if fresh_error:
                    raise ValueError(f"Invalid search configuration: {fresh_error}")
                if isinstance(graph_input, dict):
                    graph_input["search_location"] = fresh_location
                    graph_input["search_recency_days"] = fresh_recency

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

            # Explicit CLI search overrides must win even when LangGraph resumes
            # from saved state: durably apply them to the checkpoint before the
            # next search. Without an explicit override, checkpointed effective
            # values are kept untouched (even if environment defaults changed).
            if resume_requested and snapshot and snapshot.values:
                from jobapply.utils.manual_review import flush_manual_review_queue
                from jobapply.utils.search_config import resolve_state_search_params

                state_overrides: dict[str, Any] = {}
                # Checkpoint keys are authoritative here — including an
                # explicitly stored None recency.
                saved_location, saved_recency = resolve_state_search_params(
                    snapshot.values, settings
                )

                if cli_location is not None and cli_location != saved_location:
                    state_overrides["search_location"] = cli_location
                if recency_days is not None and cli_recency_days != saved_recency:
                    # ``--recency-days 0`` normalized to None: an explicit
                    # disable of a checkpointed filter is persisted as None.
                    state_overrides["search_recency_days"] = cli_recency_days

                # Central safe retry boundary for resumed execution: any
                # pending manual-review queue items flush before further work.
                pending_flush = await flush_manual_review_queue(snapshot.values)
                if "manual_review_queue_pending" in pending_flush and pending_flush[
                    "manual_review_queue_pending"
                ] != list(snapshot.values.get("manual_review_queue_pending") or []):
                    state_overrides.update(pending_flush)

                if state_overrides:
                    try:
                        await graph.aupdate_state(config, state_overrides)
                        log_event(
                            "info",
                            "run.resume_state_overridden",
                            "Explicit overrides applied to the resumed checkpoint.",
                            run_id=str(run_id),
                            node="main",
                            details={"keys": sorted(state_overrides.keys())},
                        )
                    except Exception as override_exc:
                        # Fail closed: never continue a resumed run on stale
                        # parameters when an explicit override could not be
                        # durably persisted. Finalization records the failure
                        # in a failed summary and re-raises.
                        log_event(
                            "error",
                            "run.resume_state_override_failed",
                            "Resumed-checkpoint override could not be persisted; aborting run.",
                            run_id=str(run_id),
                            node="main",
                            exc=override_exc,
                        )
                        raise RuntimeError(
                            "resume_override_not_persisted: explicit CLI search "
                            "overrides could not be applied to the checkpoint; "
                            f"refusing to continue with stale parameters (keys={sorted(state_overrides.keys())})"
                        ) from override_exc

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

            # Capture the authoritative terminal state; the summary itself is
            # published exactly once during finalization after cleanup results.
            terminal_state = result if isinstance(result, dict) else {}

            # Final central retry for pending manual-review queue items so the
            # durable summary reports the true remaining backlog.
            try:
                from jobapply.utils.manual_review import flush_manual_review_queue

                final_flush = await flush_manual_review_queue(terminal_state)
                if final_flush:
                    terminal_state = {**terminal_state, **final_flush}
            except Exception:
                pass  # Summary still reports the un-flushed pending backlog.

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
    except asyncio.CancelledError as cancelled:
        # Real Ctrl+C under asyncio.run arrives here as CancelledError
        # (BaseException). Classify as interrupted, run full finalization,
        # then re-raise the original cancellation after finalization.
        cancelled_error = cancelled
        interrupted = True
        terminal_state = await _read_checkpoint_state(graph, config)
        logger.warning("\n⚠️  Interrupted by user. State saved to checkpoint.")
        logger.info(f"   Resume with: jobapply --run-id {run_id}")
    except KeyboardInterrupt:
        # Preserve existing interrupted behavior: no re-raise. Cleanup problems
        # are recorded (and included in the summary) by finalization.
        interrupted = True
        terminal_state = await _read_checkpoint_state(graph, config)
        logger.warning("\n⚠️  Interrupted by user. State saved to checkpoint.")
        logger.info(f"   Resume with: jobapply --run-id {run_id}")
    except Exception as e:
        workflow_error = e
        terminal_state = await _read_checkpoint_state(graph, config)
        logger.error(f"❌ Error: {redact_string(str(e))}")
        logger.info(f"   Resume with: jobapply --run-id {run_id}")

    # Centralized finalization: cleanup attempts, single guarded summary
    # publication, logging shutdown last, then escape with the original
    # primary failure (cancellation first, or first cleanup failure when the
    # workflow succeeded).
    await _finalize_run(
        run_id=run_id,
        dry_run=effective_dry_run,
        started_at=started_at,
        state=terminal_state,
        workflow_error=workflow_error,
        interrupted=interrupted,
        cancelled=cancelled_error,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_review_parser() -> argparse.ArgumentParser:
    """Build the parser for `jobapply review` operator subcommands."""
    review = argparse.ArgumentParser(
        prog="jobapply review",
        description="Inspect and resolve the durable manual-review queue",
    )
    review_sub = review.add_subparsers(dest="review_command", required=True)

    list_parser = review_sub.add_parser("list", help="list bounded manual-review items")
    list_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=20,
        help="maximum number of items to list (bounded)",
    )
    list_parser.add_argument(
        "--state", choices=("open", "acknowledged", "resolved", "all"), default="all"
    )

    for name, help_text in (
        ("acknowledge", "atomically acknowledge an open item"),
        ("resolve", "atomically resolve an open/acknowledged item"),
    ):
        sub = review_sub.add_parser(name, help=help_text)
        sub.add_argument("item_id", help="manual-review queue item ID")
        sub.add_argument("--note", default=None, help="optional bounded operator note")

    resolve_parser = review_sub.choices["resolve"]
    resolve_parser.add_argument(
        "--label",
        dest="resolution_label",
        default=None,
        choices=(
            "confirmed_not_submitted",
            "confirmed_submitted",
            "requires_followup",
        ),
        help="explicit safe resolution label (required for submission-unknown items)",
    )
    return review


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
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Override the LinkedIn search location filter",
    )
    parser.add_argument(
        "--recency-days",
        type=int,
        default=None,
        help="Search recency window in days (1-30); 0 disables the filter",
    )

    # Operator commands are extracted before workflow parsing so every legacy
    # flag-only invocation keeps working exactly as before.
    argv = sys.argv[1:]
    command = None
    command_args: list[str] = []
    if argv and argv[0] in ("doctor", "review"):
        command = argv[0]
        command_args = argv[1:]

    if command == "doctor":
        doctor_parser = argparse.ArgumentParser(
            prog="jobapply doctor",
            description="Diagnose JobApply configuration; offline and non-mutating by default",
        )
        doctor_parser.add_argument(
            "--live-checks",
            action="store_true",
            help="opt in to bounded read-only connectivity checks (MongoDB ping, Telegram getMe, Gemma endpoint, Edge CDP)",
        )
        doctor_args = doctor_parser.parse_args(command_args)
        raise SystemExit(run_doctor_cli(live_checks=doctor_args.live_checks))

    if command == "review":
        review_args = _build_review_parser().parse_args(command_args)
        raise SystemExit(asyncio.run(run_review_cli(review_args)))

    args = parser.parse_args(argv)

    asyncio.run(
        run(
            run_id=args.run_id,
            dry_run=args.dry_run,
            max_applications=args.max_applications,
            max_jobs=args.max_jobs,
            location=args.location,
            recency_days=args.recency_days,
        )
    )


if __name__ == "__main__":
    main()
