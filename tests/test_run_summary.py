"""Offline deterministic tests for run summaries and terminal-path publication."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobapply.main import run
from jobapply.utils.summary import (
    SUMMARY_SCHEMA_VERSION,
    build_summary,
    summary_path,
    write_summary,
)

RUN_ID = "summary-run"


def _started(minutes_ago: int = 5) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)


def _outcome(status: str, **extra) -> dict:
    outcome = {"status": status, "title": "Engineer", "company": "Acme"}
    outcome.update(extra)
    return outcome


def _completed_state() -> dict:
    return {
        "run_id": RUN_ID,
        "dry_run": False,
        "jobs_evaluated_count": 8,
        "qualified_jobs_count": 5,
        "not_qualified_jobs_count": 3,
        "application_outcomes": [
            _outcome("submitted", job_id="j1"),
            _outcome("submitted", job_id="j2"),
            _outcome("dry_run", job_id="j3"),
            _outcome("skipped", reason="already_applied"),
            _outcome("skipped", reason="already_applied"),
            _outcome("skipped", reason="form_qa_timeout"),
            _outcome(
                "needs_manual_review",
                attempt_status="submission_unknown",
                ambiguous_submission=True,
            ),
            _outcome("needs_manual_review", reason="external_or_assessment"),
            _outcome("failed", error="Execution failed"),
        ],
        "errors": ["first error", "second error"],
        "account_safety_paused": False,
    }


def test_summary_path_is_contained_under_requested_run_directory(tmp_path):
    path = summary_path(RUN_ID, base_dir=tmp_path)
    root = tmp_path.resolve()
    assert path == root / RUN_ID / "summary.json"
    assert path.relative_to(root)

    hostile = summary_path("../../evil", base_dir=tmp_path)
    assert hostile.relative_to(root)


def test_summary_publication_is_atomic_and_preserves_older_valid_file_on_failure(
    tmp_path, monkeypatch
):
    first = build_summary(
        run_id=RUN_ID,
        terminal_status="completed",
        dry_run=True,
        started_at=_started(),
        state={"application_outcomes": [_outcome("dry_run")]},
    )
    written = write_summary(first, run_id=RUN_ID, base_dir=tmp_path)
    original_payload = written.read_text(encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("jobapply.utils.summary.os.replace", failing_replace)
    second = dict(first, terminal_status="failed")
    with pytest.raises(OSError):
        write_summary(second, run_id=RUN_ID, base_dir=tmp_path)

    # The older valid summary is intact and no temporary siblings remain.
    assert json.loads(written.read_text(encoding="utf-8"))["terminal_status"] == "completed"
    assert written.read_text(encoding="utf-8") == original_payload
    run_dir = tmp_path / RUN_ID
    assert not list(run_dir.glob("*.tmp"))
    assert not list(run_dir.glob(".summary*"))


def test_completed_summary_counts_are_truthful_and_field_ordered():
    started = _started(2)
    finished = started + timedelta(seconds=90)
    summary = build_summary(
        run_id=RUN_ID,
        terminal_status="completed",
        dry_run=False,
        started_at=started,
        finished_at=finished,
        state=_completed_state(),
    )

    assert list(summary.keys()) == [
        "schema_version",
        "run_id",
        "terminal_status",
        "dry_run",
        "started_at",
        "finished_at",
        "duration_seconds",
        "jobs_evaluated",
        "jobs_qualified",
        "jobs_not_qualified",
        "confirmed_submissions",
        "dry_run_ready",
        "skipped_count",
        "skipped_by_reason",
        "needs_manual_review_count",
        "submission_unknown_count",
        "failed_count",
        "repost_count",
        "manual_review_queue_pending_count",
        "manual_review_queue_overflow_count",
        "dry_run_metrics",
        "error_count",
        "errors",
        "account_safety_pause",
        "artifacts",
    ]
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["terminal_status"] == "completed"
    assert summary["duration_seconds"] == 90.0
    assert summary["jobs_evaluated"] == 8
    assert summary["jobs_qualified"] == 5
    assert summary["jobs_not_qualified"] == 3
    assert summary["confirmed_submissions"] == 2
    assert summary["dry_run_ready"] == 1
    assert summary["skipped_count"] == 3
    assert summary["skipped_by_reason"] == {"already_applied": 2, "form_qa_timeout": 1}
    assert summary["needs_manual_review_count"] == 2
    assert summary["submission_unknown_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["error_count"] == 2
    assert summary["errors"] == ["first error", "second error"]
    assert summary["account_safety_pause"] is None


def test_dry_run_jobs_are_never_reported_as_submissions():
    state = {
        "dry_run": True,
        "applications_count": 0,
        "application_outcomes": [
            _outcome("dry_run"),
            _outcome("dry_run"),
            _outcome("skipped", reason="no_easy_apply"),
        ],
    }
    summary = build_summary(
        run_id=RUN_ID,
        terminal_status="completed",
        dry_run=True,
        started_at=_started(),
        state=state,
    )
    assert summary["confirmed_submissions"] == 0
    assert summary["dry_run_ready"] == 2
    assert summary["skipped_count"] == 1


def test_skipped_manual_review_and_submission_unknown_counts_remain_distinct():
    state = {
        "application_outcomes": [
            _outcome("skipped", reason="user_skipped"),
            _outcome("needs_manual_review", reason="unresolved_required_fields"),
            _outcome(
                "needs_manual_review",
                attempt_status="submission_unknown",
                ambiguous_submission=True,
            ),
            _outcome("failed", error="boom"),
        ],
    }
    summary = build_summary(
        run_id=RUN_ID,
        terminal_status="completed",
        dry_run=False,
        started_at=_started(),
        state=state,
    )
    assert summary["skipped_count"] == 1
    assert summary["needs_manual_review_count"] == 2
    assert summary["submission_unknown_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["confirmed_submissions"] == 0


def test_paused_summary_includes_bounded_redacted_safety_evidence(tmp_path):
    secret_reason = "Checkpoint challenge token 123456789:AAFakeSyntheticTelegramTokenXYZ123 leaked"
    state = {
        "account_safety_paused": True,
        "account_safety_barrier_type": "checkpoint",
        "account_safety_stage": "execution_pre_submit",
        "account_safety_reason": secret_reason,
        "account_safety_url": "https://www.linkedin.com/checkpoint/challenge/9f3ab2c9d8e7f6a5b4c3/",
        "account_safety_detected_at": "2026-08-24T10:00:00+00:00",
        "account_safety_evidence": "Detected via URL at checkpoint challenge",
        "account_safety_resume_instructions": "Resolve manually then start a NEW session.",
        "application_outcomes": [],
        "errors": [secret_reason],
    }
    summary = build_summary(
        run_id=RUN_ID,
        terminal_status="paused",
        dry_run=False,
        started_at=_started(),
        state=state,
    )
    pause = summary["account_safety_pause"]
    assert pause is not None
    assert pause["barrier_type"] == "CHECKPOINT"
    assert pause["stage"] == "execution_pre_submit"
    assert pause["detected_at"] == "2026-08-24T10:00:00+00:00"
    assert pause["evidence"].startswith("Detected via URL")
    assert "linkedin.com" in pause["url"]

    published = write_summary(summary, run_id=RUN_ID, base_dir=tmp_path)
    raw = published.read_text(encoding="utf-8")
    assert "AAFakeSyntheticTelegramToken" not in raw
    payload = json.loads(raw)
    assert payload["terminal_status"] == "paused"
    assert len(payload["errors"][0]) <= 200


def test_failed_summary_records_redacted_bounded_primary_failure():
    failure = RuntimeError(f"connection dropped password=super-secret-db-pass {'x' * 500}")
    summary = build_summary(
        run_id=RUN_ID,
        terminal_status="failed",
        dry_run=False,
        started_at=_started(),
        state={"errors": [], "application_outcomes": []},
        failure=failure,
    )
    assert summary["terminal_status"] == "failed"
    assert summary["error_count"] == 1
    (error,) = summary["errors"]
    assert error.startswith("RuntimeError:")
    assert "super-secret-db-pass" not in error
    assert len(error) <= 200


def test_interrupted_summary_reflects_partial_checkpoint_state():
    state = {
        "jobs_evaluated_count": 4,
        "qualified_jobs_count": 2,
        "not_qualified_jobs_count": 2,
        "application_outcomes": [_outcome("submitted")],
        "errors": [],
    }
    summary = build_summary(
        run_id=RUN_ID,
        terminal_status="interrupted",
        dry_run=False,
        started_at=_started(),
        state=state,
    )
    assert summary["terminal_status"] == "interrupted"
    assert summary["confirmed_submissions"] == 1
    assert summary["jobs_evaluated"] == 4
    assert summary["duration_seconds"] >= 0.0


def test_invalid_terminal_status_is_rejected():
    with pytest.raises(ValueError):
        build_summary(
            run_id=RUN_ID,
            terminal_status="mostly-fine",
            dry_run=True,
            started_at=_started(),
            state={},
        )


def test_artifact_references_require_containment_under_run_directory(tmp_path):
    inside = tmp_path / RUN_ID / "cover_letter_job.txt"
    inside.parent.mkdir(parents=True)
    inside.write_text("letter", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    summary = build_summary(
        run_id=RUN_ID,
        terminal_status="completed",
        dry_run=True,
        started_at=_started(),
        state={"resume_path": str(outside), "cover_letter_path": str(inside)},
        base_dir=tmp_path,
    )
    assert summary["artifacts"] == ["cover_letter_job.txt"]


def test_resuming_same_run_id_replaces_the_same_summary_file(tmp_path):
    first = build_summary(
        run_id=RUN_ID,
        terminal_status="interrupted",
        dry_run=True,
        started_at=_started(),
        state={"application_outcomes": []},
    )
    destination = write_summary(first, run_id=RUN_ID, base_dir=tmp_path)

    second = build_summary(
        run_id=RUN_ID,
        terminal_status="completed",
        dry_run=True,
        started_at=_started(),
        state={"application_outcomes": [_outcome("dry_run")]},
    )
    replaced = write_summary(second, run_id=RUN_ID, base_dir=tmp_path)

    assert replaced == destination
    summaries = list((tmp_path / RUN_ID).glob("*summary*.json"))
    assert summaries == [destination]
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["terminal_status"] == "completed"
    assert payload["dry_run_ready"] == 1


def _mock_dedup():
    store = AsyncMock()
    store.get_daily_count.return_value = 0
    store_cls = MagicMock(return_value=store)
    store_cls.close = AsyncMock()
    return patch("jobapply.main.DeduplicationStore", store_cls), store


def _checkpoint_mock(values=None):
    snapshot = MagicMock()
    snapshot.values = values
    return snapshot


@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_main_success_writes_completed_summary(
    mock_compile_graph, _mock_close, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mock_dedup_cls, _store = _mock_dedup()
    final_state = {
        "account_safety_paused": False,
        "jobs_evaluated_count": 1,
        "qualified_jobs_count": 1,
        "not_qualified_jobs_count": 0,
        "applications_count": 0,
        "application_outcomes": [_outcome("dry_run")],
        "errors": [],
        "resume_path": None,
        "cover_letter_path": None,
    }
    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock({"account_safety_paused": False})
    graph.ainvoke.return_value = final_state
    mock_compile_graph.return_value = graph

    with mock_dedup_cls:
        await run(run_id="summary-int-completed", dry_run=True)

    summary_file = tmp_path / "outputs" / "summary-int-completed" / "summary.json"
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert payload["terminal_status"] == "completed"
    assert payload["dry_run_ready"] == 1
    assert payload["confirmed_submissions"] == 0


@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_main_pause_writes_paused_summary(
    mock_compile_graph, _mock_close, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mock_dedup_cls, _store = _mock_dedup()
    final_state = {
        "account_safety_paused": True,
        "account_safety_barrier_type": "captcha",
        "account_safety_stage": "search_navigation",
        "account_safety_reason": "CAPTCHA challenge detected",
        "account_safety_url": "https://www.linkedin.com/checkpoint/challenge/captcha",
        "account_safety_detected_at": None,
        "account_safety_evidence": None,
        "account_safety_resume_instructions": None,
        "jobs_evaluated_count": 0,
        "application_outcomes": [],
        "errors": [],
    }
    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock({"account_safety_paused": False})
    graph.ainvoke.return_value = final_state
    mock_compile_graph.return_value = graph

    with mock_dedup_cls:
        await run(run_id="summary-int-paused", dry_run=True)

    summary_file = tmp_path / "outputs" / "summary-int-paused" / "summary.json"
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert payload["terminal_status"] == "paused"
    assert payload["account_safety_pause"]["barrier_type"] == "CAPTCHA"


@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_main_keyboard_interrupt_writes_interrupted_summary(
    mock_compile_graph, _mock_close, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mock_dedup_cls, _store = _mock_dedup()
    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(None)
    graph.ainvoke.side_effect = KeyboardInterrupt()
    mock_compile_graph.return_value = graph

    with mock_dedup_cls:
        await run(run_id="summary-int-interrupted", dry_run=True)

    summary_file = tmp_path / "outputs" / "summary-int-interrupted" / "summary.json"
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert payload["terminal_status"] == "interrupted"


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_main_failure_preserves_primary_error_when_summary_write_fails(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mock_dedup_cls, _store = _mock_dedup()
    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(None)
    graph.ainvoke.side_effect = RuntimeError("primary workflow explosion")
    mock_compile_graph.return_value = graph
    mock_write_summary.side_effect = OSError("disk full while writing summary")

    with mock_dedup_cls:
        with pytest.raises(RuntimeError, match="primary workflow explosion"):
            await run(run_id="summary-int-failed", dry_run=True)

    # Exactly one publication attempt; its failure did not mask the workflow.
    assert mock_write_summary.call_count == 1
    # Logging shutdown still ran exactly once before the primary error escaped.
    mock_shutdown.assert_called_once()


@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_main_compile_failure_writes_failed_summary_and_reraises(
    mock_compile_graph, _mock_close, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    mock_dedup_cls, _store = _mock_dedup()
    mock_compile_graph.side_effect = RuntimeError("graph construction failed")

    with mock_dedup_cls:
        with pytest.raises(RuntimeError, match="graph construction failed"):
            await run(run_id="summary-int-compile", dry_run=True)

    summary_file = tmp_path / "outputs" / "summary-int-compile" / "summary.json"
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert payload["terminal_status"] == "failed"
    assert any("RuntimeError: graph construction failed" in err for err in payload["errors"])


def test_hostile_run_id_payload_run_id_matches_safe_directory_name(tmp_path):
    hostile = "../../evil/run"
    summary = build_summary(
        run_id=hostile,
        terminal_status="completed",
        dry_run=True,
        started_at=_started(),
        state={},
    )
    destination = write_summary(summary, run_id=hostile, base_dir=tmp_path)

    root = tmp_path.resolve()
    # No traversal: the published file stays inside the outputs root.
    assert destination.relative_to(root)
    parent_name = destination.parent.name
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert "/" not in payload["run_id"]
    assert "\\" not in payload["run_id"]
    assert ".." not in payload["run_id"]
    assert payload["run_id"] == parent_name


@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.compile_graph")
async def test_get_daily_count_failure_writes_failed_summary_and_still_closes_everything(
    mock_compile_graph, mock_shutdown, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    failing_store = MagicMock()
    failing_store.get_daily_count = AsyncMock(side_effect=RuntimeError("mongo count down"))
    store_cls.return_value = failing_store
    store_cls.close = AsyncMock()

    with (
        patch("jobapply.main.DeduplicationStore", store_cls),
        patch("jobapply.main.close_llm_client", new_callable=AsyncMock) as mock_close_llm,
    ):
        with pytest.raises(RuntimeError, match="mongo count down"):
            await run(run_id="summary-final-dailycount", dry_run=True)

    # Graph compilation was never reached.
    mock_compile_graph.assert_not_called()
    # Every cleanup step was attempted despite the primary failure.
    mock_close_llm.assert_awaited_once()
    store_cls.close.assert_awaited_once()
    mock_shutdown.assert_called_once()

    summary_file = tmp_path / "outputs" / "summary-final-dailycount" / "summary.json"
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert payload["terminal_status"] == "failed"
    assert any("RuntimeError: mongo count down" in err for err in payload["errors"])


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_workflow_failure_with_both_cleanup_failures_preserves_workflow_error(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock(side_effect=RuntimeError("mongo close exploded"))
    mock_close_llm.side_effect = RuntimeError("llm close exploded")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(None)
    graph.ainvoke.side_effect = RuntimeError("primary workflow failure")
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with pytest.raises(RuntimeError, match="primary workflow failure"):
            await run(run_id="summary-final-double-cleanup", dry_run=True)

    # Every cleanup was attempted even though the workflow already failed.
    mock_close_llm.assert_awaited_once()
    store_cls.close.assert_awaited_once()
    # The single terminal publication happened after cleanup results were known.
    mock_write_summary.assert_called_once()
    # Logging shutdown ran exactly once despite the escaping primary failure.
    mock_shutdown.assert_called_once()
    published = mock_write_summary.call_args[0][0]
    assert published["terminal_status"] == "failed"
    error_blob = json.dumps(published["errors"])
    assert "RuntimeError: primary workflow failure" in error_blob
    assert "cleanup RuntimeError" in error_blob


@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_successful_workflow_with_cleanup_failure_publishes_failed_summary(
    mock_compile_graph, mock_close_llm, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()
    mock_close_llm.side_effect = RuntimeError("llm close exploded after success")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock({"account_safety_paused": False})
    graph.ainvoke.return_value = {
        "account_safety_paused": False,
        "jobs_evaluated_count": 1,
        "application_outcomes": [_outcome("dry_run")],
        "errors": [],
    }
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with pytest.raises(RuntimeError, match="llm close exploded after success"):
            await run(run_id="summary-final-success-cleanup", dry_run=True)

    summary_file = tmp_path / "outputs" / "summary-final-success-cleanup" / "summary.json"
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    # A successful workflow whose required cleanup failed is truthfully failed,
    # while dry-run outcomes are still never counted as submissions.
    assert payload["terminal_status"] == "failed"
    assert payload["confirmed_submissions"] == 0
    assert payload["dry_run_ready"] == 1
    assert any("cleanup RuntimeError" in err for err in payload["errors"])


@patch("jobapply.main.write_summary")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_summary_write_failure_never_masks_workflow_failure_and_publishes_once(
    mock_compile_graph, mock_close_llm, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()
    mock_write_summary.side_effect = OSError("disk full while writing summary")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(None)
    graph.ainvoke.side_effect = RuntimeError("primary workflow explosion")
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with pytest.raises(RuntimeError, match="primary workflow explosion"):
            await run(run_id="summary-final-write-fail", dry_run=True)

    # Exactly one publication attempt; its failure did not mask the workflow.
    assert mock_write_summary.call_count == 1


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_logger_shutdown_runs_exactly_once_on_success(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock({"account_safety_paused": False})
    graph.ainvoke.return_value = {
        "account_safety_paused": False,
        "jobs_evaluated_count": 0,
        "application_outcomes": [],
        "errors": [],
    }
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        await run(run_id="summary-final-shutdown", dry_run=True)

    mock_shutdown.assert_called_once()
    mock_write_summary.assert_called_once()


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_interrupted_run_records_cleanup_problems_without_reraising_them(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock(side_effect=RuntimeError("mongo close during interrupt"))
    mock_close_llm.side_effect = RuntimeError("llm close during interrupt")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(None)
    graph.ainvoke.side_effect = KeyboardInterrupt()
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        # Existing KeyboardInterrupt behavior: nothing escapes the interrupted path.
        await run(run_id="summary-final-interrupt", dry_run=True)

    mock_close_llm.assert_awaited_once()
    store_cls.close.assert_awaited_once()
    mock_shutdown.assert_called_once()
    mock_write_summary.assert_called_once()
    published = mock_write_summary.call_args[0][0]
    assert published["terminal_status"] == "interrupted"
    assert any("cleanup RuntimeError" in err for err in published["errors"])


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_successful_run_with_summary_write_failure_raises_publication_error(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()
    mock_write_summary.side_effect = OSError("summary disk exploded")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock({"account_safety_paused": False})
    graph.ainvoke.return_value = {
        "account_safety_paused": False,
        "jobs_evaluated_count": 0,
        "application_outcomes": [],
        "errors": [],
    }
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        # A successful run can never silently lack its durable summary.
        with pytest.raises(OSError, match="summary disk exploded"):
            await run(run_id="summary-final-success-writefail", dry_run=True)

    mock_write_summary.assert_called_once()
    mock_shutdown.assert_called_once()


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_paused_run_with_summary_write_failure_raises_publication_error(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()
    mock_write_summary.side_effect = OSError("paused summary write failed")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock({"account_safety_paused": False})
    graph.ainvoke.return_value = {
        "account_safety_paused": True,
        "account_safety_barrier_type": "captcha",
        "account_safety_stage": "search_navigation",
        "jobs_evaluated_count": 0,
        "application_outcomes": [],
        "errors": [],
    }
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with pytest.raises(OSError, match="paused summary write failed"):
            await run(run_id="summary-final-paused-writefail", dry_run=True)

    mock_write_summary.assert_called_once()
    mock_shutdown.assert_called_once()


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_cleanup_failure_beats_summary_write_failure(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()
    mock_close_llm.side_effect = RuntimeError("llm cleanup boom")
    mock_write_summary.side_effect = OSError("summary disk exploded too")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock({"account_safety_paused": False})
    graph.ainvoke.return_value = {
        "account_safety_paused": False,
        "jobs_evaluated_count": 0,
        "application_outcomes": [],
        "errors": [],
    }
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        # Precedence: first cleanup failure wins over publication failure.
        with pytest.raises(RuntimeError, match="llm cleanup boom"):
            await run(run_id="summary-final-cleanup-vs-write", dry_run=True)

    mock_close_llm.assert_awaited_once()
    mock_write_summary.assert_called_once()
    mock_shutdown.assert_called_once()

    published = mock_write_summary.call_args[0][0]
    assert published["terminal_status"] == "failed"


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_async_cancellation_writes_interrupted_summary_and_reraises(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()

    partial_checkpoint = {
        "jobs_evaluated_count": 3,
        "qualified_jobs_count": 1,
        "not_qualified_jobs_count": 2,
        "application_outcomes": [_outcome("submitted")],
        "errors": [],
    }
    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(partial_checkpoint)
    graph.ainvoke.side_effect = asyncio.CancelledError()
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with pytest.raises(asyncio.CancelledError):
            await run(run_id="summary-final-cancelled", dry_run=True)

    # Every finalization step ran despite the cancellation.
    mock_close_llm.assert_awaited_once()
    store_cls.close.assert_awaited_once()
    assert mock_write_summary.call_count == 1
    assert mock_shutdown.call_count == 1

    published = mock_write_summary.call_args[0][0]
    # Cancellation is classified as interrupted and uses the partial checkpoint.
    assert published["terminal_status"] == "interrupted"
    assert published["jobs_evaluated"] == 3
    assert published["confirmed_submissions"] == 1


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_controller_run_cancellation_keeps_process_resources_open(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock()
    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(None)
    graph.ainvoke.side_effect = asyncio.CancelledError()
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with pytest.raises(asyncio.CancelledError):
            await run(
                run_id="summary-controller-cancelled",
                dry_run=True,
                close_resources=False,
            )

    mock_close_llm.assert_not_awaited()
    store_cls.close.assert_not_awaited()
    mock_write_summary.assert_called_once()
    mock_shutdown.assert_called_once()


@patch("jobapply.main.write_summary")
@patch("jobapply.main.shutdown_logging")
@patch("jobapply.main.close_llm_client", new_callable=AsyncMock)
@patch("jobapply.main.compile_graph")
async def test_cancellation_takes_precedence_over_cleanup_and_publication_failures(
    mock_compile_graph, mock_close_llm, mock_shutdown, mock_write_summary, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    store_cls = MagicMock()
    store_instance = MagicMock()
    store_instance.get_daily_count = AsyncMock(return_value=0)
    store_cls.return_value = store_instance
    store_cls.close = AsyncMock(side_effect=RuntimeError("mongo cancel cleanup"))
    mock_close_llm.side_effect = RuntimeError("llm cancel cleanup")
    mock_write_summary.side_effect = OSError("summary write during cancel")

    graph = AsyncMock()
    graph.aget_state.return_value = _checkpoint_mock(None)
    graph.ainvoke.side_effect = asyncio.CancelledError()
    mock_compile_graph.return_value = graph

    with patch("jobapply.main.DeduplicationStore", store_cls):
        # The original cancellation remains the escaping primary BaseException.
        with pytest.raises(asyncio.CancelledError):
            await run(run_id="summary-final-cancel-chaos", dry_run=True)

    # All finalization steps were still attempted exactly once each.
    mock_close_llm.assert_awaited_once()
    store_cls.close.assert_awaited_once()
    assert mock_write_summary.call_count == 1
    assert mock_shutdown.call_count == 1

    published = mock_write_summary.call_args[0][0]
    assert published["terminal_status"] == "interrupted"
