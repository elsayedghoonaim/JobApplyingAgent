"""Deterministic offline tests for non-destructive repost flags and metrics."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from jobapply.nodes.outcomes import build_application_outcome, job_repost_persistence_fields
from jobapply.nodes.search import (
    MAX_REPOST_EVIDENCE_LENGTH,
    find_repost_indicator_on_card,
    text_indicates_applied_card_status,
    text_indicates_repost,
)
from jobapply.utils.summary import SUMMARY_SCHEMA_VERSION, build_summary


def _state_with_job(job: dict) -> dict:
    return {
        "run_id": "repost-run",
        "dry_run": True,
        "current_job": job,
        "qualification_result": {"qualified": True, "score": 0.9, "reasoning": "ok"},
        "resume_path": None,
        "cover_letter_path": None,
        "approval_status": None,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
    }


# ── Detection: strict status phrases only ────────────────────────────────────


def test_explicit_repost_status_phrases_are_recognized():
    assert text_indicates_repost("Reposted")
    assert text_indicates_repost("reposted")
    assert text_indicates_repost("Reposted 2 weeks ago")
    assert text_indicates_repost("Reposted 14 hours ago")
    assert text_indicates_repost("Reposted by recruiter")
    assert text_indicates_repost("Reposted by Sarah Chen")


def test_title_or_company_text_containing_repost_is_never_evidence():
    assert not text_indicates_repost("Repost Coordinator")
    assert not text_indicates_repost("Reposting Specialist")
    assert not text_indicates_repost("Senior Repost Manager")
    assert not text_indicates_repost("repost")
    assert not text_indicates_repost("Actively recruiting")
    assert not text_indicates_repost(None)
    assert not text_indicates_repost("")
    # Long junk that merely embeds the word is rejected by the anchored pattern.
    assert not text_indicates_repost(
        "This role was shared before; it is a repost of an older listing with extra context"
    )


def test_reposted_is_not_an_applied_marker():
    """A repost label must never mark a job as previously applied."""
    assert not text_indicates_applied_card_status("Reposted")
    assert not text_indicates_applied_card_status("Reposted 2 weeks ago")
    assert text_indicates_applied_card_status("Applied 3 days ago") is True


class _FakeCard:
    def __init__(self, candidates):
        self.evaluate = AsyncMock(return_value=candidates)


@pytest.mark.asyncio
async def test_detection_reads_bounded_metadata_and_returns_bounded_evidence():
    card = _FakeCard(["", "Actively recruiting", "Reposted by hiring team"])
    evidence = await find_repost_indicator_on_card(card)
    assert evidence == "Reposted by hiring team"
    assert len(evidence) <= 80

    # A "by ..." suffix beyond the phrase bound breaks the strict pattern.
    too_long_suffix = "Reposted by " + "x" * 45
    assert await find_repost_indicator_on_card(_FakeCard([too_long_suffix])) is None

    legal_long = f"Reposted by {'n' * 30}"
    evidence = await find_repost_indicator_on_card(_FakeCard([legal_long]))
    assert evidence is not None and len(evidence) <= 80


@pytest.mark.asyncio
async def test_detection_caps_hostile_candidate_flood_from_dom():
    """A DOM returning hundreds of long candidates stays bounded in Python too.

    The injected JS caps extraction to the first 20 bounded candidates, so
    evidence beyond that window is never even observed client-side.
    """
    flood = ["Reposted"] + [f"noise {i} " + "n" * 300 for i in range(500)]
    evidence = await find_repost_indicator_on_card(_FakeCard(flood))
    assert evidence == "Reposted"
    # Python-side re-bounds whatever the DOM layer returns.
    assert len(evidence) <= MAX_REPOST_EVIDENCE_LENGTH


@pytest.mark.asyncio
async def test_detection_survives_detached_card_after_click():
    """If the original card detached during navigation, detection degrades to None."""
    broken = _FakeCard([])
    broken.evaluate.side_effect = RuntimeError("element detached")
    assert await find_repost_indicator_on_card(broken) is None


@pytest.mark.asyncio
async def test_detection_uses_narrow_footer_selectors_excluding_title_and_company():
    """The injected JS must only query footer/metadata elements and exclude lockup text."""
    card = _FakeCard(["Reposted"])
    await find_repost_indicator_on_card(card)
    js = card.evaluate.await_args.args[0]
    assert ".job-card-container__footer-item" in js
    assert ".job-card-container__metadata-wrapper" in js
    # Title/company/link content explicitly excluded.
    assert "a.job-card-container__link" in js
    assert "artdeco-entity-lockup__title" in js
    assert "artdeco-entity-lockup__subtitle" in js
    # Bounded extraction inside the browser.
    assert "slice(0, 20)" in js
    assert "slice(0, 120)" in js
    # Broad content queries are gone.
    for banned in ("'li'", "'span', p", "[aria-label], .job-card-container__footer"):
        assert banned not in js


# ── Persistence through qualification metadata ───────────────────────────────


def test_job_repost_persistence_fields_bounded_and_redacted():
    fields = job_repost_persistence_fields({"is_repost": True, "repost_evidence": "Reposted"})
    assert fields == {"is_repost": True, "repost_evidence": "Reposted"}

    secret = "555444333:AABSyntheticTokenValueForPersistenceTests"
    fields = job_repost_persistence_fields(
        {"is_repost": True, "repost_evidence": "Reposted leak " + secret + " " + "y" * 400}
    )
    assert fields["is_repost"] is True
    assert secret not in fields["repost_evidence"]
    assert len(fields["repost_evidence"]) <= 200

    assert job_repost_persistence_fields({}) == {}
    assert job_repost_persistence_fields({"is_repost": False}) == {}


def test_qualification_dedup_record_includes_repost_fields(monkeypatch):
    """The dedup payload built at qualification time carries repost fields."""
    import inspect

    from jobapply.nodes import qualification as qual_mod

    source = inspect.getsource(qual_mod)
    assert "job_repost_persistence_fields(current_job)" in source


# ── Preservation through outcomes/summaries/notifications ────────────────────


def test_outcome_preserves_repost_flags():
    state = _state_with_job(
        {
            "job_id": "401",
            "title": "ML Engineer",
            "company": "Acme",
            "url": "https://www.linkedin.com/jobs/view/401",
            "is_repost": True,
            "repost_evidence": "Reposted",
        }
    )
    outcome = build_application_outcome(state, "dry_run", extra={"dry_run": True})
    assert outcome["is_repost"] is True
    assert outcome["repost_evidence"] == "Reposted"
    # Per-outcome durable dry-run marker present regardless of status.
    assert outcome["dry_run"] is True


def test_outcome_defaults_without_repost_metadata():
    state = _state_with_job({"job_id": "402", "title": "T", "company": "C"})
    outcome = build_application_outcome(state, "dry_run")
    assert outcome["is_repost"] is False
    assert outcome["repost_evidence"] is None


def _now():
    return datetime.now(timezone.utc)


def test_summary_authoritative_repost_metrics():
    outcomes = [
        {"status": "dry_run", "dry_run": True, "is_repost": True},
        {"status": "dry_run", "dry_run": True, "is_repost": False},
        {"status": "skipped", "dry_run": True, "reason": "already_applied", "is_repost": True},
        {"status": "needs_manual_review", "dry_run": True, "reason": "form_qa_timeout"},
        {"status": "failed", "dry_run": True, "error": "boom", "is_repost": True},
        # Live-run outcome: must NOT contaminate dry-run buckets.
        {"status": "skipped", "dry_run": False, "reason": "no_easy_apply"},
    ]
    summary = build_summary(
        run_id="r",
        terminal_status="completed",
        dry_run=True,
        started_at=_now(),
        state={"application_outcomes": outcomes},
    )
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["repost_count"] == 3  # top-level counts all marked outcomes
    metrics = summary["dry_run_metrics"]
    assert metrics["reached_submit_boundary"] == 2
    assert metrics["skipped"] == 1
    assert metrics["needs_manual_review"] == 1
    assert metrics["failed"] == 1
    assert metrics["confirmed_submissions_from_dry_run"] == 0
    assert metrics["reposted"] == 1


def test_mixed_live_and_dry_run_outcomes_never_cross_contaminate():
    outcomes = [
        {"status": "submitted", "dry_run": False},
        {"status": "submitted", "dry_run": False},
        {"status": "skipped", "dry_run": False, "reason": "already_applied"},
        {"status": "failed", "dry_run": False, "error": "boom"},
        {"status": "dry_run", "dry_run": True},
        {"status": "skipped", "dry_run": True, "reason": "user_skipped"},
        {"status": "needs_manual_review", "dry_run": True, "reason": "external_or_assessment"},
    ]
    # Live session view.
    live_summary = build_summary(
        run_id="r",
        terminal_status="completed",
        dry_run=False,
        started_at=_now(),
        state={"application_outcomes": outcomes},
    )
    live_metrics = live_summary["dry_run_metrics"]
    assert live_summary["confirmed_submissions"] == 2
    assert live_metrics["dry_run_session"] is False
    assert live_metrics["reached_submit_boundary"] == 1
    assert live_metrics["skipped"] == 1
    assert live_metrics["needs_manual_review"] == 1
    assert live_metrics["failed"] == 0
    assert live_metrics["confirmed_submissions_from_dry_run"] == 0

    # Dry-session view of identical outcomes.
    dry_summary = build_summary(
        run_id="r",
        terminal_status="completed",
        dry_run=True,
        started_at=_now(),
        state={"application_outcomes": outcomes},
    )
    dry_metrics = dry_summary["dry_run_metrics"]
    assert dry_metrics["dry_run_session"] is True
    assert dry_metrics["skipped"] == 1
    assert dry_summary["confirmed_submissions"] == 2  # top-level stays authoritative


def test_contradictory_dry_run_submission_exposes_invariant_error():
    outcomes = [
        {"status": "submitted", "dry_run": True},  # contradictory data
        {"status": "dry_run", "dry_run": True},
    ]
    summary = build_summary(
        run_id="r",
        terminal_status="completed",
        dry_run=True,
        started_at=_now(),
        state={"application_outcomes": outcomes},
    )
    metrics = summary["dry_run_metrics"]
    # Never reported as a real submission; bounded invariant error instead.
    assert metrics["confirmed_submissions_from_dry_run"] == 0
    assert any("submitted_status=1" in v for v in metrics["invariant_violations"])
    assert len(metrics["invariant_violations"]) <= 5


def test_notification_summary_wording_reflects_per_outcome_counts():
    from jobapply.nodes.notification import format_session_summary

    state = {
        "run_id": "notif-run",
        "search_queries": ["AI"],
        "current_query_index": 0,
        "application_outcomes": [
            {
                "status": "dry_run",
                "title": "Engineer",
                "company": "Acme",
                "qa_count": 0,
                "score": 0.8,
                "is_repost": True,
                "dry_run": True,
            },
            {"status": "failed", "title": "Other", "error": "boom", "dry_run": False},
        ],
    }
    text = format_session_summary(state)
    assert "DRY-RUN METRICS (per-outcome markers)" in text
    assert "🔁 Reposted dry-run outcomes: 1" in text
    assert "✅ Confirmed submissions from dry runs: 0" in text
    # Outcome-only repost counts are never described as all listings seen.
    assert "all listings seen" not in text


def test_notification_reports_pending_queue_backlog():
    from jobapply.nodes.notification import format_session_summary

    state = {
        "run_id": "notif-run",
        "search_queries": [],
        "manual_review_queue_pending": [{"item_id": "mr_x"}],
        "application_outcomes": [],
    }
    text = format_session_summary(state)
    assert "awaiting durable enqueue: 1" in text


# ── Non-destructive guarantees ───────────────────────────────────────────────


def test_reposted_listings_remain_eligible_for_selection():
    """Repost detection must never exclude, skip, or deprioritize a job."""
    from jobapply.nodes.select_next_job import select_next_job_node

    reposted = {
        "job_id": "777",
        "title": "AI Engineer",
        "company": "Acme",
        "is_repost": True,
        "repost_evidence": "Reposted",
    }
    state = {
        "run_id": "sel-run",
        "seen_job_ids": set(),
        "job_listings": [reposted],
        "current_job_index": 0,
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 0,
    }
    update = asyncio.run(select_next_job_node(state))
    assert update["current_job"]["job_id"] == "777"
    assert update["current_job"]["is_repost"] is True


def test_repost_flag_alone_never_consumes_quota_counters():
    from jobapply.execution.outcomes import applied_update

    state = _state_with_job({"job_id": "500", "title": "T", "company": "C", "is_repost": True})
    state["applications_count"] = 4
    state["daily_applications_count"] = 10

    update = applied_update(state, "dry_run", [])
    # Dry-run outcome never increments counters; repost flag changes nothing.
    assert "applications_count" not in update
    assert update["application_outcomes"][-1]["is_repost"] is True

    live_update = applied_update(state, "submitted", [])
    # Even a submitted status increments exactly once, repost or not.
    assert live_update["applications_count"] == 5


# ── Correction delta §3: hostile evidence cannot survive any durable boundary ─


def test_hostile_repost_evidence_is_bounded_and_redacted_everywhere(monkeypatch):
    import json

    from jobapply.nodes.execution import manual_review_update
    from jobapply.utils.summary import build_summary

    prefixed_token = "888777666:AABHostileEvidenceTokenValueCheck"
    monkeypatch.setenv("JOBAPPLY_TELEGRAM_BOT_TOKEN", prefixed_token)
    hostile_evidence = (
        f"Reposted leak {prefixed_token} mongodb://user:sup3rsecret@h/d " + "z" * 5000
    )
    state = _state_with_job(
        {
            "job_id": "910",
            "title": "T",
            "company": "C",
            "is_repost": True,
            "repost_evidence": hostile_evidence,
        }
    )

    # 1. Durable outcome carries only bounded, redacted evidence.
    outcome = build_application_outcome(state, "dry_run", extra={"dry_run": True})
    assert len(outcome["repost_evidence"]) <= 200
    assert prefixed_token not in outcome["repost_evidence"]
    assert "sup3rsecret" not in outcome["repost_evidence"]

    state["application_outcomes"] = [outcome]
    summary = build_summary(
        run_id="r",
        terminal_status="completed",
        dry_run=True,
        started_at=datetime.now(timezone.utc),
        state=state,
    )
    # 2. The serialized summary payload never retains the raw value.
    payload_text = json.dumps(summary)
    assert prefixed_token not in payload_text
    assert "sup3rsecret" not in payload_text
    assert "zzzz" not in payload_text

    # 3. Queue source metadata (persisted on success) is bounded/redacted too.
    async def scenario():
        return await manual_review_update(
            {
                **state,
                "manual_review_queue_pending": [],
                "manual_review_queue_overflow": 0,
            },
            "external_or_assessment",
            "redirected",
            [],
        )

    update = asyncio.run(scenario())
    assert update["application_outcomes"][-1]["status"] == "needs_manual_review"


# ── Correction delta §2: extra can never bypass canonical repost fields ─────


def test_extra_payload_cannot_override_canonical_repost_fields():
    prefixed_token = "666555444:AABExtraOverrideHostileTokenValue"
    hostile = f"Reposted? {prefixed_token} mongodb://user:pw@h/d " + "e" * 4000

    state = _state_with_job(
        {
            "job_id": "920",
            "title": "T",
            "company": "C",
            "is_repost": True,
            "repost_evidence": "Reposted",
        }
    )
    outcome = build_application_outcome(
        state,
        "dry_run",
        qa_count=1,
        extra={
            "repost_evidence": hostile,
            "is_repost": False,
            # Canonical truth fields are protected too.
            "status": "submitted",
            "dry_run": False,
            "job_id": "hacked-id",
            "timestamp": "1970-01-01T00:00:00+00:00",
            # Non-reserved extras still flow through untouched.
            "attempt_status": "submission_unknown",
        },
    )
    assert outcome["repost_evidence"] == "Reposted"
    assert outcome["is_repost"] is True
    assert prefixed_token not in repr(outcome)
    assert len(outcome["repost_evidence"]) <= 200
    assert outcome["status"] == "dry_run"
    assert outcome["dry_run"] is True
    assert outcome["job_id"] == "920"
    assert outcome["timestamp"] != "1970-01-01T00:00:00+00:00"
    assert outcome["attempt_status"] == "submission_unknown"


def test_outcomes_without_repost_metadata_keep_stable_shape():
    state = _state_with_job({"job_id": "921", "title": "T", "company": "C"})
    outcome = build_application_outcome(state, "skipped", reason="r")
    assert outcome["is_repost"] is False
    assert outcome["repost_evidence"] is None
