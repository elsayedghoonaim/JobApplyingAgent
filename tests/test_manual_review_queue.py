"""Deterministic offline tests for the durable manual-review queue and review CLI.

Covers idempotent enqueue across resume/replay, concurrent CAS transitions,
storage-failure degradation, submission-unknown safety, bounded redacted
listing, and distinct CLI outcomes for no-op/conflict/not-found.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from test_telegram_recovery_and_outbox import InMemoryTelegramCollection

from jobapply.models.manual_review import ManualReviewState, ResolutionLabel
from jobapply.nodes.execution import manual_review_update
from jobapply.settings import get_settings
from jobapply.utils.manual_review import (
    ManualReviewRepository,
    build_manual_review_item,
    make_manual_review_idempotency_key,
)
from jobapply.utils.mongo import MongoClientManager


@pytest.fixture(autouse=True)
async def manual_review_mongo():
    """Route every repository through one shared in-memory 'deployment'."""
    await MongoClientManager.close()
    queue = InMemoryTelegramCollection(key_field="item_id")
    mock_db = MagicMock()
    mock_motor = MagicMock()
    mock_motor.__getitem__.return_value = mock_db
    mock_db.__getitem__.side_effect = lambda name: queue
    try:
        with patch("jobapply.utils.mongo.AsyncIOMotorClient", return_value=mock_motor):
            yield queue
    finally:
        await MongoClientManager.close()


def _repo():
    return ManualReviewRepository()


def _item(run_id="run-1", job_id="401", category="form_qa_timeout", detail=None, **kwargs):
    return build_manual_review_item(
        run_id=run_id,
        job_id=job_id,
        title="ML Engineer",
        company="Acme",
        url="https://www.linkedin.com/jobs/view/401",
        reason_category=category,
        reason_detail=detail,
        **kwargs,
    )


# ── Idempotency and resume/replay ────────────────────────────────────────────


async def test_enqueue_is_idempotent_on_same_key(manual_review_mongo):
    repo = _repo()
    first = await repo.enqueue(_item())
    assert first.queued is True
    second = await repo.enqueue(_item())
    assert second.queued is False
    assert second.already_exists is True
    docs = await repo.list_items(limit=50)
    assert len(docs) == 1


async def test_replayed_item_after_repository_restart_creates_no_duplicate(manual_review_mongo):
    """Simulates process restart: fresh repository instance, same durable key."""
    first_repo = _repo()
    item = _item(detail="External assessment required")
    res1 = await first_repo.enqueue(item)

    # Reset lifecycle state like a restart would, then rebuild from scratch.
    ManualReviewRepository._reset_state()
    second_repo = _repo()
    rebuilt = _item(detail="External assessment required")  # deterministically same key
    res2 = await second_repo.enqueue(rebuilt)

    assert res1.queued is True
    assert res2.already_exists is True
    assert (
        res1.item_id
        == res2.item_id
        == make_manual_review_idempotency_key(
            "run-1", "401", "form_qa_timeout", "External assessment required"
        )
    )
    assert await second_repo.get_item(res2.item_id) is not None
    assert len(await second_repo.list_items(limit=50)) == 1


def test_configured_collection_name_is_used():
    assert get_settings().manual_review_collection == "manual_review_queue"


def test_key_is_stable_across_processes():
    a = make_manual_review_idempotency_key("r", "j", "category", "detail")
    b = make_manual_review_idempotency_key("r", "j", "category", "detail")
    assert a == b
    assert a.startswith("mr_") and len(a) <= 40
    c = make_manual_review_idempotency_key("r", "j", "category", "different")
    assert c != a


# ── Concurrent CAS transitions ───────────────────────────────────────────────


async def test_concurrent_acknowledge_allows_exactly_one_transition(manual_review_mongo):
    repo = _repo()
    item = _item()
    await repo.enqueue(item)

    results = await asyncio.gather(
        repo.acknowledge(item.item_id, note="first"),
        repo.acknowledge(item.item_id, note="second"),
    )
    changed = [r for r in results if r.changed]
    noop = [r for r in results if not r.changed]
    assert len(changed) == 1
    assert len(noop) == 1
    current = await repo.get_item(item.item_id)
    assert current.state == ManualReviewState.ACKNOWLEDGED
    assert any(r.reason == "item_already_acknowledged" for r in noop)


async def test_acknowledge_then_resolve_succeeds_and_double_resolve_is_noop(manual_review_mongo):
    repo = _repo()
    item = _item()
    await repo.enqueue(item)

    ack = await repo.acknowledge(item.item_id, note="looking")
    assert ack.changed is True
    resolve = await repo.resolve(item.item_id, resolution_label=ResolutionLabel.REQUIRES_FOLLOWUP)
    assert resolve.changed is True

    again = await repo.resolve(item.item_id, resolution_label=ResolutionLabel.REQUIRES_FOLLOWUP)
    assert again.no_op is True
    assert again.conflict is False
    assert again.state == ManualReviewState.RESOLVED


async def test_not_found_is_distinct_from_conflict(manual_review_mongo):
    repo = _repo()
    missing = await repo.acknowledge("does-not-exist")
    assert missing.not_found is True
    assert missing.success is False

    missing_resolve = await repo.resolve("does-not-exist")
    assert missing_resolve.not_found is True


# ── Submission-unknown safety ────────────────────────────────────────────────


async def test_submission_unknown_requires_explicit_label_and_never_marks_submitted(
    manual_review_mongo,
):
    repo = _repo()
    ambiguous = _item(
        category="submission_unknown_prior_attempt",
        attempt_id="att-9",
        attempt_status="submission_unknown",
        ambiguous_submission=True,
    )
    await repo.enqueue(ambiguous)

    rejected = await repo.resolve(ambiguous.item_id)
    assert rejected.success is False
    assert rejected.conflict is True
    assert rejected.reason == "submission_unknown_requires_explicit_resolution_label"
    current = await repo.get_item(ambiguous.item_id)
    assert current.state == ManualReviewState.OPEN
    assert current.attempt_status == "submission_unknown"

    resolved = await repo.resolve(
        ambiguous.item_id,
        note="verified manually by operator",
        resolution_label=ResolutionLabel.CONFIRMED_NOT_SUBMITTED,
    )
    assert resolved.changed is True
    final = await repo.get_item(ambiguous.item_id)
    assert final.state == ManualReviewState.RESOLVED
    assert final.resolution_label == ResolutionLabel.CONFIRMED_NOT_SUBMITTED
    # Resolution metadata only; no retry machinery or submission mutation exists.
    assert final.updated_at >= final.created_at


# ── Storage failure degradation ──────────────────────────────────────────────


async def test_storage_failure_keeps_outcome_needs_manual_review(manual_review_mongo):
    from jobapply.utils.manual_review import safe_enqueue_manual_review

    ok = await safe_enqueue_manual_review(_item())
    assert ok is True

    async def failing_enqueue(item):
        raise RuntimeError("mongo down")

    with patch.object(ManualReviewRepository, "enqueue", side_effect=failing_enqueue):
        degraded = await safe_enqueue_manual_review(_item(job_id="402"))
    assert degraded is False

    # The full outcome path also survives queue unavailability.
    state = {
        "run_id": "run-err",
        "current_job": {"job_id": "500", "title": "T", "company": "C"},
        "qualification_result": {"score": 0.5},
        "resume_path": None,
        "cover_letter_path": None,
        "approval_status": None,
        "application_outcomes": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
    }
    with patch.object(ManualReviewRepository, "enqueue", side_effect=failing_enqueue):
        update = await manual_review_update(state, "unresolved_required_fields", "boom", [])
    outcome = update["application_outcomes"][-1]
    assert outcome["status"] == "needs_manual_review"
    assert update["application_status"] == "needs_manual_review"
    assert outcome["reason"] == "unresolved_required_fields"


# ── Bounded redacted listing via CLI ─────────────────────────────────────────


def test_review_cli_list_acknowledge_resolve_exit_codes(manual_review_mongo, capsys):
    from types import SimpleNamespace

    from jobapply.review_cli import run_review_cli

    async def scenario():
        repo = _repo()
        item = _item(category="external_or_assessment", detail="Redirected to assessment")
        await repo.enqueue(item)

        listed = await run_review_cli(SimpleNamespace(review_command="list", limit=10, state="all"))
        assert listed == 0

        ack = await run_review_cli(
            SimpleNamespace(review_command="acknowledge", item_id=item.item_id, note="on it")
        )
        assert ack == 0

        re_ack = await run_review_cli(
            SimpleNamespace(review_command="acknowledge", item_id=item.item_id, note=None)
        )
        assert re_ack == 0  # clear no-op

        missing = await run_review_cli(
            SimpleNamespace(review_command="acknowledge", item_id="ghost", note=None)
        )
        assert missing == 4  # EXIT_NOT_FOUND

        resolved = await run_review_cli(
            SimpleNamespace(
                review_command="resolve",
                item_id=item.item_id,
                note=None,
                resolution_label="requires_followup",
            )
        )
        assert resolved == 0

        conflict = await run_review_cli(
            SimpleNamespace(
                review_command="resolve",
                item_id=item.item_id,
                note=None,
                resolution_label="confirmed_submitted",
            )
        )
        assert conflict == 3  # EXIT_CONFLICT (already resolved with different label)

        ambiguous = _item(
            category="submission_unknown_prior_attempt",
            attempt_status="submission_unknown",
            ambiguous_submission=True,
        )
        await repo.enqueue(ambiguous)
        no_label = await run_review_cli(
            SimpleNamespace(
                review_command="resolve",
                item_id=ambiguous.item_id,
                note=None,
                resolution_label=None,
            )
        )
        assert no_label == 3

        out = capsys.readouterr().out
        assert "Not found: ghost" in out
        assert "No-op:" in out
        assert "Conflict:" in out
        assert "AMBIGUOUS-SUBMISSION" in out or "submission_unknown" in out

    asyncio.run(scenario())


def test_listing_output_is_bounded_by_limit(manual_review_mongo, capsys):
    from types import SimpleNamespace

    from jobapply.review_cli import run_review_cli

    async def scenario():
        repo = _repo()
        for i in range(7):
            await repo.enqueue(_item(job_id=f"60{i}"))
        code = await run_review_cli(SimpleNamespace(review_command="list", limit=3, state="all"))
        assert code == 0
        out = capsys.readouterr().out
        assert out.count("state=open") == 3
        assert "3 item(s) listed." in out

    asyncio.run(scenario())


def test_state_filter_lists_only_requested_state(manual_review_mongo, capsys):
    from types import SimpleNamespace

    from jobapply.review_cli import run_review_cli

    async def scenario():
        repo = _repo()
        a, b = _item(job_id="701"), _item(job_id="702")
        await repo.enqueue(a)
        await repo.enqueue(b)
        await repo.acknowledge(a.item_id)

        code = await run_review_cli(SimpleNamespace(review_command="list", limit=20, state="open"))
        assert code == 0
        out = capsys.readouterr().out
        assert out.count("state=open") == 1
        assert "702" in out

    asyncio.run(scenario())


# ── Durability: pending retention, retries, visibility ───────────────────────


def _outcome_state(**extra):
    state = {
        "run_id": "dur-run",
        "current_job": {"job_id": "900", "title": "T", "company": "C"},
        "qualification_result": {"score": 0.5},
        "resume_path": None,
        "cover_letter_path": None,
        "approval_status": None,
        "application_outcomes": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
        "manual_review_queue_pending": [],
    }
    state.update(extra)
    return state


async def test_temporary_enqueue_failure_retains_item_in_pending_state(manual_review_mongo):
    from jobapply.utils.manual_review import serialize_pending_manual_review_item

    async def failing_enqueue(item):
        raise RuntimeError("mongo down")

    with patch.object(ManualReviewRepository, "enqueue", side_effect=failing_enqueue):
        update = await manual_review_update(
            _outcome_state(), "external_or_assessment", "redirected", []
        )

    pending = update["manual_review_queue_pending"]
    assert len(pending) == 1
    entry = pending[0]
    assert entry["run_id"] == "dur-run"
    assert entry["job_id"] == "900"
    # Serialized entries are bounded/redacted by construction.
    serialized = serialize_pending_manual_review_item(
        build_manual_review_item(
            run_id="r",
            job_id="j",
            title="t" * 400,
            company=None,
            url="mongodb://user:secret@host/db",
            reason_category="c",
        )
    )
    assert len(serialized["title"]) <= 256
    assert "secret" not in serialized["url"]

    # Outcome itself remains needs_manual_review.
    assert update["application_outcomes"][-1]["status"] == "needs_manual_review"


async def test_successful_flush_removes_items_and_failure_keeps_them(manual_review_mongo):
    from jobapply.utils.manual_review import (
        flush_manual_review_queue,
        serialize_pending_manual_review_item,
    )

    async def failing_enqueue(item):
        raise RuntimeError("still down")

    built = _item(job_id="950", category="form_qa_timeout", detail="d1")
    pending_entry = {
        "item_id": built.item_id,
        "idempotency_key": built.idempotency_key,
        "run_id": built.run_id,
        "job_id": built.job_id,
        "title": built.title,
        "company": built.company,
        "url": built.url,
        "reason_category": built.reason_category,
        "reason_detail": built.reason_detail,
        "attempt_id": None,
        "attempt_status": None,
        "ambiguous_submission": False,
        "created_at": built.created_at.isoformat(),
        "updated_at": built.updated_at.isoformat(),
    }
    state = {"manual_review_queue_pending": [pending_entry]}

    with patch.object(ManualReviewRepository, "enqueue", side_effect=failing_enqueue):
        result = await flush_manual_review_queue(state)
    # Kept visible, re-serialized through the bounded redaction boundary.
    assert result["manual_review_queue_pending"] == [serialize_pending_manual_review_item(built)]

    # Outage over: the same entry now enqueues successfully and leaves pending.
    result_ok = await flush_manual_review_queue(state)
    assert result_ok["manual_review_queue_pending"] == []

    # Idempotent: flushing an empty queue keeps it empty.
    again = await flush_manual_review_queue({"manual_review_queue_pending": []})
    assert again["manual_review_queue_pending"] == []


def test_persistent_failure_stays_visible_in_summary(manual_review_mongo):
    from datetime import datetime, timezone

    from jobapply.utils.summary import build_summary

    pending = [{"item_id": "mr_1", "reason_category": "x"}]

    async def failing_enqueue(item):
        raise RuntimeError("down")

    with patch.object(ManualReviewRepository, "enqueue", side_effect=failing_enqueue):
        summary = build_summary(
            run_id="r",
            terminal_status="completed",
            dry_run=True,
            started_at=datetime.now(timezone.utc),
            state={
                "application_outcomes": [],
                "manual_review_queue_pending": pending,
            },
        )
    assert summary["manual_review_queue_pending_count"] == 1


def test_cli_storage_error_returns_deterministic_code_and_redacts(monkeypatch, capsys):
    from types import SimpleNamespace

    from jobapply.review_cli import EXIT_STORAGE_ERROR, run_review_cli

    secret_url = "mongodb://produser:supersecret@prodhost:27017"

    class BrokenRepo:
        def __init__(self):
            raise RuntimeError(f"cannot connect to {secret_url} with huge " + "z" * 500)

    monkeypatch.setattr("jobapply.review_cli.ManualReviewRepository", BrokenRepo)

    async def scenario():
        return await run_review_cli(SimpleNamespace(review_command="list", limit=5, state="all"))

    code = asyncio.run(scenario())
    out = capsys.readouterr().out
    assert code == EXIT_STORAGE_ERROR
    assert "Storage error" in out
    assert secret_url not in out
    assert "produser" not in out
    assert "supersecret" not in out


def test_malformed_pending_entries_survive_flush_attempts(manual_review_mongo):
    from jobapply.utils.manual_review import flush_manual_review_queue

    malformed = {"unexpected": "shape"}
    state = {"manual_review_queue_pending": [malformed]}
    result = asyncio.run(flush_manual_review_queue(state))
    assert result["manual_review_queue_pending"] == [malformed]


# ── Model trust-boundary bounds ──────────────────────────────────────────────


def test_model_bounds_reject_hostile_documents():
    from pydantic import ValidationError

    from jobapply.models.manual_review import ManualReviewItem

    base = dict(
        item_id="mr_x",
        idempotency_key="mr_y",
        run_id="r",
        reason_category="cat",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    ok = ManualReviewItem(**base)
    assert ok.state == ManualReviewState.OPEN

    with pytest.raises(ValidationError):
        ManualReviewItem(**{**base, "title": "x" * 999})
    with pytest.raises(ValidationError):
        ManualReviewItem(**{**base, "run_id": "x" * 200})
    with pytest.raises(ValidationError):
        ManualReviewItem(**{**base, "reason_detail": "x" * 5000})


# ── Correction delta §2: adversarial redaction, loss visibility, overflow ───


def test_build_redacts_credentials_and_prefixed_env_secrets(monkeypatch, manual_review_mongo):

    prefixed_token = "999888777:AABPrefixEnvSyntheticTokenValue"
    monkeypatch.setenv("JOBAPPLY_TELEGRAM_BOT_TOKEN", prefixed_token)

    item = build_manual_review_item(
        run_id="run-sec",
        job_id="404",
        title=f"Engineer leaked {prefixed_token}",
        company="Acme",
        url="mongodb://user:sup3rsecret@prodhost:27017/db?replicaSet=rs0",
        reason_category="external_or_assessment",
        reason_detail="boom while calling mongodb://user:sup3rsecret@prodhost:27017/db",
        source_outcome={"reason": "external_or_assessment", "note": f"token={prefixed_token}"},
    )
    assert item.item_id == item.idempotency_key

    repo = _repo()
    res = asyncio.run(repo.enqueue(item))
    assert res.queued is True

    stored_doc = queue_only_doc(manual_review_mongo, item.item_id)
    blob = repr(stored_doc)
    assert "sup3rsecret" not in blob
    assert prefixed_token not in blob
    assert "[REDACTED]" in blob  # something was actively scrubbed


def test_item_key_deterministic_after_sanitization(manual_review_mongo):
    a = build_manual_review_item(
        run_id="r",
        job_id="j",
        title=None,
        company=None,
        url=None,
        reason_category="c",
        reason_detail="failed at mongodb://user:pw@h/d with trailing spaces   ",
    )
    b = build_manual_review_item(
        run_id="r",
        job_id="j",
        title=None,
        company=None,
        url=None,
        reason_category="c",
        reason_detail="failed at mongodb://user:pw@h/d with trailing spaces",
    )
    # Same sanitized payload => same deterministic key.
    assert a.idempotency_key == b.idempotency_key


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_construction_failure_keeps_retryable_state_then_recovers(monkeypatch, manual_review_mongo):
    """Sync construction failure -> bounded pending state -> real flush recovery."""
    from jobapply.nodes.execution import manual_review_update
    from jobapply.utils.manual_review import flush_manual_review_queue

    def broken_builder(**kwargs):  # synchronous, like production
        raise RuntimeError("hostile field shapes")

    monkeypatch.setattr(
        "jobapply.execution.outcomes.manual_review_item_from_outcome", broken_builder
    )
    update = asyncio.run(
        manual_review_update(
            _outcome_state(), "external_or_assessment", "redirected to assessment", []
        )
    )

    # 1-2. Bounded pending entry retained with a diagnostic marker.
    pending = update["manual_review_queue_pending"]
    assert len(pending) == 1
    entry = pending[0]
    assert entry["construction_failed"] is True
    assert entry["state"] == "open"
    assert entry["created_at"] and entry["updated_at"]
    assert entry["item_id"] == entry["idempotency_key"]
    # Outcome remains truthful regardless of the construction failure.
    assert update["application_outcomes"][-1]["status"] == "needs_manual_review"

    # 3. Queue availability restored (patch lifted).
    # 4. Real central flush boundary parses and enqueues the fallback entry.
    merged = {**_outcome_state(), **update}
    flush_result = asyncio.run(flush_manual_review_queue(merged))

    # 5. Durably enqueued exactly once and removed from pending state.
    assert flush_result["manual_review_queue_pending"] == []
    docs = list(manual_review_mongo.docs.values())
    assert len(docs) == 1
    assert docs[0]["item_id"] == entry["item_id"]
    assert "construction_failed" not in repr(docs[0])


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_full_pending_queue_retains_entries_and_reconstructs_newest(manual_review_mongo):
    """Full pending list: nothing evicted; newest failure recovers via its outcome."""
    from jobapply.utils.manual_review import (
        MAX_PENDING_MANUAL_REVIEW_ITEMS,
        flush_manual_review_queue,
    )
    from jobapply.utils.summary import build_summary

    original_ids = [f"mr_old_{i}" for i in range(MAX_PENDING_MANUAL_REVIEW_ITEMS)]
    full = [
        {"item_id": item_id, "run_id": "dur-run", "reason_category": "c", "state": "open"}
        for item_id in original_ids
    ]
    state = _outcome_state(manual_review_queue_pending=full, manual_review_queue_overflow=2)

    async def failing_enqueue(item):
        raise RuntimeError("mongo down")

    with patch.object(ManualReviewRepository, "enqueue", side_effect=failing_enqueue):
        update = asyncio.run(manual_review_update(state, "new_failure_reason", "err", []))

    # All original pending IDs remain — nothing was evicted.
    pending = update["manual_review_queue_pending"]
    assert [p["item_id"] for p in pending] == original_ids
    # The new failed item is absent from the bounded list...
    assert all(p["reason_category"] != "new_failure_reason" for p in pending)
    # ...overflow accounting incremented atomically alongside the new outcome.
    assert update["manual_review_queue_overflow"] == 3
    new_outcomes = [
        o for o in update["application_outcomes"] if o.get("reason") == "new_failure_reason"
    ]
    assert len(new_outcomes) == 1 and new_outcomes[0]["status"] == "needs_manual_review"

    summary = build_summary(
        run_id="dur-run",
        terminal_status="completed",
        dry_run=True,
        started_at=datetime.now(timezone.utc),
        state={**state, **update},
    )
    assert summary["manual_review_queue_pending_count"] == MAX_PENDING_MANUAL_REVIEW_ITEMS
    assert summary["manual_review_queue_overflow_count"] == 3

    # Later real central flush: the new item is reconstructed from its
    # authoritative outcome and durably enqueued exactly once.
    merged = {**state, **update}
    flush_result = asyncio.run(flush_manual_review_queue(merged))
    new_key_docs = [
        d for d in manual_review_mongo.docs.values() if d["reason_category"] == "new_failure_reason"
    ]
    assert len(new_key_docs) == 1
    assert [p["item_id"] for p in flush_result["manual_review_queue_pending"]] == original_ids


def test_queue_source_metadata_redacts_repost_evidence(manual_review_mongo):
    secret = "777666555:AABQueueSourceSecretTokenValue"

    outcome_state = _outcome_state()
    outcome_state["current_job"] = {
        "job_id": "901",
        "title": "T",
        "company": "C",
        "is_repost": True,
        "repost_evidence": f"Reposted leak {secret} " + "q" * 500,
    }

    async def run_it():
        return await manual_review_update(outcome_state, "external_or_assessment", "redirected", [])

    update = asyncio.run(run_it())
    assert update["application_outcomes"][-1]["status"] == "needs_manual_review"

    docs = list(manual_review_mongo.docs.values())
    assert docs, "queue document must be persisted on success"
    blob = repr(docs[0])
    assert secret not in blob
    evidence = docs[0]["source_outcome"].get("repost_evidence")
    assert evidence is not None and len(evidence) <= 200


def queue_only_doc(collection, item_id):
    for doc in collection.docs.values():
        if doc.get("item_id") == item_id:
            return doc
    raise AssertionError(f"item {item_id} not found")


# ── Correction delta §3: overflow entries are recoverable, not lost ─────────


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_overflow_evicted_items_are_reconstructed_exactly_once(manual_review_mongo):
    """More outcomes than pending slots: writes fail, storage recovers, central
    flush reconstructs every authoritative manual-review outcome exactly once."""
    from jobapply.nodes.execution import manual_review_update
    from jobapply.utils.manual_review import (
        MAX_PENDING_MANUAL_REVIEW_ITEMS,
        flush_manual_review_queue,
    )

    total = MAX_PENDING_MANUAL_REVIEW_ITEMS + 3

    def outcome_state(i):
        return _outcome_state(
            job_id=f"ovf-{i}",
            manual_review_queue_pending=list(_thread_state["pending"]),
            manual_review_queue_overflow=_thread_state["overflow"],
        )

    async def failing_enqueue(item):
        raise RuntimeError("mongo down")

    # 1. Initial queue writes all fail while threading durable state forward.
    _thread_state = {"pending": [], "overflow": 0}
    merged_state = _outcome_state()
    with patch.object(ManualReviewRepository, "enqueue", side_effect=failing_enqueue):
        for i in range(total):
            update = asyncio.run(manual_review_update(merged_state, f"reason_{i}", "err", []))
            merged_state = {**merged_state, **update}

    # 2. Bounded pending state recorded the overflow explicitly.
    assert len(merged_state["manual_review_queue_pending"]) == MAX_PENDING_MANUAL_REVIEW_ITEMS
    assert merged_state["manual_review_queue_overflow"] == 3
    assert len(merged_state["application_outcomes"]) == total  # authoritative list intact
    assert all(o["status"] == "needs_manual_review" for o in merged_state["application_outcomes"])

    # 3. Storage recovers (patch lifted).
    # 4. Central flush reconstructs from authoritative outcomes and enqueues
    #    every item exactly once; explicit pending entries dedupe via key.
    flush_result = asyncio.run(flush_manual_review_queue(merged_state))

    docs = list(manual_review_mongo.docs.values())
    assert len(docs) == total
    stored_ids = {d["idempotency_key"] for d in docs}
    assert len(stored_ids) == total
    # Expected keys derive exactly like production: from the authoritative
    # outcome payloads themselves.
    expected_ids = {
        make_manual_review_idempotency_key(
            "dur-run",
            o["job_id"],
            o["reason"],
            o["error"],
        )
        for o in merged_state["application_outcomes"]
    }
    assert stored_ids == expected_ids
    assert flush_result["manual_review_queue_pending"] == []

    # Submission truth untouched by reconstruction.
    assert all(o["status"] == "needs_manual_review" for o in merged_state["application_outcomes"])
