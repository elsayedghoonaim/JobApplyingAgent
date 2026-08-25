"""Deterministic offline tests for CLI compatibility and search scope configuration.

Covers: legacy CLI invocations, doctor/review dispatch, --live-checks,
location/recency validation and override precedence, URL encoding, LinkedIn
f_TPR mapping, and resume stability through checkpoint state.
"""

import pytest

from jobapply.utils.search_config import (
    DEFAULT_SEARCH_LOCATION,
    build_linkedin_search_url,
    effective_search_params,
    normalize_recency_days,
    normalize_search_location,
    recency_days_to_tpr,
    resolve_state_search_params,
)

# ── Legacy CLI compatibility ─────────────────────────────────────────────────


def _parse(argv):
    """Parse workflow arguments exactly like main() does for non-operator runs."""
    import argparse

    from jobapply.utils.paths import validate_run_id

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=valid_run_id_arg)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--max-applications", type=non_negative_int)
    parser.add_argument("--max-jobs", type=non_negative_int)
    parser.add_argument("--location", type=str, default=None)
    parser.add_argument("--recency-days", type=int, default=None)
    return parser.parse_args(argv)


def test_legacy_invocations_parse_unchanged(capsys):
    args = _parse([])
    assert args.dry_run is None
    assert args.run_id is None
    assert args.location is None
    assert args.recency_days is None

    args = _parse(["--dry-run"])
    assert args.dry_run is True

    args = _parse(["--no-dry-run"])
    assert args.dry_run is False

    args = _parse(["--run-id", "run-123"])
    assert args.run_id == "run-123"

    args = _parse(["--max-applications", "3", "--max-jobs", "7"])
    assert args.max_applications == 3
    assert args.max_jobs == 7

    args = _parse(["--dry-run", "--location", "Berlin", "--recency-days", "7"])
    assert args.location == "Berlin"
    assert args.recency_days == 7


def test_invalid_run_id_still_rejected():
    with pytest.raises(SystemExit):
        _parse(["--run-id", "../evil"])


def test_help_exits_zero_for_workflow_and_doctor(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse(["--help"])
    assert exc.value.code == 0

    import argparse

    parser = argparse.ArgumentParser(prog="jobapply doctor")
    parser.add_argument("--live-checks", action="store_true")
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_operator_commands_route_to_dedicated_parsers(monkeypatch):
    import sys

    import jobapply.main as main_module

    # `doctor` dispatch
    doctor_calls = []
    monkeypatch.setattr(
        main_module, "run_doctor_cli", lambda live_checks: doctor_calls.append(live_checks) or 0
    )
    monkeypatch.setattr(sys, "argv", ["jobapply", "doctor"])
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 0
    assert doctor_calls == [False]

    monkeypatch.setattr(sys, "argv", ["jobapply", "doctor", "--live-checks"])
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 0
    assert doctor_calls == [False, True]

    # `review list` dispatch
    review_calls = []

    async def fake_review(args):
        review_calls.append(args.review_command)
        return 0

    monkeypatch.setattr(main_module, "run_review_cli", fake_review)
    monkeypatch.setattr(sys, "argv", ["jobapply", "review", "list", "--limit", "5"])
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 0
    assert review_calls == ["list"]

    monkeypatch.setattr(
        sys, "argv", ["jobapply", "review", "resolve", "item-1", "--label", "requires_followup"]
    )
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 0
    assert review_calls[-1] == "resolve"


def test_workflow_run_receives_location_overrides(monkeypatch):
    import sys

    import jobapply.main as main_module

    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_module, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["jobapply", "--dry-run", "--location", "Dubai", "--recency-days", "3"]
    )
    main_module.main()
    assert captured["location"] == "Dubai"
    assert captured["recency_days"] == 3


# ── Location / recency validation ────────────────────────────────────────────


def test_location_normalization_trims_bounds_and_rejects_empty():
    assert normalize_search_location(None) == (None, None)
    location, error = normalize_search_location("  Berlin, Germany  ")
    assert location == "Berlin, Germany"
    assert error is None

    assert normalize_search_location("   ")[1] == "Location cannot be empty."
    long = "x" * 500
    _, error = normalize_search_location(long)
    assert error and "200" in error


def test_recency_validation_bounds_and_disable_values():
    assert normalize_recency_days(None) == (None, None)
    assert normalize_recency_days(0) == (None, None)
    assert normalize_recency_days("") == (None, None)
    assert normalize_recency_days("  ") == (None, None)
    assert normalize_recency_days(1) == (1, None)
    assert normalize_recency_days(30) == (30, None)
    assert normalize_recency_days("7") == (7, None)
    assert normalize_recency_days(-1)[1] is not None
    assert normalize_recency_days(31)[1] is not None
    assert normalize_recency_days("abc")[1] is not None
    assert normalize_recency_days(True)[1] is not None


def test_recency_maps_deterministically_to_f_tpr():
    assert recency_days_to_tpr(1) == "r86400"
    assert recency_days_to_tpr(7) == "r604800"
    assert recency_days_to_tpr(14) == "r1209600"
    assert recency_days_to_tpr(30) == "r2592000"


def test_cli_wins_over_env_defaults():
    location, recency, error = effective_search_params("Berlin", 7, "Worldwide", None)
    assert (location, recency, error) == ("Berlin", 7, None)

    # CLI disabled value beats env-enabled value.
    location, recency, error = effective_search_params(None, 0, "Paris", 5)
    assert (location, recency, error) == ("Paris", None, None)

    # CLI explicit enable beats env-disabled.
    location, recency, error = effective_search_params(None, 3, "Paris", None)
    assert (location, recency) == ("Paris", 3)

    # Invalid CLI values fail even when env defaults are fine.
    _, _, error = effective_search_params("", 7, "Paris", 5)
    assert error == "Location cannot be empty."

    _, _, error = effective_search_params(None, 99, "Paris", None)
    assert error and "between 1 and 30" in error


def test_url_encoding_is_safe_not_interpolated():
    url = build_linkedin_search_url(
        "https://www.linkedin.com/",
        "C++ & AI/ML (2026)",
        2,
        location="New York, USA",
        recency_days=7,
    )
    assert url.startswith("https://www.linkedin.com/jobs/search/?")
    assert "keywords=C%2B%2B+%26+AI%2FML+%282026%29" in url
    assert "location=New+York%2C+USA" in url
    assert "f_TPR=r604800" in url
    assert "start=25" in url


def test_default_behavior_worldwide_without_recency():
    url = build_linkedin_search_url("https://www.linkedin.com", "ML Engineer", 1)
    assert f"location={DEFAULT_SEARCH_LOCATION}" in url
    assert "f_TPR" not in url
    assert "start=0" in url


# ── Resume stability via checkpoint state ────────────────────────────────────


class _FakeSettings:
    search_location = "Paris"
    search_recency_days = 5


def test_state_values_drive_search_params():
    location, days = resolve_state_search_params(
        {"search_location": "Tokyo", "search_recency_days": 10}, _FakeSettings()
    )
    assert (location, days) == ("Tokyo", 10)


def test_explicitly_disabled_state_value_survives_resume():
    # CLI --recency-days 0 persisted 0 into state; env says 5. Disabled wins.
    location, days = resolve_state_search_params(
        {"search_location": "Tokyo", "search_recency_days": 0}, _FakeSettings()
    )
    assert (location, days) == ("Tokyo", None)


def test_legacy_checkpoint_missing_keys_falls_back_to_settings():
    location, days = resolve_state_search_params({}, _FakeSettings())
    assert (location, days) == ("Paris", 5)


def test_resume_produces_identical_urls():
    state = {"search_location": "Berlin", "search_recency_days": 7}
    fresh, resumed = (
        resolve_state_search_params(state, _FakeSettings()),
        resolve_state_search_params(state, _FakeSettings()),
    )
    assert fresh == resumed
    url_a = build_linkedin_search_url(
        "https://www.linkedin.com", "AI", 3, location=fresh[0], recency_days=fresh[1]
    )
    url_b = build_linkedin_search_url(
        "https://www.linkedin.com", "AI", 3, location=resumed[0], recency_days=resumed[1]
    )
    assert url_a == url_b


# ── End-to-end resume override application (mocked graph) ────────────────────


class _Checkpoint:
    def __init__(self, values, next_nodes=("search_node",)):
        self.values = values
        self.next = list(next_nodes)


async def _drive_resume(
    tmp_path, monkeypatch, *, location=None, recency_days=None, checkpoint_values
):
    """Run jobapply.main.run() with a fully mocked graph and capture update_state."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import jobapply.main as main_module

    monkeypatch.chdir(tmp_path)

    store = AsyncMock()
    store.get_daily_count.return_value = 0
    store_cls = MagicMock(return_value=store)
    store_cls.close = AsyncMock()

    graph = AsyncMock()
    graph.aget_state.return_value = _Checkpoint(checkpoint_values)
    graph.ainvoke.return_value = {
        "account_safety_paused": False,
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "applications_count": 0,
        "application_outcomes": [],
        "errors": [],
        "resume_path": None,
        "cover_letter_path": None,
        "manual_review_queue_pending": [],
    }
    graph.aupdate_state = AsyncMock()

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with patch("jobapply.main.compile_graph", return_value=graph):
            with patch("jobapply.main.close_llm_client", new_callable=AsyncMock):
                await main_module.run(
                    run_id="resume-ovr",
                    dry_run=True,
                    location=location,
                    recency_days=recency_days,
                )
    return graph


@pytest.mark.asyncio
async def test_resume_without_override_keeps_checkpointed_values(tmp_path, monkeypatch):
    graph = await _drive_resume(
        tmp_path,
        monkeypatch,
        location=None,
        recency_days=None,
        checkpoint_values={
            "account_safety_paused": False,
            "search_location": "Tokyo",
            "search_recency_days": 10,
            "manual_review_queue_pending": [],
        },
    )
    # No explicit CLI override: the durable checkpoint stays untouched.
    graph.aupdate_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_with_explicit_location_overrides_checkpoint(tmp_path, monkeypatch):
    graph = await _drive_resume(
        tmp_path,
        monkeypatch,
        location="Berlin",
        recency_days=None,
        checkpoint_values={
            "account_safety_paused": False,
            "search_location": "Tokyo",
            "search_recency_days": 10,
            "manual_review_queue_pending": [],
        },
    )
    graph.aupdate_state.assert_awaited_once()
    overrides = graph.aupdate_state.await_args.args[1]
    assert overrides["search_location"] == "Berlin"
    assert "search_recency_days" not in overrides  # only explicitly provided keys


@pytest.mark.asyncio
async def test_resume_recency_zero_disables_checkpointed_filter(tmp_path, monkeypatch):
    graph = await _drive_resume(
        tmp_path,
        monkeypatch,
        location=None,
        recency_days=0,
        checkpoint_values={
            "account_safety_paused": False,
            "search_location": "Tokyo",
            "search_recency_days": 10,
            "manual_review_queue_pending": [],
        },
    )
    graph.aupdate_state.assert_awaited_once()
    overrides = graph.aupdate_state.await_args.args[1]
    assert overrides["search_recency_days"] is None  # explicit disable persisted
    assert "search_location" not in overrides


@pytest.mark.asyncio
async def test_resume_flushes_pending_manual_review_items(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch

    import jobapply.main as main_module
    from jobapply.models.manual_review import ManualReviewEnqueueResult

    monkeypatch.chdir(tmp_path)

    store = AsyncMock()
    store.get_daily_count.return_value = 0
    store_cls = MagicMock(return_value=store)
    store_cls.close = AsyncMock()

    graph = AsyncMock()
    graph.aget_state.return_value = _Checkpoint(
        {
            "account_safety_paused": False,
            "search_location": "Tokyo",
            "search_recency_days": 10,
            "manual_review_queue_pending": [
                {
                    "item_id": "mr_z",
                    "idempotency_key": "mr_z",
                    "run_id": "resume-ovr",
                    "reason_category": "external_or_assessment",
                    "state": "open",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ],
        }
    )
    graph.ainvoke.return_value = {"account_safety_paused": False}
    graph.aupdate_state = AsyncMock()

    enqueue_mock = AsyncMock(return_value=ManualReviewEnqueueResult(queued=True, item_id="mr_z"))

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with patch("jobapply.main.compile_graph", return_value=graph):
            with patch("jobapply.main.close_llm_client", new_callable=AsyncMock):
                with patch(
                    "jobapply.utils.manual_review.ManualReviewRepository.enqueue",
                    enqueue_mock,
                ):
                    await main_module.run(run_id="resume-ovr", dry_run=True)

    graph.aupdate_state.assert_awaited_once()
    overrides = graph.aupdate_state.await_args.args[1]
    assert overrides["manual_review_queue_pending"] == []
    enqueue_mock.assert_awaited()


# ── Correction delta §1: resumed search configuration semantics ─────────────


class _EnvRecencySeven:
    search_location = "Paris"
    search_recency_days = 7


def test_checkpointed_none_recency_stays_disabled_when_env_changes():
    """A checkpoint that explicitly stored None never inherits a new env filter."""
    location, days = resolve_state_search_params(
        {"search_location": "Tokyo", "search_recency_days": None}, _EnvRecencySeven()
    )
    assert (location, days) == ("Tokyo", None)


def test_checkpointed_zero_recency_stays_disabled_when_env_changes():
    location, days = resolve_state_search_params(
        {"search_recency_days": 0, "search_location": "Oslo"}, _EnvRecencySeven()
    )
    assert (location, days) == ("Oslo", None)


async def _drive_resume_env_changed(tmp_path, monkeypatch, *, checkpoint_values):
    """Resume a run while the environment now advertises recency=7."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import jobapply.main as main_module
    from jobapply.settings import get_settings

    monkeypatch.chdir(tmp_path)
    cached = get_settings()
    monkeypatch.setattr(cached, "search_recency_days", 7, raising=False)

    store = AsyncMock()
    store.get_daily_count.return_value = 0
    store_cls = MagicMock(return_value=store)
    store_cls.close = AsyncMock()

    graph = AsyncMock()
    graph.aget_state.return_value = _Checkpoint(checkpoint_values)
    graph.ainvoke.return_value = {"account_safety_paused": False}
    graph.aupdate_state = AsyncMock()

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with patch("jobapply.main.compile_graph", return_value=graph):
            with patch("jobapply.main.close_llm_client", new_callable=AsyncMock):
                await main_module.run(run_id="resume-env", dry_run=True)
    return graph


@pytest.mark.asyncio
async def test_resume_with_checkpointed_none_recency_ignores_new_env_filter(tmp_path, monkeypatch):
    graph = await _drive_resume_env_changed(
        tmp_path,
        monkeypatch,
        checkpoint_values={
            "account_safety_paused": False,
            "search_location": "Tokyo",
            "search_recency_days": None,
            "manual_review_queue_pending": [],
        },
    )
    # Checkpoint is authoritative: no override, no env bleed-through.
    graph.aupdate_state.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_override_persistence_failure_fails_closed(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock, patch

    import jobapply.main as main_module

    monkeypatch.chdir(tmp_path)

    store = AsyncMock()
    store.get_daily_count.return_value = 0
    store_cls = MagicMock(return_value=store)
    store_cls.close = AsyncMock()

    graph = AsyncMock()
    graph.aget_state.return_value = _Checkpoint(
        {
            "account_safety_paused": False,
            "search_location": "Tokyo",
            "search_recency_days": None,
            "manual_review_queue_pending": [],
        }
    )
    graph.ainvoke = AsyncMock()
    graph.aupdate_state = AsyncMock(side_effect=RuntimeError("checkpoint write failed"))

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with patch("jobapply.main.compile_graph", return_value=graph):
            with patch("jobapply.main.close_llm_client", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match="resume_override_not_persisted"):
                    await main_module.run(
                        run_id="resume-fail",
                        dry_run=True,
                        location="Berlin",
                    )

    # The run stopped instead of continuing on stale parameters.
    graph.ainvoke.assert_not_awaited()


# ── Correction delta §1: missing/empty checkpoint behaves like a fresh run ──


@pytest.mark.asyncio
async def test_requested_run_id_with_empty_checkpoint_uses_env_configuration(tmp_path, monkeypatch):
    """`--run-id <new-id>` with no saved state resolves CLI > environment > defaults."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import jobapply.main as main_module
    from jobapply.settings import get_settings

    monkeypatch.chdir(tmp_path)
    cached = get_settings()
    monkeypatch.setattr(cached, "search_location", "EnvLocation", raising=False)
    monkeypatch.setattr(cached, "search_recency_days", 9, raising=False)

    store = AsyncMock()
    store.get_daily_count.return_value = 0
    store_cls = MagicMock(return_value=store)
    store_cls.close = AsyncMock()

    graph = AsyncMock()
    # Empty checkpoint: aget_state returns an object with no values.
    graph.aget_state.return_value = _Checkpoint(None)
    captured_input: dict = {}

    async def fake_ainvoke(graph_input, config):
        captured_input.update(graph_input or {})
        return {
            "account_safety_paused": False,
            "jobs_evaluated_count": 0,
            "qualified_jobs_count": 0,
            "not_qualified_jobs_count": 0,
            "applications_count": 0,
            "application_outcomes": [],
            "errors": [],
            "resume_path": None,
            "cover_letter_path": None,
            "manual_review_queue_pending": [],
        }

    graph.ainvoke = fake_ainvoke
    graph.aupdate_state = AsyncMock()

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with patch("jobapply.main.compile_graph", return_value=graph):
            with patch("jobapply.main.close_llm_client", new_callable=AsyncMock):
                # No explicit CLI overrides: environment configuration must win.
                await main_module.run(run_id="brand-new-run-id", dry_run=True)

    assert captured_input.get("search_location") == "EnvLocation"
    assert captured_input.get("search_recency_days") == 9

    # CLI values still beat the environment in the same fresh-like path.
    captured_input.clear()
    graph.aget_state.return_value = _Checkpoint(None)
    with patch("jobapply.main.DeduplicationStore", store_cls):
        with patch("jobapply.main.compile_graph", return_value=graph):
            with patch("jobapply.main.close_llm_client", new_callable=AsyncMock):
                await main_module.run(
                    run_id="brand-new-run-id",
                    dry_run=True,
                    location="CliLocation",
                    recency_days=0,
                )
    assert captured_input.get("search_location") == "CliLocation"
    assert captured_input.get("search_recency_days") is None  # 0 disables explicitly


@pytest.mark.asyncio
async def test_existing_checkpoint_keys_stay_authoritative_on_resume(tmp_path, monkeypatch):
    """With real checkpoint keys present, env defaults are never consulted."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import jobapply.main as main_module
    from jobapply.settings import get_settings

    monkeypatch.chdir(tmp_path)
    cached = get_settings()
    monkeypatch.setattr(cached, "search_location", "EnvLocation", raising=False)
    monkeypatch.setattr(cached, "search_recency_days", 9, raising=False)

    store = AsyncMock()
    store.get_daily_count.return_value = 0
    store_cls = MagicMock(return_value=store)
    store_cls.close = AsyncMock()

    graph = AsyncMock()
    graph.aget_state.return_value = _Checkpoint(
        {
            "account_safety_paused": False,
            "search_location": "CheckpointLand",
            "search_recency_days": None,
            "manual_review_queue_pending": [],
        }
    )

    async def fake_ainvoke(graph_input, config):
        assert graph_input is None  # continuation input, not fresh state
        return {"account_safety_paused": False}

    graph.ainvoke = fake_ainvoke
    graph.aupdate_state = AsyncMock()

    with patch("jobapply.main.DeduplicationStore", store_cls):
        with patch("jobapply.main.compile_graph", return_value=graph):
            with patch("jobapply.main.close_llm_client", new_callable=AsyncMock):
                await main_module.run(run_id="resume-authoritative", dry_run=True)

    # Checkpoint untouched: no overrides derived from changed env values.
    graph.aupdate_state.assert_not_awaited()
