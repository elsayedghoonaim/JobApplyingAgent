"""Tests for search completeness, batched deduplication, deterministic exclusions, and immutable state updates."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobapply.main import run
from jobapply.nodes.qualification import qualification_node
from jobapply.nodes.search import search_node
from jobapply.utils.account_safety import (
    AccountSafetyBarrierError,
    AccountSafetyBarrierType,
    AccountSafetyDetection,
)
from jobapply.utils.dedup import (
    DeduplicationStore,
    DeduplicationStoreError,
    bound_and_redact_metadata,
    canonicalize_job_id,
)


def _base_state(**updates):
    state = {
        "run_id": "test-run-task4",
        "dry_run": True,
        "search_queries": ["Software Engineer"],
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": 3,
        "job_listings": [],
        "current_job_index": 0,
        "current_job": None,
        "search_failed": False,
        "seen_job_ids": set(),
        "applications_count": 0,
        "daily_applications_count": 0,
        "max_applications": 1,
        "daily_application_cap": 10,
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
    }
    state.update(updates)
    return state


class AsyncCursor:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class CustomAwaitableFuture:
    """A non-coroutine awaitable object (similar to Motor Future/Tornado Future)."""

    def __init__(self, value=None):
        self._value = value
        self.awaited = False

    def __await__(self):
        self.awaited = True
        yield
        return self._value


def _create_mock_page(cards_data):
    """Create a mock Playwright page with cards and desc elements."""
    page = AsyncMock()
    page.goto.return_value = SimpleNamespace(status=200)
    page.is_closed = MagicMock(return_value=False)
    page.close = AsyncMock()

    card_mocks = []
    card_by_id = {}

    for cdata in cards_data:
        jid = cdata["job_id"]
        title = cdata.get("title", "Software Engineer")
        company = cdata.get("company", "Acme Corp")
        location = cdata.get("location", "Remote")
        applied = cdata.get("applied")
        dom_fail = cdata.get("dom_fail", False)

        card = AsyncMock()
        card.get_attribute.side_effect = lambda attr, j=jid: (
            j if attr == "data-occludable-job-id" else None
        )
        card.evaluate.return_value = [applied] if applied else []

        if dom_fail:
            card.query_selector.return_value = None
        else:
            link = AsyncMock()
            link.get_attribute.return_value = title
            link.inner_text.return_value = title
            link.click = AsyncMock()

            comp = AsyncMock()
            comp.inner_text.return_value = company

            loc = AsyncMock()
            loc.inner_text.return_value = location

            async def card_qs(sel, link_el=link, company_el=comp, location_el=loc):
                if "job-card-container__link" in sel:
                    return link_el
                if "artdeco-entity-lockup__subtitle" in sel:
                    return company_el
                if "job-card-container__metadata-wrapper" in sel:
                    return location_el
                return None

            card.query_selector.side_effect = card_qs

        card.scroll_into_view_if_needed = AsyncMock()
        card_mocks.append(card)
        card_by_id[jid] = card

    page.query_selector_all.return_value = card_mocks

    async def page_query_selector(sel):
        for jid, card in card_by_id.items():
            if f"'{jid}'" in sel or f'"{jid}"' in sel:
                return card
        return None

    page.query_selector.side_effect = page_query_selector

    desc_elem = AsyncMock()
    desc_elem.inner_text.return_value = "Standard job description with Python."
    page.wait_for_selector.return_value = desc_elem

    return page


@pytest.mark.asyncio
@patch("jobapply.nodes.search.asyncio.sleep", new_callable=AsyncMock)
@patch("jobapply.nodes.search.DeduplicationStore")
@patch("jobapply.nodes.search.parse_job_description")
@patch("jobapply.nodes.search.managed_browser")
async def test_search_completeness_inspects_all_cards_regardless_of_application_cap(
    mock_managed_browser,
    mock_parse_job,
    mock_dedup_cls,
    mock_search_sleep,
):
    """A page with more cards than the remaining application cap is fully inspected, batch-checked once, and listings remain unmarked."""
    mock_store = AsyncMock()
    mock_store.find_seen_ids.return_value = set()
    mock_dedup_cls.return_value = mock_store

    mock_parse_job.return_value = {
        "clean_description": "Clean description",
        "required_languages": ["English"],
    }

    # 6 cards on page, while max_applications=1
    card_ids = [f"job_{i}" for i in range(1, 7)]
    cards_data = [
        {"job_id": jid, "title": f"Developer {jid}", "company": f"Company {jid}"}
        for jid in card_ids
    ]
    page = _create_mock_page(cards_data)
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = (browser, context)
    cm.__aexit__.return_value = None
    mock_managed_browser.return_value = cm

    initial_seen = {"already_seen_0"}
    state = _base_state(max_applications=1, applications_count=0, seen_job_ids=initial_seen)
    result = await search_node(state)

    # find_seen_ids must be called exactly once with all 6 card IDs
    mock_store.find_seen_ids.assert_awaited_once()
    passed_ids = mock_store.find_seen_ids.call_args[0][0]
    assert set(passed_ids) == set(card_ids)

    # All 6 eligible cards must be extracted
    assert len(result["job_listings"]) == 6
    extracted_ids = [j["job_id"] for j in result["job_listings"]]
    assert extracted_ids == card_ids

    # Eligible listings must NOT be marked seen in search result (marked later in qualification)
    for jid in card_ids:
        assert jid not in result["seen_job_ids"]
    assert "already_seen_0" in result["seen_job_ids"]


@pytest.mark.asyncio
@patch("jobapply.nodes.search.asyncio.sleep", new_callable=AsyncMock)
@patch("jobapply.nodes.search.DeduplicationStore")
@patch("jobapply.nodes.search.parse_job_description")
@patch("jobapply.nodes.search.managed_browser")
async def test_search_inspects_more_than_25_cards_on_page(
    mock_managed_browser,
    mock_parse_job,
    mock_dedup_cls,
    mock_search_sleep,
):
    """More than 25 loaded cards are all inspected and batch-checked rather than being truncated by [:25]."""
    mock_store = AsyncMock()
    mock_store.find_seen_ids.return_value = set()
    mock_dedup_cls.return_value = mock_store

    mock_parse_job.return_value = {
        "clean_description": "Clean description",
        "required_languages": ["English"],
    }

    card_ids = [f"job_{i}" for i in range(32)]
    cards_data = [
        {"job_id": jid, "title": f"Engineer {jid}", "company": f"Corp {jid}"} for jid in card_ids
    ]
    page = _create_mock_page(cards_data)
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = (browser, context)
    cm.__aexit__.return_value = None
    mock_managed_browser.return_value = cm

    state = _base_state(max_applications=100)
    result = await search_node(state)

    mock_store.find_seen_ids.assert_awaited_once()
    assert set(mock_store.find_seen_ids.call_args[0][0]) == set(card_ids)

    assert len(result["job_listings"]) == 32
    assert [j["job_id"] for j in result["job_listings"]] == card_ids


@pytest.mark.asyncio
async def test_persisted_seen_ids_single_in_query_and_live_run_semantics():
    """find_seen_ids executes a single bounded $in query and respects live-run exclusion semantics."""
    await DeduplicationStore.close()
    store = object.__new__(DeduplicationStore)
    store.collection = MagicMock()
    store.collection.create_index.return_value = None
    store.collection.find.return_value = AsyncCursor([{"job_id": "job_1"}, {"job_id": "job_3"}])

    # Normal run
    seen = await store.find_seen_ids(["job_1", "job_2", "job_3", "job_1"], for_live_run=False)
    assert seen == {"job_1", "job_3"}
    query, projection = store.collection.find.call_args[0]
    assert set(query["job_id"]["$in"]) == {"job_1", "job_2", "job_3"}
    assert "status" not in query
    assert projection == {"job_id": 1}

    # Live run semantics
    store.collection.find.reset_mock()
    store.collection.find.return_value = AsyncCursor([{"job_id": "job_submitted"}])
    seen_live = await store.find_seen_ids(["job_submitted", "job_qualified"], for_live_run=True)
    assert seen_live == {"job_submitted"}
    query_live, _ = store.collection.find.call_args[0]
    assert query_live["status"] == {"$nin": ["qualified", "dry_run"]}
    await DeduplicationStore.close()


@pytest.mark.asyncio
@patch("jobapply.main.compile_graph")
@patch("jobapply.main.DeduplicationStore")
async def test_main_startup_does_not_load_full_seen_collection(mock_dedup_cls, mock_compile_graph):
    """Main startup queries daily count but does not load the full seen collection."""
    mock_store = AsyncMock()
    mock_store.get_daily_count.return_value = 5
    mock_dedup_cls.return_value = mock_store
    mock_dedup_cls.close = AsyncMock()

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "applications_count": 0,
        "application_outcomes": [],
    }
    mock_compile_graph.return_value = mock_graph

    # Run as fresh session (run_id=None)
    await run(run_id=None, dry_run=True)

    mock_store.get_daily_count.assert_awaited_once()
    mock_store.load_seen_ids.assert_not_called()
    mock_dedup_cls.close.assert_awaited_once()

    graph_input = mock_compile_graph.return_value.ainvoke.call_args[0][0]
    assert graph_input["seen_job_ids"] == set()


@pytest.mark.asyncio
@patch("jobapply.nodes.search.asyncio.sleep", new_callable=AsyncMock)
@patch("jobapply.nodes.search.DeduplicationStore")
@patch("jobapply.nodes.search.parse_job_description")
@patch("jobapply.nodes.search.managed_browser")
async def test_duplicate_cards_and_current_page_ids_handled_deterministically(
    mock_managed_browser,
    mock_parse_job,
    mock_dedup_cls,
    mock_search_sleep,
):
    """Duplicate cards on the same page produce exactly one listing and do not duplicate extraction."""
    mock_store = AsyncMock()
    mock_store.find_seen_ids.return_value = set()
    mock_dedup_cls.return_value = mock_store

    mock_parse_job.return_value = {
        "clean_description": "Clean description",
        "required_languages": ["English"],
    }

    cards_data = [
        {"job_id": "dup_1", "title": "Developer A", "company": "Acme"},
        {"job_id": "dup_1", "title": "Developer A", "company": "Acme"},
        {"job_id": "dup_2", "title": "Developer B", "company": "Beta"},
        {"job_id": "dup_2", "title": "Developer B", "company": "Beta"},
    ]
    page = _create_mock_page(cards_data)
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = (browser, context)
    cm.__aexit__.return_value = None
    mock_managed_browser.return_value = cm

    state = _base_state()
    result = await search_node(state)

    assert len(result["job_listings"]) == 2
    assert [j["job_id"] for j in result["job_listings"]] == ["dup_1", "dup_2"]


@pytest.mark.asyncio
@patch("jobapply.nodes.search.asyncio.sleep", new_callable=AsyncMock)
@patch("jobapply.nodes.search.DeduplicationStore")
@patch("jobapply.nodes.search.parse_job_description")
@patch("jobapply.nodes.search.managed_browser")
async def test_deterministic_exclusions_persisted_and_transient_failures_not_persisted(
    mock_managed_browser,
    mock_parse_job,
    mock_dedup_cls,
    mock_search_sleep,
):
    """Applied, senior, and language exclusions are persisted via bulk upsert; transient DOM failures are not."""
    mock_store = AsyncMock()
    mock_store.find_seen_ids.return_value = set()
    mock_dedup_cls.return_value = mock_store

    mock_parse_job.side_effect = [
        # for lang_excluded
        {"clean_description": "Must speak German fluently", "required_languages": ["German"]},
        # for valid_job
        {"clean_description": "Valid Python role", "required_languages": ["English"]},
    ]

    cards_data = [
        {"job_id": "already_applied_1", "title": "Dev 1", "applied": "Applied 2 days ago"},
        {"job_id": "senior_1", "title": "Senior Staff Architect", "company": "BigCo"},
        {"job_id": "lang_1", "title": "Dev 2", "company": "EuroCorp"},
        {"job_id": "dom_fail_1", "title": "Dev 3", "dom_fail": True},
        {"job_id": "valid_1", "title": "Software Developer", "company": "GoodCorp"},
    ]
    page = _create_mock_page(cards_data)
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = (browser, context)
    cm.__aexit__.return_value = None
    mock_managed_browser.return_value = cm

    state = _base_state()
    result = await search_node(state)

    # In single-shot search->qualification, search extracts non-senior non-applied cards for qualification
    assert len(result["job_listings"]) == 2
    extracted_ids = {j["job_id"] for j in result["job_listings"]}
    assert extracted_ids == {"lang_1", "valid_1"}

    # Verify search exclusions (applied and senior) persisted in bulk
    mock_store.mark_seen_many.assert_awaited_once()
    persisted_items = mock_store.mark_seen_many.call_args[0][0]
    persisted_ids = {item["job_id"] for item in persisted_items}
    assert "already_applied_1" in persisted_ids
    assert "senior_1" in persisted_ids
    assert "dom_fail_1" not in persisted_ids
    assert "valid_1" not in persisted_ids

    # Returned seen_job_ids contains search-excluded IDs so they are not re-inspected
    assert "already_applied_1" in result["seen_job_ids"]
    assert "senior_1" in result["seen_job_ids"]
    assert "valid_1" not in result["seen_job_ids"]


@pytest.mark.asyncio
async def test_concurrent_callers_share_single_initialization_success_and_failure():
    """Concurrent callers share a single in-flight initialization on success and failure without unhandled future warnings."""
    await DeduplicationStore.close()
    store = object.__new__(DeduplicationStore)
    mock_coll = MagicMock()

    # 1. Concurrent Success
    delay_event = asyncio.Event()

    async def delayed_create_index(key, unique=True):
        await delay_event.wait()
        return "job_id_1"

    mock_coll.create_index = MagicMock(side_effect=delayed_create_index)
    store.collection = mock_coll

    task1 = asyncio.create_task(store.ensure_indexes())
    task2 = asyncio.create_task(store.ensure_indexes())

    await asyncio.sleep(0.01)
    delay_event.set()

    res1, res2 = await asyncio.gather(task1, task2)
    assert res1 is True
    assert res2 is True
    assert mock_coll.create_index.call_count == 1

    # 2. Concurrent Failure
    await DeduplicationStore.close()
    mock_coll.create_index.reset_mock()
    fail_event = asyncio.Event()

    async def delayed_failing_index(key, unique=True):
        await fail_event.wait()
        raise RuntimeError("DB connection failed with password=topsecret123")

    mock_coll.create_index = MagicMock(side_effect=delayed_failing_index)
    store.collection = mock_coll

    fail_task1 = asyncio.create_task(store.ensure_indexes())
    fail_task2 = asyncio.create_task(store.ensure_indexes())

    await asyncio.sleep(0.01)
    fail_event.set()

    with pytest.raises(DeduplicationStoreError) as exc_info1:
        await fail_task1
    with pytest.raises(DeduplicationStoreError) as exc_info2:
        await fail_task2

    assert "topsecret123" not in str(exc_info1.value)
    assert "topsecret123" not in str(exc_info2.value)
    assert mock_coll.create_index.call_count == 1

    # Close resets state cleanly
    await DeduplicationStore.close()


@pytest.mark.asyncio
async def test_index_failure_stops_dependent_operations_and_sanitizes_without_credentials():
    """Index failure raises DeduplicationStoreError with sanitized error and stops dependent calls without DB retry."""
    await DeduplicationStore.close()
    store = object.__new__(DeduplicationStore)
    mock_coll = MagicMock()
    mock_coll.create_index.side_effect = RuntimeError("Mongo auth error with password=my_secret_pw")
    store.collection = mock_coll

    # First call raises sanitized DeduplicationStoreError
    with pytest.raises(DeduplicationStoreError) as exc_info:
        await store.ensure_indexes()
    assert "my_secret_pw" not in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)
    assert mock_coll.create_index.call_count == 1

    # Dependent operations must stop and raise the same cached error without retrying create_index
    with pytest.raises(DeduplicationStoreError):
        await store.is_seen("job_123")
    assert mock_coll.create_index.call_count == 1

    with pytest.raises(DeduplicationStoreError):
        await store.find_seen_ids(["job_123"])
    assert mock_coll.create_index.call_count == 1

    with pytest.raises(DeduplicationStoreError):
        await store.mark_seen("job_123", {"title": "Dev"})
    assert mock_coll.create_index.call_count == 1

    with pytest.raises(DeduplicationStoreError):
        await store.mark_seen_many([{"job_id": "job_123"}])
    assert mock_coll.create_index.call_count == 1

    with pytest.raises(DeduplicationStoreError):
        await store.get_daily_count()
    assert mock_coll.create_index.call_count == 1

    await DeduplicationStore.close()


@pytest.mark.asyncio
async def test_canonicalize_job_id_long_id_collision_resistance():
    """Distinct long job IDs produce distinct collision-resistant canonical IDs and distinct bulk operations."""
    await DeduplicationStore.close()
    store = object.__new__(DeduplicationStore)
    mock_coll = AsyncMock()
    store.collection = mock_coll

    # Normal short ID
    assert canonicalize_job_id("normal_job_123") == "normal_job_123"

    # Distinct long IDs with identical prefix
    long_prefix = "https://www.linkedin.com/jobs/view/long_path_prefix_" + "x" * 100
    long_id_1 = long_prefix + "_variant_1"
    long_id_2 = long_prefix + "_variant_2"

    c_id_1 = canonicalize_job_id(long_id_1)
    c_id_2 = canonicalize_job_id(long_id_2)

    # Both <= 128 chars and NOT equal
    assert len(c_id_1) <= 128
    assert len(c_id_2) <= 128
    assert c_id_1 != c_id_2

    # Two distinct long IDs produce 2 bulk operations
    await store.mark_seen_many(
        [
            {"job_id": long_id_1, "title": "Dev 1"},
            {"job_id": long_id_2, "title": "Dev 2"},
        ]
    )
    ops = mock_coll.bulk_write.call_args[0][0]
    assert len(ops) == 2
    assert {ops[0]._filter["job_id"], ops[1]._filter["job_id"]} == {c_id_1, c_id_2}

    # Duplicate identical long IDs deduplicate to 1 bulk operation
    mock_coll.bulk_write.reset_mock()
    await store.mark_seen_many(
        [
            {"job_id": long_id_1, "title": "Dev 1"},
            {"job_id": long_id_1, "title": "Dev 1 Updated"},
        ]
    )
    ops_dup = mock_coll.bulk_write.call_args[0][0]
    assert len(ops_dup) == 1
    assert ops_dup[0]._filter["job_id"] == c_id_1

    # Authoritative mark_seen parameter
    await store.mark_seen("auth_job_1", {"job_id": "impostor_job_2", "title": "Dev"})
    doc = mock_coll.update_one.call_args[0][1]["$set"]
    assert doc["job_id"] == "auth_job_1"
    assert mock_coll.update_one.call_args[0][0] == {"job_id": "auth_job_1"}

    await DeduplicationStore.close()


def test_deeply_recursive_metadata_bounding_and_redaction():
    """Genuinely recursive bounding enforces depth caps, dict key limits, list limits, key lengths, and secret redaction."""
    hostile_structure = {
        "job_id": "job_123",
        "api_key": "sk-secret-key-12345",
        "level1": {
            "token": "bearer-secret-token",
            "long_key_" + "k" * 200: "value",
            "level2": {
                "password": "supersecretpassword",
                "level3": {
                    "level4": {
                        "level5_should_truncate": {"deep": "value"},
                    },
                },
            },
            "giant_list": [
                {"item_key": f"val_{i}", "secret_key": "raw_secret"} for i in range(150)
            ],
        },
        "description": "Short description",
    }

    bounded = bound_and_redact_metadata(hostile_structure)

    # Top-level secrets redacted
    assert bounded["api_key"] == "[REDACTED]"

    # Nested exact sensitive keys redacted
    assert bounded["level1"]["token"] == "[REDACTED]"
    assert bounded["level1"]["level2"]["password"] == "[REDACTED]"

    # Long dict keys bounded
    long_keys = [k for k in bounded["level1"].keys() if k.startswith("long_key_")]
    assert len(long_keys) == 1
    assert len(long_keys[0]) <= 64

    # Deep nesting truncated past max depth boundary
    assert (
        bounded["level1"]["level2"]["level3"]["level4"]["level5_should_truncate"]
        == "[TRUNCATED_DEPTH]"
    )

    # Giant list capped to max items
    assert len(bounded["level1"]["giant_list"]) <= 50
    assert bounded["level1"]["giant_list"][0]["secret_key"] == "[REDACTED]"


@pytest.mark.asyncio
@patch("jobapply.nodes.qualification.get_llm")
@patch("jobapply.nodes.qualification.DeduplicationStore")
async def test_qualification_exclusion_persistence_failure_returns_immutable_state(
    mock_store_cls,
    mock_get_llm,
):
    """Persistence failure in qualification records bounded generic error without leaking secrets and returns state immutably."""
    mock_store = AsyncMock()
    mock_store.mark_seen.side_effect = RuntimeError("Connection dropped password=my_db_pass_999")
    mock_store_cls.return_value = mock_store

    initial_seen = {"existing_1"}
    initial_errors = ["initial error"]
    state = _base_state(
        current_job={
            "job_id": "senior_excl",
            "title": "Senior Director of AI",
            "company": "MegaCorp",
            "location": "Remote",
            "url": "https://linkedin.com/jobs/view/senior_excl",
        },
        seen_job_ids=initial_seen,
        errors=initial_errors,
    )

    result = await qualification_node(state)

    # Initial collections not mutated
    assert initial_seen == {"existing_1"}
    assert initial_errors == ["initial error"]

    # Result state updated immutably
    assert result["qualification_result"]["qualified"] is False
    assert "senior_excl" in result["seen_job_ids"]
    assert result["not_qualified_jobs_count"] == 1
    assert len(result["errors"]) == 2
    assert "my_db_pass_999" not in result["errors"][-1]
    assert "RuntimeError" in result["errors"][-1]
    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
@patch("jobapply.nodes.search.asyncio.sleep", new_callable=AsyncMock)
@patch("jobapply.nodes.search.managed_browser")
async def test_account_safety_pause_in_search_preserves_safety_guards_and_immutability(
    mock_managed_browser, mock_search_sleep
):
    """Account-safety barrier in search produces paused state without mutating input collections."""
    detection = AccountSafetyDetection(
        detected=True,
        barrier_type=AccountSafetyBarrierType.CHECKPOINT,
        stage="search_navigation",
        reason="Security challenge detected",
        url="https://www.linkedin.com/checkpoint/challenge",
        resume_instructions="Solve security challenge",
    )

    page = AsyncMock()
    page.is_closed = MagicMock(return_value=False)
    page.close = AsyncMock()
    page.goto.side_effect = AccountSafetyBarrierError(detection)
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = (browser, context)
    cm.__aexit__.return_value = None
    mock_managed_browser.return_value = cm

    initial_seen = {"seen_1"}
    initial_logs = ["starting search"]
    state = _base_state(seen_job_ids=initial_seen, logs=initial_logs)

    result = await search_node(state)

    assert result["account_safety_paused"] is True
    assert result["account_safety_barrier_type"] == "checkpoint"
    assert result["search_failed"] is False
    assert initial_seen == {"seen_1"}
    assert initial_logs == ["starting search"]
    assert result["seen_job_ids"] is not initial_seen
    page.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("jobapply.nodes.search.asyncio.sleep", new_callable=AsyncMock)
@patch("jobapply.nodes.search.managed_browser")
async def test_search_node_immutability_on_error_branches(mock_managed_browser, mock_search_sleep):
    """Search node preserves input state immutability on infrastructure error branch."""
    page = AsyncMock()
    page.is_closed = MagicMock(return_value=False)
    page.goto.side_effect = RuntimeError("Playwright network crash")
    context = AsyncMock()
    context.new_page.return_value = page
    browser = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = (browser, context)
    cm.__aexit__.return_value = None
    mock_managed_browser.return_value = cm

    initial_seen = {"seen_test"}
    initial_errors = ["prev error"]
    initial_logs = ["prev log"]
    initial_listings = [{"job_id": "old"}]
    state = _base_state(
        seen_job_ids=initial_seen,
        errors=initial_errors,
        logs=initial_logs,
        job_listings=initial_listings,
    )

    result = await search_node(state)

    assert result["search_failed"] is True
    # Input state unchanged
    assert initial_seen == {"seen_test"}
    assert initial_errors == ["prev error"]
    assert initial_logs == ["prev log"]
    assert initial_listings == [{"job_id": "old"}]
    # Output state is separate
    assert result["seen_job_ids"] is not initial_seen
    assert result["errors"] is not initial_errors
    assert result["logs"] is not initial_logs
    assert len(result["errors"]) == 2
