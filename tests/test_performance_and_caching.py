"""Tests for Task 8: LLM/database/browser performance, caching, client reuse, and bounds."""

import asyncio
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobapply.models.job import CombinedJobAnalysis
from jobapply.nodes.qualification import qualification_node
from jobapply.nodes.search import get_loaded_card_ids, load_search_cards_adaptively, search_node
from jobapply.settings import Settings
from jobapply.state import JobApplyState
from jobapply.utils.attempts import AttemptRepository, QuotaRepository
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.mongo import MongoClientManager
from jobapply.utils.prompts import (
    TRUNCATION_MARKER,
    get_cover_letter_prompt,
    get_job_parser_prompt,
    get_qualification_prompt,
    get_resume_edit_prompt,
    get_urgency_check_prompt,
    truncate_head_tail,
)
from jobapply.utils.source_cache import (
    SourceCache,
    reset_source_cache,
)
from jobapply.utils.telegram_storage import (
    NotificationOutboxRepository,
    TelegramPersistenceManager,
    TelegramRepository,
)


@pytest.fixture(autouse=True)
async def cleanup_mongo_and_cache_state():
    yield
    await DeduplicationStore.close()
    await AttemptRepository.close()
    await QuotaRepository.close()
    await TelegramPersistenceManager.close()
    await MongoClientManager.close()
    reset_source_cache()


# ---------------------------------------------------------------------------
# 1. LLM Call Count & Combined Extraction Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_node_makes_zero_llm_calls():
    """Verify production search_node extracts job cards without making any LLM requests."""
    fake_state: JobApplyState = {
        "search_queries": ["Python Developer"],
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": 1,
        "dry_run": True,
        "seen_job_ids": set(),
        "job_listings": [],
        "current_job_index": 0,
        "current_job": None,
    }

    mock_card = AsyncMock()
    mock_card.get_attribute = AsyncMock(
        side_effect=lambda attr: "123456" if attr == "data-occludable-job-id" else None
    )
    mock_card.evaluate = AsyncMock(return_value=[])
    mock_card.scroll_into_view_if_needed = AsyncMock()

    mock_link = AsyncMock()
    mock_link.get_attribute = AsyncMock(
        side_effect=lambda attr: "Python Developer" if attr == "aria-label" else None
    )
    mock_link.inner_text = AsyncMock(return_value="Python Developer")
    mock_link.click = AsyncMock()

    mock_company = AsyncMock()
    mock_company.inner_text = AsyncMock(return_value="TechCorp")

    mock_location = AsyncMock()
    mock_location.inner_text = AsyncMock(return_value="Remote")

    mock_desc = AsyncMock()
    mock_desc.inner_text = AsyncMock(return_value="We are seeking a Python Developer.")

    async def mock_query_selector(selector):
        if "a.job-card-container__link" in selector:
            return mock_link
        if "subtitle" in selector:
            return mock_company
        if "metadata" in selector:
            return mock_location
        if "123456" in selector:
            return mock_card
        return None

    mock_card.query_selector = AsyncMock(side_effect=mock_query_selector)
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
    mock_page.query_selector_all = AsyncMock(return_value=[mock_card])
    mock_page.query_selector = AsyncMock(side_effect=mock_query_selector)
    mock_page.wait_for_selector = AsyncMock(
        side_effect=lambda sel, **kwargs: mock_desc if "description" in sel else AsyncMock()
    )

    with (
        patch("jobapply.nodes.search.managed_browser") as mock_mb,
        patch("jobapply.nodes.search.get_llm") as mock_get_llm,
        patch("jobapply.nodes.search.guard_page_account_safety", new_callable=AsyncMock),
        patch("jobapply.nodes.search.load_search_cards_adaptively", new_callable=AsyncMock),
        patch.object(
            DeduplicationStore, "find_seen_ids", new_callable=AsyncMock, return_value=set()
        ),
    ):
        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = AsyncMock()

        class AsyncContextManager:
            async def __aenter__(self):
                return mock_browser, mock_context

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_mb.return_value = AsyncContextManager()

        result = await search_node(fake_state)

        # Confirm exactly 0 LLM calls made during search
        mock_get_llm.assert_not_called()
        assert len(result["job_listings"]) == 1
        job = result["job_listings"][0]
        assert job["job_id"] == "123456"
        assert job["title"] == "Python Developer"
        assert job["company"] == "TechCorp"
        assert job["description"] == "We are seeking a Python Developer."


@pytest.mark.asyncio
async def test_qualification_node_makes_single_combined_llm_call(tmp_path):
    """Verify qualification_node performs exactly 1 structured LLM call and returns structured fields."""
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text(
        "name: Test Candidate\nskills:\n  - Python\n  - FastAPI\n", encoding="utf-8"
    )

    current_job = {
        "job_id": "789012",
        "title": "Backend Engineer",
        "company": "Cloud Inc",
        "location": "San Francisco, CA",
        "url": "https://www.linkedin.com/jobs/view/789012",
        "description": "About the job\nTitle: Backend Engineer\nLooking for Python developer in SF. Remote OK. 6 months.",
    }

    fake_state: JobApplyState = {
        "current_job": current_job,
        "seen_job_ids": set(),
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
    }

    llm_mock = AsyncMock()
    combined_response = (
        '{"qualified": true, "score": 0.85, "reasoning": "Strong match.", '
        '"key_matches": ["Python", "Backend"], "gaps": [], '
        '"job_summary": "Develop backend services.", '
        '"parsed_location": "San Francisco, CA", "duration": "6 months", '
        '"work_type": "Remote", "responsibilities": ["Build APIs"], '
        '"requirements": ["Python proficiency"], "required_languages": ["English"], '
        '"clean_description": "Looking for Python developer in SF. Remote OK. 6 months."}'
    )
    llm_mock.ainvoke.return_value = MagicMock(content=combined_response)

    with (
        patch("jobapply.settings.get_settings") as mock_settings,
        patch("jobapply.utils.source_cache.get_settings") as mock_cache_settings,
        patch("jobapply.nodes.qualification.get_llm", return_value=llm_mock) as mock_get_llm,
        patch.object(DeduplicationStore, "mark_seen", new_callable=AsyncMock) as mock_mark_seen,
    ):
        settings_obj = Settings(
            data_dir=str(tmp_path),
            anthropic_api_key="test-key",
            gemini_api_key="test-key",
            mongodb_url="mongodb://localhost:27017",
            qualification_threshold=0.7,
            max_clean_description_chars=3500,
        )
        mock_settings.return_value = settings_obj
        mock_cache_settings.return_value = settings_obj
        reset_source_cache()

        result = await qualification_node(fake_state)

        # Verify LLM was invoked exactly once
        assert llm_mock.ainvoke.call_count == 1
        mock_get_llm.assert_called_once()
        call_kwargs = mock_get_llm.call_args[1]
        assert call_kwargs["response_json_schema"] == CombinedJobAnalysis.model_json_schema()
        assert call_kwargs["max_output_tokens"] == 4096

        # Check qualification output
        assert result["qualification_result"]["qualified"] is True
        assert result["qualification_result"]["score"] == 0.85
        assert result["qualified_jobs_count"] == 1

        # Check immutable current_job update with structured fields
        updated_job = result["current_job"]
        assert updated_job["duration"] == "6 months"
        assert updated_job["work_type"] == "Remote"
        assert updated_job["responsibilities"] == ["Build APIs"]
        assert (
            updated_job["description"] == "Looking for Python developer in SF. Remote OK. 6 months."
        )

        # Check deduplication store persistence
        mock_mark_seen.assert_awaited_once()
        persisted_data = mock_mark_seen.call_args[0][1]
        assert persisted_data["status"] == "qualified"
        assert persisted_data["qualification_score"] == 0.85
        assert persisted_data["duration"] == "6 months"
        assert persisted_data["work_type"] == "Remote"


@pytest.mark.asyncio
async def test_language_filtering_phrase_in_raw_description_triggers_exactly_one_llm_call_and_rejects(
    tmp_path,
):
    """Verify phrase like 'German is required' receives exactly 1 combined LLM call and is rejected post-analysis."""
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text("name: Test Candidate\nskills: [Python]\n", encoding="utf-8")

    current_job = {
        "job_id": "999888",
        "title": "Software Engineer",
        "company": "Berlin Tech",
        "location": "Berlin, Germany",
        "url": "https://www.linkedin.com/jobs/view/999888",
        "description": "Looking for Python dev. German is required for this role.",
    }

    fake_state: JobApplyState = {
        "current_job": current_job,
        "seen_job_ids": set(),
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
    }

    llm_mock = AsyncMock()
    # LLM returned high score (0.95), claiming qualified = True
    combined_response = (
        '{"qualified": true, "score": 0.95, "reasoning": "Excellent technical fit.", '
        '"key_matches": ["Python"], "gaps": [], '
        '"job_summary": "Build backend software.", '
        '"parsed_location": "Berlin", "duration": "Permanent", '
        '"work_type": "Onsite", "responsibilities": ["Coding"], '
        '"requirements": ["German speaking"], "required_languages": ["German"], '
        '"clean_description": "Looking for Python dev."}'
    )
    llm_mock.ainvoke.return_value = MagicMock(content=combined_response)

    with (
        patch("jobapply.settings.get_settings") as mock_settings,
        patch("jobapply.utils.source_cache.get_settings") as mock_cache_settings,
        patch("jobapply.nodes.qualification.get_llm", return_value=llm_mock),
        patch.object(DeduplicationStore, "mark_seen", new_callable=AsyncMock) as mock_mark_seen,
    ):
        settings_obj = Settings(
            data_dir=str(tmp_path),
            anthropic_api_key="test-key",
            gemini_api_key="test-key",
            mongodb_url="mongodb://localhost:27017",
            qualification_threshold=0.7,
        )
        mock_settings.return_value = settings_obj
        mock_cache_settings.return_value = settings_obj
        reset_source_cache()

        result = await qualification_node(fake_state)

        # Assert exactly one LLM call occurred (NOT rejected pre-LLM)
        assert llm_mock.ainvoke.call_count == 1

        # Assert deterministic post-analysis rejection overriding high score
        assert result["qualification_result"]["qualified"] is False
        assert result["qualification_result"]["score"] == 0.0
        assert "Disallowed required language: German" in result["qualification_result"]["reasoning"]
        assert result["qualified_jobs_count"] == 0
        assert result["not_qualified_jobs_count"] == 1

        # Check deduplication store persistence
        mock_mark_seen.assert_awaited_once()
        persisted_data = mock_mark_seen.call_args[0][1]
        assert persisted_data["status"] == "not_qualified"
        assert persisted_data["qualification_score"] == 0.0
        assert "German" in persisted_data["disallowed_languages"]


# ---------------------------------------------------------------------------
# 2. Description Bounding & Immutability Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_node_bounds_long_raw_description():
    """Verify search_node bounds raw descriptions before adding to job_listings/state."""
    settings = Settings(
        max_job_description_chars=500,
        anthropic_api_key="test-key",
        gemini_api_key="test-key",
        mongodb_url="mongodb://localhost:27017",
    )

    long_dom_desc = "START_DESC_" + ("A" * 3000) + "_END_DESC"

    mock_card = AsyncMock()
    mock_card.get_attribute = AsyncMock(
        side_effect=lambda attr: "9911" if attr == "data-occludable-job-id" else None
    )
    mock_card.evaluate = AsyncMock(return_value=[])
    mock_card.scroll_into_view_if_needed = AsyncMock()

    mock_link = AsyncMock()
    mock_link.get_attribute = AsyncMock(
        side_effect=lambda attr: "Developer" if attr == "aria-label" else None
    )
    mock_link.inner_text = AsyncMock(return_value="Developer")
    mock_link.click = AsyncMock()

    mock_company = AsyncMock()
    mock_company.inner_text = AsyncMock(return_value="BigCorp")

    mock_location = AsyncMock()
    mock_location.inner_text = AsyncMock(return_value="Remote")

    mock_desc = AsyncMock()
    mock_desc.inner_text = AsyncMock(return_value=long_dom_desc)

    async def mock_query_selector(selector):
        if "a.job-card-container__link" in selector:
            return mock_link
        if "subtitle" in selector:
            return mock_company
        if "metadata" in selector:
            return mock_location
        if "9911" in selector:
            return mock_card
        return None

    mock_card.query_selector = AsyncMock(side_effect=mock_query_selector)
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=MagicMock(status=200))
    mock_page.query_selector_all = AsyncMock(return_value=[mock_card])
    mock_page.query_selector = AsyncMock(side_effect=mock_query_selector)
    mock_page.wait_for_selector = AsyncMock(
        side_effect=lambda sel, **kwargs: mock_desc if "description" in sel else AsyncMock()
    )

    with (
        patch("jobapply.settings.get_settings", return_value=settings),
        patch("jobapply.nodes.search.get_settings", return_value=settings),
        patch("jobapply.nodes.search.managed_browser") as mock_mb,
        patch("jobapply.nodes.search.guard_page_account_safety", new_callable=AsyncMock),
        patch("jobapply.nodes.search.load_search_cards_adaptively", new_callable=AsyncMock),
        patch.object(
            DeduplicationStore, "find_seen_ids", new_callable=AsyncMock, return_value=set()
        ),
    ):
        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page
        mock_browser = AsyncMock()

        class AsyncContextManager:
            async def __aenter__(self):
                return mock_browser, mock_context

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_mb.return_value = AsyncContextManager()

        state: JobApplyState = {
            "search_queries": ["Dev"],
            "current_query_index": 0,
            "current_page": 1,
            "pages_per_query": 1,
            "dry_run": True,
            "seen_job_ids": set(),
            "job_listings": [],
            "current_job_index": 0,
            "current_job": None,
        }

        result = await search_node(state)
        extracted = result["job_listings"][0]
        assert len(extracted["description"]) <= 500
        assert TRUNCATION_MARKER in extracted["description"]
        assert "START_DESC_" in extracted["description"]


@pytest.mark.asyncio
async def test_qualification_bounds_oversized_clean_description_and_preserves_immutability(
    tmp_path,
):
    """Verify qualification bounds oversized model clean_description to max_clean_description_chars and preserves input immutability."""
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text("name: Test Candidate\nskills: [Python]\n", encoding="utf-8")

    original_raw_desc = "Original raw description"
    input_current_job = {
        "job_id": "554433",
        "title": "Backend Developer",
        "company": "FastCo",
        "location": "Remote",
        "url": "https://www.linkedin.com/jobs/view/554433",
        "description": original_raw_desc,
    }

    fake_state: JobApplyState = {
        "current_job": input_current_job,
        "seen_job_ids": set(),
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
    }

    oversized_clean_desc = "CLEAN_START_" + ("X" * 5000) + "_CLEAN_END"
    llm_mock = AsyncMock()
    combined_response = (
        '{"qualified": true, "score": 0.88, "reasoning": "Great fit.", '
        '"key_matches": ["Python"], "gaps": [], '
        '"job_summary": "Summary", '
        '"parsed_location": "Remote", "duration": "1 year", '
        '"work_type": "Remote", "responsibilities": ["Coding"], '
        '"requirements": ["Python"], "required_languages": ["English"], '
        f'"clean_description": "{oversized_clean_desc}"}}'
    )
    llm_mock.ainvoke.return_value = MagicMock(content=combined_response)

    settings_obj = Settings(
        data_dir=str(tmp_path),
        max_job_description_chars=1000,
        max_clean_description_chars=500,
        anthropic_api_key="test-key",
        gemini_api_key="test-key",
        mongodb_url="mongodb://localhost:27017",
    )

    with (
        patch("jobapply.settings.get_settings", return_value=settings_obj),
        patch("jobapply.nodes.qualification.get_settings", return_value=settings_obj),
        patch("jobapply.utils.source_cache.get_settings", return_value=settings_obj),
        patch("jobapply.nodes.qualification.get_llm", return_value=llm_mock),
        patch.object(DeduplicationStore, "mark_seen", new_callable=AsyncMock) as mock_mark_seen,
    ):
        result = await qualification_node(fake_state)

        # Confirm model's oversized clean_description was bounded to max_clean_description_chars (500)
        updated_job = result["current_job"]
        assert len(updated_job["description"]) <= 500
        assert TRUNCATION_MARKER in updated_job["description"]

        # Confirm persisted description in MongoDB was also bounded to max_clean_description_chars
        persisted_data = mock_mark_seen.call_args[0][1]
        assert len(persisted_data["description"]) <= 500
        assert TRUNCATION_MARKER in persisted_data["description"]

        # Assert explicit immutability of input dict
        assert input_current_job["description"] == original_raw_desc
        assert "parsed_location" not in input_current_job


# ---------------------------------------------------------------------------
# 3. SourceCache Tests (Digest Participation, Immutability, Bounds)
# ---------------------------------------------------------------------------


def test_source_cache_digest_reuses_parsed_yaml_on_metadata_only_change(tmp_path):
    """Verify SourceCache actively uses digest to reuse parsed YAML on metadata-only/touch updates."""
    cache = SourceCache(max_entries=4)
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text("skills:\n  - Python\n  - Rust\n", encoding="utf-8")

    # First read (populates cache)
    data1, text1 = cache.get_profile(str(profile_file))
    assert data1 == {"skills": ["Python", "Rust"]}

    # Update mtime only (touch file, content unchanged)
    stat_before = profile_file.stat()
    os.utime(str(profile_file), (stat_before.st_atime + 5, stat_before.st_mtime + 5))

    # Second read: with yaml.safe_load patched out, read should succeed because digest matches
    with patch("yaml.safe_load", side_effect=AssertionError("yaml.safe_load should not be called")):
        data2, text2 = cache.get_profile(str(profile_file))
        assert data2 == {"skills": ["Python", "Rust"]}
        assert text2 == text1


def test_source_cache_resume_digest_reuses_raw_string_on_metadata_only_change(tmp_path):
    """Verify SourceCache.read_resume_markdown reuses cached string object on metadata-only change."""
    cache = SourceCache(max_entries=4)
    resume_file = tmp_path / "resume.md"
    long_resume_content = (
        "# Resume of Software Engineer\n\n"
        + "## Summary\nExperienced engineer specializing in distributed systems.\n\n"
        + "## Skills\n"
        + "- Python, FastAPI, asyncio, MongoDB, Redis, Docker, Kubernetes\n" * 20
    )
    resume_file.write_text(long_resume_content, encoding="utf-8")

    # First read (populates cache)
    text1 = cache.get_resume_markdown(str(resume_file))
    assert text1 == long_resume_content

    # Update mtime only (metadata touch, content digest unchanged)
    stat_before = resume_file.stat()
    os.utime(str(resume_file), (stat_before.st_atime + 10, stat_before.st_mtime + 10))

    # Second read: must return identical string reference, proving digest-based reuse
    text2 = cache.get_resume_markdown(str(resume_file))
    assert text2 is text1

    # Invalidate by changing file content
    resume_file.write_text(long_resume_content + "\n## Additional Experience", encoding="utf-8")
    os.utime(str(resume_file), (stat_before.st_atime + 20, stat_before.st_mtime + 20))
    text3 = cache.get_resume_markdown(str(resume_file))
    assert text3 is not text1
    assert "Additional Experience" in text3


def test_source_cache_deepcopy_prevents_mutation_poisoning(tmp_path):
    """Verify mutating returned parsed YAML dictionary does not corrupt the cached copy."""
    cache = SourceCache(max_entries=4)
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text("skills:\n  - Python\n", encoding="utf-8")

    data1, _ = cache.get_profile(str(profile_file))
    # Caller mutates returned nested list
    data1["skills"].append("MUTATION_POISON")

    # Next read must return pristine data
    data2, _ = cache.get_profile(str(profile_file))
    assert "MUTATION_POISON" not in data2["skills"]
    assert data2["skills"] == ["Python"]


def test_source_cache_bounded_capacity(tmp_path):
    """Verify SourceCache does not grow unbounded under many files."""
    cache = SourceCache(max_entries=3)

    files = []
    for i in range(6):
        f = tmp_path / f"resume_{i}.md"
        f.write_text(f"# Resume {i}", encoding="utf-8")
        files.append(str(f))

    for f in files:
        cache.get_resume_markdown(f)

    assert len(cache._entries) <= 3


def test_source_cache_thread_safety(tmp_path):
    """Verify SourceCache is concurrency-safe under multi-threaded reads."""
    cache = SourceCache(max_entries=10)
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text("skills: [Python, Concurrency]\n", encoding="utf-8")

    results = []

    def worker():
        for _ in range(20):
            data, _ = cache.get_profile(str(profile_file))
            results.append(data["skills"][0])

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 100
    assert all(r == "Python" for r in results)


# ---------------------------------------------------------------------------
# 4. Prompt Input Bounding Tests
# ---------------------------------------------------------------------------


def test_truncate_head_tail():
    """Verify truncate_head_tail preserves head, tail, and insertion marker within max_chars."""
    marker = TRUNCATION_MARKER
    short_text = "Short text under bound."
    assert truncate_head_tail(short_text, max_chars=100, marker=marker) == short_text

    long_text = "START_" + ("x" * 200) + "_END"
    max_len = 50
    bounded = truncate_head_tail(long_text, max_chars=max_len, marker=marker)

    assert len(bounded) <= max_len
    assert marker in bounded
    assert bounded.startswith("START_")
    assert bounded.endswith("_END")


def test_prompt_builders_apply_bounds():
    """Verify all prompt formatting functions enforce settings-based character bounds."""
    settings = Settings(
        max_job_description_chars=500,
        max_clean_description_chars=600,
        max_profile_context_chars=500,
        max_resume_context_chars=500,
        anthropic_api_key="test-key",
        gemini_api_key="test-key",
        mongodb_url="mongodb://localhost:27017",
    )

    with patch("jobapply.utils.prompts.get_settings", return_value=settings):
        long_desc = "DESC_START_" + ("D" * 2000) + "_DESC_END"
        long_profile = "PROF_START_" + ("P" * 2000) + "_PROF_END"
        long_resume = "RES_START_" + ("R" * 2000) + "_RES_END"

        job = {"title": "Dev", "company": "Co", "location": "Remote", "description": long_desc}

        qual_prompt = get_qualification_prompt(long_profile, job)
        assert TRUNCATION_MARKER in qual_prompt
        assert "DESC_START_" in qual_prompt
        assert "PROF_START_" in qual_prompt
        assert "600 characters" in qual_prompt

        urgency_prompt = get_urgency_check_prompt(long_resume, long_desc, 0.9)
        assert TRUNCATION_MARKER in urgency_prompt
        assert "RES_START_" in urgency_prompt

        resume_edit_prompt = get_resume_edit_prompt(long_resume, "Proposed edits")
        assert TRUNCATION_MARKER in resume_edit_prompt

        cover_prompt = get_cover_letter_prompt(long_profile, job, ["Python"])
        assert TRUNCATION_MARKER in cover_prompt

        parser_prompt = get_job_parser_prompt(long_desc)
        assert TRUNCATION_MARKER in parser_prompt


# ---------------------------------------------------------------------------
# 5. Shared MongoDB Client Lifecycle & Central Reset
# ---------------------------------------------------------------------------


def test_mongo_client_manager_process_ownership_across_threads_and_loops():
    """Verify MongoClientManager provides the exact same process-owned client instance across threads and loops until explicit close."""
    with patch("jobapply.utils.mongo.AsyncIOMotorClient") as mock_motor:
        mock_client = MagicMock()
        mock_motor.return_value = mock_client

        # Main thread client request
        client_main = MongoClientManager.get_client()
        assert client_main is mock_client
        assert mock_motor.call_count == 1

        # Request client from a separate thread running its own asyncio loop
        client_thread_box = []

        def thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                c = MongoClientManager.get_client()
                client_thread_box.append(c)
            finally:
                loop.close()

        t = threading.Thread(target=thread_target)
        t.start()
        t.join()

        assert len(client_thread_box) == 1
        assert client_thread_box[0] is client_main
        # Still only one constructor call made for the process
        assert mock_motor.call_count == 1


def test_concurrent_close_entrypoints_coordinate_and_block_until_complete():
    """Verify concurrent close callers coordinate via condition variable, block until first close finishes, and close client once."""
    close_started = threading.Event()
    can_finish_close = threading.Event()

    mock_client = MagicMock()

    def blocking_client_close():
        close_started.set()
        can_finish_close.wait(timeout=5)

    mock_client.close.side_effect = blocking_client_close

    with patch("jobapply.utils.mongo.AsyncIOMotorClient", return_value=mock_client):
        # Initialize client and simulate successful index states across repositories
        _ = MongoClientManager.get_client()
        DeduplicationStore._index_succeeded = True
        AttemptRepository._index_succeeded = True
        QuotaRepository._index_succeeded = True
        TelegramPersistenceManager._correlations_index_succeeded = True
        TelegramPersistenceManager._outbox_index_succeeded = True

        thread1_done = threading.Event()
        thread2_done = threading.Event()
        thread3_done = threading.Event()

        def runner1():
            asyncio.run(AttemptRepository.close())
            thread1_done.set()

        def runner2():
            asyncio.run(QuotaRepository.close())
            thread2_done.set()

        def runner3():
            asyncio.run(MongoClientManager.close())
            thread3_done.set()

        t1 = threading.Thread(target=runner1)
        t2 = threading.Thread(target=runner2)
        t3 = threading.Thread(target=runner3)

        t1.start()
        # Wait until thread 1 has begun executing close and is blocked inside mock_client.close
        assert close_started.wait(timeout=5)

        # Thread 1 is currently blocked inside client.close
        assert not thread1_done.is_set()

        # Start thread 2 and thread 3 while thread 1 is actively blocking inside client.close
        t2.start()
        t3.start()

        # Give threads 2 & 3 time to encounter the active close state
        time.sleep(0.1)

        # Assert threads 2 & 3 are still waiting and have NOT returned prematurely
        assert not thread2_done.is_set()
        assert not thread3_done.is_set()

        # Unblock thread 1
        can_finish_close.set()

        # All threads must now complete successfully
        t1.join(timeout=5)
        t2.join(timeout=5)
        t3.join(timeout=5)

        assert thread1_done.is_set()
        assert thread2_done.is_set()
        assert thread3_done.is_set()

        # Verify all repository states reset
        assert DeduplicationStore._index_succeeded is False
        assert AttemptRepository._index_succeeded is False
        assert QuotaRepository._index_succeeded is False
        assert TelegramPersistenceManager._correlations_index_succeeded is False
        assert TelegramPersistenceManager._outbox_index_succeeded is False

        # Verify client.close was called exactly once
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_shared_mongo_client_lifecycle_and_all_repo_close_entrypoints():
    """Verify all close entrypoints reset all repo index states and close client at most once."""
    with patch("jobapply.utils.mongo.AsyncIOMotorClient") as mock_motor:
        mock_client = MagicMock()
        mock_motor.return_value = mock_client

        # Initialize shared client
        client1 = MongoClientManager.get_client()
        assert client1 is mock_client

        # Simulate successful index initialization across all repositories
        DeduplicationStore._index_succeeded = True
        AttemptRepository._index_succeeded = True
        QuotaRepository._index_succeeded = True
        TelegramPersistenceManager._correlations_index_succeeded = True
        TelegramPersistenceManager._outbox_index_succeeded = True

        # Calling close on AttemptRepository delegates to centralized reset
        await AttemptRepository.close()

        # All repository index states must be reset together
        assert DeduplicationStore._index_succeeded is False
        assert AttemptRepository._index_succeeded is False
        assert QuotaRepository._index_succeeded is False
        assert TelegramPersistenceManager._correlations_index_succeeded is False
        assert TelegramPersistenceManager._outbox_index_succeeded is False

        # Client was closed exactly once
        mock_client.close.assert_called_once()

        # Calling close again on other entrypoints is idempotent and non-recursive
        await QuotaRepository.close()
        await TelegramRepository.close()
        await NotificationOutboxRepository.close()
        await DeduplicationStore.close()
        await TelegramPersistenceManager.close()
        await MongoClientManager.close()

        # Still closed exactly once
        mock_client.close.assert_called_once()

        # Next lifecycle creates and shares a single new client
        mock_client2 = MagicMock()
        mock_motor.return_value = mock_client2

        dedup2 = DeduplicationStore()
        attempts2 = AttemptRepository()
        quota2 = QuotaRepository()
        telegram2 = TelegramRepository()

        assert dedup2.client is mock_client2
        assert attempts2.client is mock_client2
        assert quota2.client is mock_client2
        assert telegram2.client is mock_client2
        assert mock_motor.call_count == 2


@pytest.mark.asyncio
async def test_repositories_independent_index_states():
    """Verify repository index failure in one repository does not impact another repository during normal run."""
    with patch("jobapply.utils.mongo.AsyncIOMotorClient"):
        await MongoClientManager.close()

        dedup = DeduplicationStore()
        attempts = AttemptRepository()

        # Mock dedup index creation failure
        dedup.collection = MagicMock()
        dedup.collection.create_index.side_effect = Exception("Mongo connection lost")

        # Mock attempts index creation success
        attempts.collection = MagicMock()
        attempts.collection.create_index = AsyncMock(return_value="index_created")

        with pytest.raises(Exception):
            await dedup.ensure_indexes()

        # Confirm dedup recorded error state
        assert DeduplicationStore._index_succeeded is False
        assert DeduplicationStore._index_error is not None

        # AttemptRepository index should succeed independently
        success = await attempts.ensure_indexes()
        assert success is True
        assert AttemptRepository._index_succeeded is True
        assert AttemptRepository._index_error is None


# ---------------------------------------------------------------------------
# 6. Hardened Browser Adaptive Card Loading Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adaptive_loader_stops_early_on_stable_rounds():
    """Verify load_search_cards_adaptively stops early when rendered card IDs stabilize."""
    mock_page = AsyncMock()

    card1 = AsyncMock()
    card1.get_attribute = AsyncMock(return_value="101")
    card2 = AsyncMock()
    card2.get_attribute = AsyncMock(return_value="102")

    mock_page.query_selector_all = AsyncMock(return_value=[card1, card2])

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await load_search_cards_adaptively(
            mock_page,
            max_scroll_rounds=10,
            stability_rounds=2,
            scroll_delay_seconds=0.1,
        )

        assert mock_page.evaluate.call_count == 2
        assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_adaptive_loader_continues_when_new_cards_appear():
    """Verify load_search_cards_adaptively continues scrolling as new card IDs are loaded."""
    mock_page = AsyncMock()

    card1 = AsyncMock()
    card1.get_attribute = AsyncMock(return_value="101")
    card2 = AsyncMock()
    card2.get_attribute = AsyncMock(return_value="102")
    card3 = AsyncMock()
    card3.get_attribute = AsyncMock(return_value="103")

    mock_page.query_selector_all = AsyncMock(
        side_effect=[
            [card1],
            [card1, card2],
            [card1, card2, card3],
            [card1, card2, card3],
            [card1, card2, card3],
        ]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await load_search_cards_adaptively(
            mock_page,
            max_scroll_rounds=5,
            stability_rounds=2,
            scroll_delay_seconds=0.1,
        )

        assert mock_page.evaluate.call_count == 4


@pytest.mark.asyncio
async def test_adaptive_card_loader_propagates_selector_or_browser_failure():
    """Verify get_loaded_card_ids and loader do NOT silently swallow selector or browser errors."""
    mock_page = AsyncMock()
    mock_page.query_selector_all.side_effect = RuntimeError("Browser context destroyed")

    with pytest.raises(RuntimeError) as exc_info:
        await get_loaded_card_ids(mock_page)
    assert "Browser context destroyed" in str(exc_info.value)

    with pytest.raises(RuntimeError) as exc_info2:
        await load_search_cards_adaptively(mock_page)
    assert "Browser context destroyed" in str(exc_info2.value)
