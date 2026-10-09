"""Regression checks for slow LinkedIn results and qualification failures."""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from jobapply.nodes.search import (
    EMPTY_RESULTS_SELECTOR,
    search_node,
    wait_for_search_results,
)


@pytest.mark.asyncio
async def test_waits_for_cards_after_navigation():
    page = AsyncMock()
    page.query_selector_all.return_value = [object()]
    assert await wait_for_search_results(page, timeout_ms=60000)
    assert page.wait_for_selector.await_args.kwargs["timeout"] == 60000
    assert page.wait_for_selector.await_args.kwargs["state"] == "visible"
    page.query_selector.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_requires_explicit_banner():
    page = AsyncMock()
    page.query_selector_all.return_value = []
    page.query_selector.return_value = object()
    assert not await wait_for_search_results(page, timeout_ms=60000)
    page.query_selector.assert_awaited_once_with(EMPTY_RESULTS_SELECTOR)


@pytest.mark.asyncio
async def test_results_disappearing_is_not_empty():
    page = AsyncMock()
    page.query_selector_all.return_value = []
    page.query_selector.return_value = None
    with pytest.raises(RuntimeError, match="not marked exhausted"):
        await wait_for_search_results(page, timeout_ms=60000)


@pytest.mark.asyncio
async def test_slow_page_does_not_exhaust_query(monkeypatch):
    page = AsyncMock()
    page.goto.return_value = None
    page.wait_for_selector.side_effect = PlaywrightTimeoutError("still loading")
    context = AsyncMock()

    @asynccontextmanager
    async def browser():
        yield None, context

    monkeypatch.setattr("jobapply.nodes.search.managed_browser", browser)
    monkeypatch.setattr("jobapply.nodes.search.get_linkedin_page", AsyncMock(return_value=page))
    guard = AsyncMock()
    monkeypatch.setattr("jobapply.nodes.search.guard_page_account_safety", guard)
    state = {
        "search_queries": ["Machine Learning Engineer"],
        "current_query_index": 0,
        "current_page": 2,
        "pages_per_query": 100,
        "seen_job_ids": {"old"},
        "errors": [],
    }
    result = await search_node(state)
    assert result["search_failed"] is True
    assert result["query_exhausted"] is False
    assert "did not finish loading" in result["errors"][0]
    assert state["errors"] == []
    assert result["seen_job_ids"] == {"old"}
    assert guard.await_count == 2
    page.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_qualification_configuration_failure_is_caught(monkeypatch):
    from jobapply.nodes.qualification import qualification_node

    monkeypatch.setattr("jobapply.nodes.qualification.get_cached_profile", lambda: ({}, "profile"))
    def fail(**kwargs):
        raise RuntimeError("ANTHROPIC_API_KEY is missing")
    monkeypatch.setattr("jobapply.nodes.qualification.get_llm", fail)
    state = {
        "current_job": {
            "job_id": "test",
            "title": "Machine Learning Engineer",
            "company": "Ampace",
            "description": "Python ML",
        },
        "seen_job_ids": set(),
        "errors": [],
    }
    result = await qualification_node(state)
    assert result["qualification_result"]["qualified"] is False
    assert "ANTHROPIC_API_KEY is missing" in result["errors"][0]
    assert result["seen_job_ids"] == set()
    assert state["errors"] == []
