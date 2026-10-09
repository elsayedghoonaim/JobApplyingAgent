import pytest

from jobapply.graph import (
    route_after_approval,
    route_after_generation,
    route_after_qualification,
    route_post_processing,
    route_select_next_job,
)


def test_route_select_next_job():
    state = {"search_failed": True}
    assert route_select_next_job(state) == "notification_node"

    # Max jobs limit reached
    state = {
        "max_jobs_to_evaluate": 5,
        "jobs_evaluated_count": 5,
    }
    assert route_select_next_job(state) == "notification_node"

    # Job selected
    state = {
        "max_jobs_to_evaluate": 5,
        "jobs_evaluated_count": 3,
        "current_job": {"job_id": "1"},
    }
    assert route_select_next_job(state) == "qualification_node"

    # No job selected, should fetch next page
    state = {
        "max_jobs_to_evaluate": 5,
        "jobs_evaluated_count": 3,
        "current_job": None,
        "current_page": 1,
        "pages_per_query": 3,
        "current_query_index": 0,
        "search_queries": ["a", "b"],
    }
    assert route_select_next_job(state) == "increment_page_node"

    # No job selected, next query
    state = {
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 3,
        "current_job": None,
        "current_page": 3,
        "pages_per_query": 3,
        "current_query_index": 0,
        "search_queries": ["a", "b"],
    }
    assert route_select_next_job(state) == "increment_query_node"

    # Everything exhausted
    state = {
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 3,
        "current_job": None,
        "current_page": 3,
        "pages_per_query": 3,
        "current_query_index": 1,
        "search_queries": ["a", "b"],
    }
    assert route_select_next_job(state) == "notification_node"


def test_route_after_qualification():
    state = {"qualification_result": {"qualified": True}}
    assert route_after_qualification(state) == "generation_node"

    state = {"qualification_result": {"qualified": False}}
    assert route_after_qualification(state) == "select_next_job_node"


def test_route_after_generation():
    assert route_after_generation({"application_status": "failed"}) == "select_next_job_node"
    assert route_after_generation({"account_safety_paused": True}) == "notification_node"

    state = {"edits_urgent": True}
    assert route_after_generation(state) == "execution_node"

    state = {"edits_urgent": False}
    assert route_after_generation(state) == "execution_node"


def test_route_after_approval():
    state = {"approval_status": "skip"}
    assert route_after_approval(state) == "select_next_job_node"

    state = {"approval_status": "approved"}
    assert route_after_approval(state) == "execution_node"


def test_route_post_processing():
    # Session cap reached
    state = {
        "applications_count": 5,
        "max_applications": 5,
        "daily_applications_count": 2,
        "daily_application_cap": 10,
    }
    assert route_post_processing(state) == "notification_node"

    # Daily cap reached
    state = {
        "applications_count": 2,
        "max_applications": 5,
        "daily_applications_count": 10,
        "daily_application_cap": 10,
    }
    assert route_post_processing(state) == "notification_node"

    # Continues session
    state = {
        "applications_count": 2,
        "max_applications": 5,
        "daily_applications_count": 3,
        "daily_application_cap": 10,
    }
    assert route_post_processing(state) == "select_next_job_node"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failed", "paused"])
async def test_compiled_graph_generation_routes(monkeypatch, outcome):
    from jobapply import graph as module

    async def search(state):
        return {"query_exhausted": True}

    async def select(state):
        return {
            "current_job": None
            if state.get("application_status") == "failed"
            else {"job_id": "test", "title": "ML Engineer"}
        }

    async def qualify(state):
        return {"qualification_result": {"qualified": True}}

    async def generate(state):
        return {
            "application_status": "failed" if outcome == "failed" else None,
            "account_safety_paused": outcome == "paused",
        }

    async def execute(state):
        return {"applications_count": 1}

    async def notify(state):
        return {}

    for name, node in [
        ("search_node", search),
        ("select_next_job_node", select),
        ("qualification_node", qualify),
        ("generation_node", generate),
        ("execution_node", execute),
        ("notification_node", notify),
    ]:
        monkeypatch.setattr(module, name, node)
    graph = module.build_graph().compile()
    state = {
        "search_queries": ["ML"],
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": 1,
        "max_applications": 1,
        "applications_count": 0,
    }
    visited = []
    async for update in graph.astream(state):
        visited.extend(update)
    assert visited[-1] == "notification_node"
    if outcome == "success":
        assert "execution_node" in visited
    else:
        assert "execution_node" not in visited
    if outcome == "failed":
        assert visited.count("select_next_job_node") == 2
