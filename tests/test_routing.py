import pytest
from jobapply.graph import (
    route_select_next_job,
    route_after_qualification,
    route_after_generation,
    route_after_approval,
    route_post_processing,
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
    state = {"edits_urgent": True}
    assert route_after_generation(state) == "approval_node"

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
