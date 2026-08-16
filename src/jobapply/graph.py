"""LangGraph state graph with routing logic."""

from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from pymongo import MongoClient

from jobapply.nodes import (
    approval_node,
    execution_node,
    generation_node,
    notification_node,
    qualification_node,
    search_node,
    select_next_job_node,
)
from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.limits import caps_reached

# ── Routing functions ──


def route_select_next_job(state: JobApplyState) -> str:
    """Three-tier exhaustion: next unseen job → next page → next query → done."""

    if state.get("account_safety_paused"):
        return "notification_node"

    if caps_reached(state):
        return "notification_node"

    if state.get("search_failed"):
        return "notification_node"

    # Check if max jobs limit reached
    max_jobs = state.get("max_jobs_to_evaluate")
    if max_jobs is not None:
        jobs_evaluated = state.get("jobs_evaluated_count", 0)
        if jobs_evaluated >= max_jobs:
            return "notification_node"

    # 1. If select_next_job_node selected a job, qualify that exact job.
    # Do not scan future listings here: current_job_index already points past
    # the selected job, so scanning from it skips the last unseen job on a page.
    if state.get("current_job"):
        return "qualification_node"

    # 2. No job selected; current listings are exhausted - need more pages?
    # Check if we should fetch next page (either max_jobs override or pages_per_query)
    should_fetch_next_page = False

    if max_jobs is not None:
        jobs_evaluated = state.get("jobs_evaluated_count", 0)
        if jobs_evaluated < max_jobs and state["current_page"] < 20:  # Safety limit
            should_fetch_next_page = True
    elif state["current_page"] < state["pages_per_query"]:
        should_fetch_next_page = True

    if should_fetch_next_page:
        return "increment_page_node"

    # 3. More queries?
    if state["current_query_index"] < len(state["search_queries"]) - 1:
        return "increment_query_node"

    # 4. Everything exhausted
    return "notification_node"


def route_start(state: JobApplyState) -> str:
    """Avoid browser and LLM work when an application cap is already reached or paused."""
    if state.get("account_safety_paused") or caps_reached(state):
        return "notification_node"
    return "search_node"


def route_after_qualification(state: JobApplyState) -> str:
    """Route after qualification check."""
    if state.get("account_safety_paused"):
        return "notification_node"
    result = state.get("qualification_result") or {}
    if result.get("qualified"):
        return "generation_node"
    return "select_next_job_node"  # skip to next job (not search!)


def route_after_generation(state: JobApplyState) -> str:
    """Route after generation - fast path or approval path."""
    if state.get("account_safety_paused"):
        return "notification_node"
    if state.get("application_status") == "failed":
        return "select_next_job_node"
    if state.get("edits_urgent", False):
        return "approval_node"
    return "execution_node"


def route_after_approval(state: JobApplyState) -> str:
    """Route after user approval response."""
    if state.get("account_safety_paused"):
        return "notification_node"
    if state.get("approval_status") == "skip":
        return "select_next_job_node"  # continues session
    return "execution_node"


def route_post_processing(state: JobApplyState) -> str:
    """Unified exit router — used after execution."""
    if state.get("account_safety_paused") or caps_reached(state):
        return "notification_node"
    return "select_next_job_node"


# ── Helper nodes for state increments ──


async def increment_page_node(state: JobApplyState) -> dict:
    """Increment page counter before fetching next page."""
    max_jobs = state.get("max_jobs_to_evaluate")
    jobs_evaluated = state.get("jobs_evaluated_count", 0)

    if max_jobs is not None:
        print(
            f"📄 Moving to page {state['current_page'] + 1} (need {max_jobs - jobs_evaluated} more jobs)"
        )
    else:
        print(f"📄 Moving to page {state['current_page'] + 1}")

    return {"current_page": state["current_page"] + 1, "search_failed": False}


async def increment_query_node(state: JobApplyState) -> dict:
    """Increment query index and reset page counter."""
    next_query = state["search_queries"][state["current_query_index"] + 1]
    print(f"🔄 Moving to next query: {next_query}")
    return {
        "current_query_index": state["current_query_index"] + 1,
        "current_page": 1,
        "search_failed": False,
    }


# ── Graph construction ──


def build_graph() -> StateGraph:
    """Build the LangGraph state graph.

    Returns:
        StateGraph builder instance (not yet compiled).
    """
    builder = StateGraph(JobApplyState)

    # Register all nodes with metadata
    builder.add_node(
        "search_node",
        search_node,
        metadata={
            "description": "LinkedIn job search via Playwright",
            "operations": ["CDP connect", "page navigation", "job card extraction"],
        },
    )
    builder.add_node(
        "select_next_job_node",
        select_next_job_node,
        metadata={
            "description": "Iterator/router for unseen jobs",
            "operations": ["job iteration", "pagination", "query advancement"],
        },
    )
    builder.add_node(
        "qualification_node",
        qualification_node,
        metadata={
            "description": "LLM-powered job fit evaluation",
            "operations": ["LLM scoring", "MongoDB dedup"],
        },
    )
    builder.add_node(
        "generation_node",
        generation_node,
        metadata={
            "description": "Cover letter + urgency check generation",
            "operations": ["cover letter LLM", "urgency check LLM", "file output"],
        },
    )
    builder.add_node(
        "approval_node",
        approval_node,
        metadata={
            "description": "Telegram human-in-the-loop for urgent edits",
            "operations": ["Telegram send", "Telegram wait"],
        },
    )
    builder.add_node(
        "execution_node",
        execution_node,
        metadata={
            "description": "Easy Apply automation with inline Q&A",
            "operations": ["browser navigation", "form filling", "Telegram Q&A", "submission"],
        },
    )
    builder.add_node(
        "notification_node",
        notification_node,
        metadata={"description": "Session summary via Telegram", "operations": ["Telegram send"]},
    )
    builder.add_node(
        "increment_page_node",
        increment_page_node,
        metadata={"description": "Increment page counter", "operations": ["state update"]},
    )
    builder.add_node(
        "increment_query_node",
        increment_query_node,
        metadata={
            "description": "Increment query index and reset page",
            "operations": ["state update"],
        },
    )

    # ── Edges ──

    # START → fetch first batch of listings
    builder.add_conditional_edges(
        START,
        route_start,
        {
            "search_node": "search_node",
            "notification_node": "notification_node",
        },
    )

    # After fetching listings → pick a job
    builder.add_edge("search_node", "select_next_job_node")

    # select_next_job → qualify, increment page, increment query, or notify (exhausted)
    builder.add_conditional_edges(
        "select_next_job_node",
        route_select_next_job,
        {
            "qualification_node": "qualification_node",
            "increment_page_node": "increment_page_node",
            "increment_query_node": "increment_query_node",
            "notification_node": "notification_node",
        },
    )

    # After incrementing page/query → fetch new results
    builder.add_edge("increment_page_node", "search_node")
    builder.add_edge("increment_query_node", "search_node")

    # qualify → generate (qualified) or select_next_job (not qualified)
    builder.add_conditional_edges(
        "qualification_node",
        route_after_qualification,
        {
            "generation_node": "generation_node",
            "select_next_job_node": "select_next_job_node",
            "notification_node": "notification_node",
        },
    )

    # generate → execute (fast path) or approval (rare edit path)
    builder.add_conditional_edges(
        "generation_node",
        route_after_generation,
        {
            "execution_node": "execution_node",
            "approval_node": "approval_node",
            "notification_node": "notification_node",
        },
    )

    # approval → execute or select_next_job (skip)
    builder.add_conditional_edges(
        "approval_node",
        route_after_approval,
        {
            "execution_node": "execution_node",
            "select_next_job_node": "select_next_job_node",
            "notification_node": "notification_node",
        },
    )

    # execution → post-processing (handles both submitted + timed-out-skip)
    builder.add_conditional_edges(
        "execution_node",
        route_post_processing,
        {
            "select_next_job_node": "select_next_job_node",
            "notification_node": "notification_node",
        },
    )

    # notification → END
    builder.add_edge("notification_node", END)

    return builder


def compile_graph():
    """Build and compile the graph with MongoDB checkpointer.

    Returns:
        Compiled graph ready for execution.
    """
    settings = get_settings()
    # Use synchronous MongoDB client for checkpointing
    client = MongoClient(settings.mongodb_url)
    checkpointer = MongoDBSaver(client, db_name=settings.mongodb_db)
    builder = build_graph()
    return builder.compile(checkpointer=checkpointer)


def make_graph_config(
    run_id: str,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> RunnableConfig:
    """LangGraph config with thread_id for checkpoint resume.

    The run_id is used as thread_id so that:
    - A crashed run can resume from the last checkpoint by reusing the same run_id
    - A new run gets a fresh thread by generating a new run_id (uuid4)

    Args:
        run_id: Unique run identifier to use as thread_id.
        tags: Optional list of tags for tracing.
        metadata: Optional metadata dict for tracing.

    Returns:
        Config dict for LangGraph.
    """
    config: RunnableConfig = {"configurable": {"thread_id": run_id}}

    if tags:
        config["tags"] = tags

    if metadata:
        config["metadata"] = metadata

    return config
