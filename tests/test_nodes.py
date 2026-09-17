from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobapply.models.telegram import CorrelationStatus, CorrelationWaitResult
from jobapply.nodes.approval import approval_node, format_approval_question
from jobapply.nodes.execution import execution_node
from jobapply.nodes.generation import generation_node


def test_approval_question_is_structured():
    prompt = format_approval_question(
        {
            "current_job": {
                "title": "ML Engineer",
                "company": "Acme",
                "url": "https://example.test/job/1",
            },
            "qualification_result": {"score": 0.85},
            "edit_reasoning": "Highlight production ML experience.",
            "proposed_edits": "Add MLOps deployment bullet.",
        }
    )

    assert prompt.startswith("<b>📝 RESUME DECISION REQUIRED</b>\n━━━━━━━━━━━━━━━━━━━━")
    assert "<b>JOB DETAILS</b>\n• Role: ML Engineer\n• Company: Acme\n• Fit score: 85%" in prompt
    assert "<b>JOB LINK</b>\nhttps://example.test/job/1" in prompt
    assert "<b>WHY AN EDIT WAS SUGGESTED</b>\nHighlight production ML experience." in prompt
    assert "<b>PROPOSED CHANGES</b>\nAdd MLOps deployment bullet." in prompt
    assert "<b>ACTION REQUIRED</b>\n• Approve" in prompt


@pytest.mark.asyncio
@patch("jobapply.nodes.generation.get_cover_letter_path", return_value="cover-letter.txt")
@patch("jobapply.nodes.generation.open", new_callable=MagicMock)
@patch("jobapply.nodes.generation.get_cached_profile", return_value=({}, "Candidate profile"))
@patch("jobapply.nodes.generation.get_llm")
async def test_generation_prepares_base_resume_without_approval_or_edit_call(
    mock_get_llm,
    mock_profile,
    mock_open,
    mock_cover_path,
):
    llm = AsyncMock()
    llm.ainvoke.return_value = MagicMock(content="Cover letter")
    mock_get_llm.return_value = llm

    result = await generation_node(
        {
            "run_id": "run-auto",
            "current_job": {
                "job_id": "123",
                "title": "ML Engineer",
                "company": "Acme",
                "description": "Build ML systems",
            },
            "qualification_result": {"qualified": True, "score": 0.95},
            "errors": [],
        }
    )

    mock_get_llm.assert_called_once()
    llm.ainvoke.assert_awaited_once()
    assert result["resume_path"].endswith("resume.pdf")
    assert result["edits_urgent"] is False
    assert result["approval_status"] is None
    assert result["application_status"] is None


@pytest.mark.asyncio
@patch("jobapply.nodes.approval.TelegramClient")
async def test_approval_node_skip(mock_telegram_class):
    mock_telegram = AsyncMock()
    mock_telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED, reply_text="skip", nonce="n1"
    )
    mock_telegram_class.return_value = mock_telegram

    state = {
        "current_job": {"job_id": "123", "title": "SWE", "company": "Google", "url": "http://g.co"},
        "qualification_result": {"score": 0.85},
        "skipped_jobs": [],
        "application_outcomes": [],
    }

    result = await approval_node(state)
    assert result["approval_status"] == "skip"
    assert len(result["skipped_jobs"]) == 1
    assert result["skipped_jobs"][0]["status"] == "skipped"
    assert result["skipped_jobs"][0]["reason"] == "user_skipped"


@pytest.mark.asyncio
@patch("jobapply.nodes.approval.TelegramClient")
async def test_approval_node_use_base_on_timeout(mock_telegram_class):
    mock_telegram = AsyncMock()
    mock_telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.TIMED_OUT, timed_out=True, nonce="n1"
    )
    mock_telegram_class.return_value = mock_telegram

    state = {
        "current_job": {"job_id": "123", "title": "SWE", "company": "Google"},
        "qualification_result": {"score": 0.85},
    }

    result = await approval_node(state)
    assert result["approval_status"] == "use_base"
    assert result["resume_edited"] is False


@pytest.mark.asyncio
@patch("jobapply.nodes.approval.TelegramClient")
@patch("jobapply.nodes.approval.get_llm")
@patch("jobapply.nodes.approval.markdown_to_pdf")
@patch("jobapply.nodes.approval.open", new_callable=MagicMock)
@patch("jobapply.nodes.approval.os.path.exists")
async def test_approval_node_approve(
    mock_exists, mock_open, mock_markdown_to_pdf, mock_get_llm, mock_telegram_class
):
    mock_telegram = AsyncMock()
    mock_telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED, reply_text="approve", nonce="n1"
    )
    mock_telegram_class.return_value = mock_telegram

    # Mock original resume markdown
    mock_file = MagicMock()
    mock_file.read.return_value = "Original Resume Markdown"
    mock_open.return_value.__enter__.return_value = mock_file

    # Mock LLM and markdown_to_pdf
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content="Edited Resume Markdown")
    mock_get_llm.return_value = mock_llm

    mock_exists.return_value = True

    state = {
        "run_id": "run_test_123",
        "current_job": {"job_id": "123", "title": "SWE", "company": "Google"},
        "qualification_result": {"score": 0.85},
        "proposed_edits": "Fix this keyword",
    }

    result = await approval_node(state)
    assert result["approval_status"] == "approved"
    assert result["resume_edited"] is True
    assert "edited_resume_123.pdf" in result["resume_path"]


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.managed_browser")
@patch("jobapply.nodes.execution.TelegramClient")
@patch("jobapply.nodes.execution.take_error_screenshot")
async def test_execution_node_unexpected_exception(
    mock_screenshot, mock_telegram_class, mock_managed_browser
):
    # Mock browser throwing exception
    mock_managed_browser.side_effect = RuntimeError("CDP Connection failed")

    state = {
        "run_id": "run_test_123",
        "current_job": {"job_id": "123", "title": "SWE", "company": "Google", "url": "http://g.co"},
        "qualification_result": {"score": 0.85},
        "resume_path": "resume.pdf",
        "errors": [],
        "application_outcomes": [],
    }

    # Mock os.path.exists to pass state validation
    with (
        patch("jobapply.nodes.execution.os.path.exists", return_value=True),
        patch("yaml.safe_load", return_value={}),
    ):
        result = await execution_node(state)

        assert result["application_status"] == "failed"
        assert "CDP Connection failed" in result["application_error"]
        assert len(result["application_outcomes"]) == 1
        assert result["application_outcomes"][0]["status"] == "failed"
        assert len(result["errors"]) == 1
