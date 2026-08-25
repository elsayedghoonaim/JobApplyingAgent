from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from jobapply.graph import route_after_generation, route_start
from jobapply.live import linkedin_url_is_signed_in, run_live
from jobapply.main import resolve_graph_input
from jobapply.models.job import QualificationResult
from jobapply.models.telegram import (
    CorrelationStatus,
    CorrelationWaitResult,
    OutboxDeliveryResult,
    OutboxStatus,
)
from jobapply.nodes.execution import (
    STANDARD_TEXT_FIELD_SELECTOR,
    UserSkippedJob,
    ask_user_for_question,
    choice_is_unanswered,
    classify_navigation_action,
    execution_node,
    extract_answer_from_reply,
    find_already_applied_indicator,
    format_application_receipt,
    format_job_question_summary,
    get_choice_label,
    get_form_field_label,
    get_radio_option_label,
    is_known_field,
    is_required_field,
    is_skip_job_reply,
    match_choice_index,
    select_live_radio_option,
    select_radio_option,
    text_indicates_already_applied,
    translate_question_for_telegram,
    wait_for_submission_confirmation,
)
from jobapply.nodes.notification import format_session_summary, notification_node
from jobapply.nodes.qualification import qualification_node
from jobapply.nodes.search import (
    build_search_url,
    find_applied_status_on_card,
    text_indicates_applied_card_status,
)
from jobapply.settings import get_settings
from jobapply.utils.browser import edge_debug_ports, edge_profile_path
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.job_filters import (
    find_disallowed_required_languages,
    get_job_exclusion_reason,
    is_senior_position_title,
)
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.limits import caps_reached
from jobapply.utils.llm import _extract_text, get_llm
from jobapply.utils.prompts import QUALIFICATION_PROMPT
from jobapply.utils.telegram import TelegramClient, correlated_reply_text


def _base_state(**updates):
    state = {
        "run_id": "test-run",
        "dry_run": True,
        "applications_count": 0,
        "daily_applications_count": 0,
        "max_applications": 2,
        "daily_application_cap": 10,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "errors": [],
    }
    state.update(updates)
    return state


def test_single_model_configuration():
    settings = get_settings()
    assert settings.llm_model == "gemma-4-31b-it"
    assert get_llm().model == "gemma-4-31b-it"


def test_env_has_every_documented_variable_and_no_old_provider_keys():
    def keys(path):
        return {
            line.split("=", 1)[0]
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#") and "=" in line
        }

    actual = keys(".env")
    expected = keys(".env.example")
    assert expected <= actual
    assert not any(
        marker in key
        for key in actual
        for marker in ("OPENMODEL", "LAYER1", "LAYER2", "LLM_ACTIVE", "LLM_GEMINI")
    )


def test_gemma_response_uses_non_thought_parts():
    data = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "private reasoning", "thought": True},
                        {"text": "final answer"},
                    ]
                }
            }
        ]
    }
    assert _extract_text(data) == "final answer"


def test_json_extraction_handles_reasoning_wrapper():
    assert extract_json_object('analysis first\n{"score": 0.8}\nfinished') == {"score": 0.8}


def test_score_is_constrained_to_zero_one():
    with pytest.raises(ValidationError):
        QualificationResult(
            qualified=True,
            score=2,
            reasoning="bad",
            key_matches=[],
            gaps=[],
            job_summary="bad",
        )


def test_prompts_mark_job_text_as_untrusted():
    assert "untrusted data" in QUALIFICATION_PROMPT


def test_search_url_encodes_special_characters():
    url = build_search_url("https://www.linkedin.com", "C++ & AI", 2)
    assert "keywords=C%2B%2B+%26+AI" in url
    assert "start=25" in url


@pytest.mark.parametrize(
    "title",
    [
        "Senior Software Engineer",
        "Sr. Data Scientist",
        "Lead Backend Engineer",
        "Staff ML Engineer",
        "Principal Developer",
        "Engineering Manager",
        "Director of Data",
        "Head of AI",
        "Solutions Architect",
    ],
)
def test_senior_position_titles_are_excluded(title):
    assert is_senior_position_title(title)
    assert get_job_exclusion_reason({"title": title}).startswith("Senior-level")


def test_non_senior_titles_are_not_excluded_by_title_words():
    assert not is_senior_position_title("Junior Software Engineer")
    assert not is_senior_position_title("Lead Generation Specialist")


@pytest.mark.parametrize(
    "title",
    [
        "Mid-Senior AI Engineer",
        "Mid -Senior AI Engineer",
        "Mid / Senior AI Engineer",
        "Mid to Senior AI Engineer",
        "Middle Senior AI Engineer",
        "Mid-Level / Senior AI Engineer",
    ],
)
def test_mixed_mid_senior_titles_are_included(title):
    assert not is_senior_position_title(title)
    assert get_job_exclusion_reason({"title": title}) is None


def test_only_mandatory_non_arabic_english_languages_are_excluded():
    assert (
        find_disallowed_required_languages(
            {
                "required_languages": ["English", "Arabic"],
                "description": "German is a nice-to-have skill.",
            }
        )
        == []
    )
    assert (
        find_disallowed_required_languages(
            {
                "required_languages": ["English (C1)", "Arabic - fluent"],
            }
        )
        == []
    )
    assert find_disallowed_required_languages(
        {
            "required_languages": ["English", "German"],
        }
    ) == ["German"]
    assert find_disallowed_required_languages(
        {
            "description": "You must speak Spanish to support our customers.",
        }
    ) == ["Spanish"]
    assert find_disallowed_required_languages(
        {
            "description": "You must be fluent in English and French.",
        }
    ) == ["French"]
    assert (
        find_disallowed_required_languages(
            {
                "description": "French is preferred but not required.",
            }
        )
        == []
    )


@pytest.mark.parametrize(
    "status",
    [
        "Applied",
        "Already applied",
        "You applied 2 days ago",
        "Application submitted",
        "Applied 3 weeks ago",
    ],
)
def test_search_card_recognizes_explicit_applied_status(status):
    assert text_indicates_applied_card_status(status)


@pytest.mark.parametrize(
    "text",
    ["Easy Apply", "Apply now", "Applied Scientist", "Application Engineer"],
)
def test_search_card_does_not_confuse_job_text_with_applied_status(text):
    assert not text_indicates_applied_card_status(text)


@pytest.mark.asyncio
async def test_search_card_returns_applied_evidence_before_extraction():
    card = AsyncMock()
    card.evaluate.return_value = ["Radley James", "Applied"]

    assert await find_applied_status_on_card(card) == "Applied"


def test_current_linkedin_experience_and_required_field_markers():
    assert not is_known_field("How many years of Generative AI experience do you have?")
    assert is_required_field(None, "true", "Experience")
    assert is_required_field("", None, "Experience")
    assert is_required_field(None, None, "Experience*")
    assert not is_required_field(None, None, "Optional note")


def test_choice_placeholders_and_option_matching():
    assert choice_is_unanswered(None)
    assert choice_is_unanswered("", "Select an option")
    assert choice_is_unanswered("choose an option", "Choose an option")
    assert not choice_is_unanswered("yes", "Yes")
    assert match_choice_index("yes please", ["Yes", "No"]) == 0
    assert match_choice_index("remote", ["On-site", "Remote", "Hybrid"]) == 1
    assert match_choice_index("maybe", ["Yes", "No"]) is None


def test_standard_field_discovery_covers_all_text_entry_types():
    assert "input:not([type])" in STANDARD_TEXT_FIELD_SELECTOR
    assert "input[type='text']" in STANDARD_TEXT_FIELD_SELECTOR
    assert "input[type='number']" in STANDARD_TEXT_FIELD_SELECTOR
    assert "input[type='url']" in STANDARD_TEXT_FIELD_SELECTOR
    assert "textarea" in STANDARD_TEXT_FIELD_SELECTOR
    assert "contenteditable='true'" in STANDARD_TEXT_FIELD_SELECTOR


@pytest.mark.asyncio
async def test_standard_field_prefers_accessible_question_label():
    control = AsyncMock()
    control.get_attribute.side_effect = [
        "How many years of Python experience do you have?",
    ]

    assert await get_form_field_label(control, AsyncMock()) == (
        "How many years of Python experience do you have?"
    )


@pytest.mark.asyncio
async def test_choice_question_uses_linkedin_preceding_label():
    group = AsyncMock()
    group.get_attribute.return_value = None
    group.evaluate.return_value = "Are you legally authorized to work in Denmark?*"

    assert await get_choice_label(group, AsyncMock()) == (
        "Are you legally authorized to work in Denmark?*"
    )


@pytest.mark.asyncio
async def test_radio_option_uses_role_radio_visible_text():
    radio = AsyncMock()
    radio.get_attribute.side_effect = [None, "radio-yes"]
    radio.evaluate.return_value = "Yes"
    empty_label = AsyncMock()
    empty_label.inner_text.return_value = ""
    fieldset = AsyncMock()
    fieldset.query_selector.return_value = empty_label

    assert await get_radio_option_label(radio, fieldset) == "Yes"


@pytest.mark.asyncio
async def test_hidden_native_radio_clicks_visible_role_wrapper():
    wrapper = AsyncMock()
    wrapper.is_visible.return_value = True
    handle = MagicMock()
    handle.as_element.return_value = wrapper
    handle.dispose = AsyncMock()
    radio = AsyncMock()
    radio.evaluate_handle.return_value = handle
    radio.is_checked.return_value = True

    method = await select_radio_option(radio, AsyncMock())

    assert method == "visible role=radio control"
    wrapper.click.assert_awaited_once_with(timeout=5_000)
    radio.check.assert_not_awaited()


@pytest.mark.asyncio
async def test_hidden_native_radio_uses_forced_check_fallback():
    wrapper = AsyncMock()
    wrapper.is_visible.return_value = False
    handle = MagicMock()
    handle.as_element.return_value = wrapper
    handle.dispose = AsyncMock()
    radio = AsyncMock()
    radio.evaluate_handle.return_value = handle
    radio.get_attribute.return_value = None
    radio.is_checked.return_value = True

    method = await select_radio_option(radio, AsyncMock())

    assert method == "forced native radio fallback"
    radio.check.assert_awaited_once_with(force=True, timeout=5_000)


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.select_radio_option")
async def test_live_radio_selection_re_resolves_after_detached_element(mock_select):
    first_legend = AsyncMock()
    first_legend.inner_text.return_value = "RAG experience?*"
    second_legend = AsyncMock()
    second_legend.inner_text.return_value = "RAG experience?*"

    first_radio = AsyncMock()
    first_radio.get_attribute.return_value = "Yes"
    first_radio.is_checked.return_value = False
    second_radio = AsyncMock()
    second_radio.get_attribute.return_value = "Yes"
    second_radio.is_checked.return_value = False

    first_fieldset = AsyncMock()
    first_fieldset.is_visible.return_value = True
    first_fieldset.query_selector.return_value = first_legend
    first_fieldset.query_selector_all.return_value = [first_radio]
    second_fieldset = AsyncMock()
    second_fieldset.is_visible.return_value = True
    second_fieldset.query_selector.return_value = second_legend
    second_fieldset.query_selector_all.return_value = [second_radio]

    page = AsyncMock()
    page.query_selector_all.side_effect = [[first_fieldset], [second_fieldset]]
    mock_select.side_effect = [
        RuntimeError("Element is not attached to the DOM"),
        "forced native radio fallback",
    ]

    method = await select_live_radio_option(page, "RAG experience?*", "Yes")

    assert method == "forced native radio fallback"
    assert mock_select.await_count == 2
    mock_select.assert_awaited_with(second_radio, second_fieldset)


@pytest.mark.parametrize(
    ("text", "aria_label", "expected"),
    [
        ("Continue", None, "advance"),
        ("", "Continue to next step", "advance"),
        ("Proceed", None, "advance"),
        ("Review", None, "advance"),
        ("Submit application", None, "submit"),
        ("", "Send application", "submit"),
        ("Back", None, None),
        ("Close", None, None),
        ("Save", None, None),
    ],
)
def test_navigation_action_classification(text, aria_label, expected):
    assert classify_navigation_action(text, aria_label) == expected


def test_telegram_accepts_natural_direct_reply_and_nonce_fallback():
    direct = {
        "chat": {"id": 123},
        "text": "I have about three years",
        "reply_to_message": {"message_id": 77},
    }
    assert correlated_reply_text(direct, "123", "abc123", 77) == direct["text"]
    assert correlated_reply_text(direct, "123", "abc123", 88) is None
    plain_next_message = {
        "chat": {"id": 123},
        "message_id": 79,
        "text": "2 years",
    }
    assert correlated_reply_text(plain_next_message, "123", "abc123", 77) == "2 years"
    assert correlated_reply_text(plain_next_message, "123", "abc123", 80) is None
    nonce_reply = {"chat": {"id": 123}, "text": "abc123 4 years"}
    assert correlated_reply_text(nonce_reply, "123", "abc123", None) == "4 years"


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.get_llm")
async def test_gemma_extracts_form_value_from_natural_reply(mock_get_llm):
    llm = AsyncMock()
    llm.ainvoke.return_value = SimpleNamespace(content='{"answer": "3"}')
    mock_get_llm.return_value = llm
    answer = await extract_answer_from_reply(
        "How many years of Generative AI experience do you have?",
        "I have worked with it for around three years.",
    )
    assert answer == "3"


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.translate_question_for_telegram")
async def test_telegram_question_message_contains_question_options_and_skip(mock_translate):
    mock_translate.return_value = ("Are you willing to relocate?", ["Yes", "No"])
    telegram = AsyncMock()
    telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.TIMED_OUT, timed_out=True
    )
    settings = SimpleNamespace(
        form_qa_timeout_seconds=30,
        telegram_question_language="English",
    )
    await ask_user_for_question(
        "Are you willing to relocate?",
        {"title": "Engineer", "company": "Acme"},
        telegram,
        settings,
        options=["Yes", "No"],
    )
    telegram.send_and_wait_for_reply.assert_awaited_once()
    assert (
        telegram.send_and_wait_for_reply.call_args.kwargs["prompt_text"]
        == "Are you willing to relocate?\n\n- Yes\n- No\n\n- /skip — Skip this job"
    )


def test_job_question_summary_contains_context_before_questions():
    message = format_job_question_summary(
        {
            "title": "ML Engineer",
            "company": "Acme",
            "location": "London",
            "work_type": "Remote",
            "url": "https://example.test/jobs/42",
        },
        {
            "score": 0.84,
            "job_summary": "Build reliable machine-learning services.",
            "key_matches": ["Python", "MLOps"],
        },
    )

    assert message.startswith("JOB SUMMARY\n")
    assert "Job: ML Engineer" in message
    assert "Company: Acme" in message
    assert "Location: London" in message
    assert "Fit score: 84%" in message
    assert "Role summary: Build reliable machine-learning services." in message
    assert "Key matches: Python, MLOps" in message
    assert "Reply /skip to any question" in message


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.translate_question_for_telegram")
async def test_explicit_skip_reply_stops_question_processing(mock_translate):
    mock_translate.return_value = ("Are you willing to relocate?", ["Yes", "No"])
    telegram = AsyncMock()
    telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED, reply_text="/skip"
    )
    settings = SimpleNamespace(
        form_qa_timeout_seconds=30,
        telegram_question_language="English",
    )

    with pytest.raises(UserSkippedJob) as skipped:
        await ask_user_for_question(
            "Are you willing to relocate?",
            {"title": "Engineer", "company": "Acme"},
            telegram,
            settings,
            options=["Yes", "No"],
        )

    assert skipped.value.question == "Are you willing to relocate?"
    assert is_skip_job_reply("skip this job")
    assert not is_skip_job_reply("Do not skip this answer")


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.get_llm")
async def test_gemma_translates_non_english_question_and_options(mock_get_llm):
    llm = AsyncMock()
    llm.ainvoke.return_value = SimpleNamespace(
        content=(
            '{"question":"Are you legally authorized to work in Germany?*","options":["Yes","No"]}'
        )
    )
    mock_get_llm.return_value = llm

    question, options = await translate_question_for_telegram(
        "Sind Sie berechtigt, in Deutschland zu arbeiten?*",
        ["Ja", "Nein"],
        "English",
    )

    assert question == "Are you legally authorized to work in Germany?*"
    assert options == ["Yes", "No"]


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.get_llm")
async def test_translated_reply_maps_back_to_original_option(mock_get_llm):
    llm = AsyncMock()
    llm.ainvoke.side_effect = RuntimeError("translation mapping unavailable")

    answer = await extract_answer_from_reply(
        "Sind Sie berechtigt, in Deutschland zu arbeiten?*",
        "No",
        ["Ja", "Nein"],
        displayed_options=["Yes", "No"],
    )

    assert answer == "Nein"


def test_application_receipts_distinguish_submitted_dry_run_and_already_applied():
    outcome = {
        "title": "ML Engineer",
        "company": "Acme",
        "score": 0.85,
        "qa_count": 3,
        "url": "https://example.test/job",
        "timestamp": "2026-07-21T20:00:00+00:00",
    }
    submitted = format_application_receipt({**outcome, "status": "submitted"})
    dry_run = format_application_receipt({**outcome, "status": "dry_run"})
    already_applied = format_application_receipt(
        {**outcome, "status": "skipped", "reason": "already_applied"}
    )
    assert "APPLICATION SUBMITTED" in submitted
    assert "LinkedIn confirmed" in submitted
    assert "NOT SUBMITTED" in dry_run
    assert "Submit was not clicked" in dry_run
    assert "ALREADY APPLIED" in already_applied
    assert "submitted previously" in already_applied


def test_already_applied_text_requires_explicit_status():
    assert text_indicates_already_applied("Applied")
    assert text_indicates_already_applied("You applied 2 days ago")
    assert text_indicates_already_applied("Application submitted")
    assert not text_indicates_already_applied("Easy Apply")
    assert not text_indicates_already_applied("Apply now")


@pytest.mark.asyncio
async def test_finds_visible_already_applied_indicator():
    indicator = AsyncMock()
    indicator.is_visible.return_value = True
    indicator.inner_text.return_value = "Applied"
    indicator.get_attribute.return_value = None
    page = AsyncMock()
    page.query_selector_all.side_effect = [[indicator], [], [], []]

    assert await find_already_applied_indicator(page) == "Applied"


def test_session_summary_is_plain_and_structured():
    state = _base_state(
        run_id="summary-1",
        search_queries=["ML Engineer"],
        current_query_index=0,
        application_outcomes=[
            {
                "status": "submitted",
                "title": "Submitted Job",
                "company": "Acme",
                "score": 0.8,
                "qa_count": 2,
            },
            {
                "status": "dry_run",
                "title": "Dry Job",
                "company": "Beta",
                "score": 0.7,
                "qa_count": 1,
            },
        ],
    )
    message = format_session_summary(state)
    assert "Confirmed submitted: 1" in message
    assert "Dry run only (not submitted): 1" in message
    assert "CONFIRMED SUBMISSIONS" in message
    assert "DRY RUNS — SUBMIT WAS NOT CLICKED" in message
    assert "**" not in message


def test_edge_dynamic_debug_port_discovery_ignored(tmp_path):
    marker = tmp_path / "DevToolsActivePort"
    marker.write_text("43123\n/devtools/browser/test", encoding="utf-8")
    assert edge_debug_ports(9222, marker) == [9222]


def test_edge_automation_profile_is_resolved_outside_normal_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    profile = edge_profile_path(r"%LOCALAPPDATA%\JobApply\EdgeProfile")
    assert "JobApply" in str(profile)


def test_linkedin_signin_url_detection():
    assert linkedin_url_is_signed_in("https://www.linkedin.com/feed/")
    assert linkedin_url_is_signed_in("https://www.linkedin.com/jobs/")
    assert not linkedin_url_is_signed_in("https://www.linkedin.com/login")
    assert not linkedin_url_is_signed_in("https://www.linkedin.com/checkpoint/challenge")
    assert not linkedin_url_is_signed_in("https://example.com/feed/")


@pytest.mark.asyncio
@patch("jobapply.live.DeduplicationStore.close", new_callable=AsyncMock)
@patch("jobapply.live.run", new_callable=AsyncMock)
@patch("jobapply.live.run_preflight", new_callable=AsyncMock)
async def test_live_preflight_only_never_runs_workflow(
    mock_preflight,
    mock_run,
    mock_close,
):
    await run_live(preflight_only=True)
    mock_preflight.assert_awaited_once()
    mock_run.assert_not_awaited()
    mock_close.assert_awaited_once()


@pytest.mark.asyncio
@patch("jobapply.live.DeduplicationStore.close", new_callable=AsyncMock)
@patch("jobapply.live.TelegramClient")
@patch("jobapply.live.run", new_callable=AsyncMock)
@patch("jobapply.live.run_preflight", new_callable=AsyncMock)
async def test_live_runner_forces_real_submission_mode(
    mock_preflight,
    mock_run,
    mock_telegram_class,
    mock_close,
):
    telegram = AsyncMock()
    mock_telegram_class.return_value = telegram
    await run_live(max_jobs=4, max_applications=2)
    mock_preflight.assert_awaited_once()
    mock_run.assert_awaited_once_with(
        dry_run=False,
        max_jobs=4,
        max_applications=2,
    )
    telegram.send_message.assert_awaited_once()
    mock_close.assert_awaited_once()


def test_caps_stop_before_search_and_allow_missing_caps_in_partial_states():
    capped = _base_state(max_applications=0)
    assert caps_reached(capped)
    assert route_start(capped) == "notification_node"
    assert not caps_reached({"dry_run": True})


def test_dry_run_count_enforces_session_cap_but_not_daily_cap():
    state = _base_state(
        max_applications=1,
        daily_applications_count=999,
        daily_application_cap=1,
        application_outcomes=[{"status": "dry_run"}],
    )
    assert caps_reached(state)


@pytest.mark.asyncio
async def test_live_seen_filter_retries_dry_run_and_pending_qualified_jobs():
    class Cursor:
        def __init__(self):
            self._items = iter([{"job_id": "submitted-1"}])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    store = object.__new__(DeduplicationStore)
    store.collection = MagicMock()
    store.collection.find.return_value = Cursor()
    result = await store.load_seen_ids(for_live_run=True)
    assert result == {"submitted-1"}
    store.collection.find.assert_called_once_with(
        {"status": {"$nin": ["qualified", "dry_run"]}},
        {"job_id": 1},
    )


def test_generation_failure_never_routes_to_execution():
    assert route_after_generation({"application_status": "failed"}) == "select_next_job_node"


def test_resume_input_selection():
    initial = {"run_id": "new"}
    pending = SimpleNamespace(values={"run_id": "saved"}, next=("execution_node",))
    complete = SimpleNamespace(values={"run_id": "saved"}, next=())
    assert resolve_graph_input(True, pending, initial) == (None, None)
    assert resolve_graph_input(True, complete, initial) == (None, {"run_id": "saved"})
    assert resolve_graph_input(False, pending, initial) == (initial, None)


@pytest.mark.asyncio
async def test_execution_cap_guard_avoids_browser():
    state = _base_state(
        max_applications=0,
        current_job={"job_id": "1", "title": "Engineer", "company": "Acme"},
        qualification_result={"score": 0.9},
    )
    with patch("jobapply.nodes.execution.managed_browser") as browser:
        result = await execution_node(state)
    assert result["application_status"] == "skipped"
    assert result["application_outcomes"][0]["reason"] == "application_cap_reached"
    browser.assert_not_called()


@pytest.mark.asyncio
async def test_submission_requires_explicit_confirmation():
    page = MagicMock()
    page.wait_for_function = AsyncMock(return_value=True)
    assert await wait_for_submission_confirmation(page)

    page.wait_for_function = AsyncMock(side_effect=TimeoutError)
    body = MagicMock()
    body.inner_text = AsyncMock(return_value="Validation error: answer required")
    page.locator.return_value = body
    assert not await wait_for_submission_confirmation(page)


@pytest.mark.asyncio
@patch("jobapply.nodes.qualification.DeduplicationStore")
@patch("jobapply.nodes.qualification.get_llm")
async def test_qualification_stage_in_isolation(mock_get_llm, mock_store_class):
    llm = AsyncMock()
    llm.ainvoke.return_value = SimpleNamespace(
        content=(
            '{"qualified": true, "score": 0.82, "reasoning": "Strong match", '
            '"key_matches": ["Python"], "gaps": [], "job_summary": "Build systems"}'
        )
    )
    mock_get_llm.return_value = llm
    store = MagicMock()
    store.mark_seen = AsyncMock()
    mock_store_class.return_value = store
    state = _base_state(
        current_job={
            "job_id": "42",
            "title": "ML Engineer",
            "company": "Acme",
            "location": "Remote",
            "url": "https://example.test/job/42",
            "description": "Build Python ML systems",
        },
        seen_job_ids=set(),
        jobs_evaluated_count=0,
        qualified_jobs_count=0,
        not_qualified_jobs_count=0,
    )
    result = await qualification_node(state)
    assert result["qualification_result"]["qualified"] is True
    assert result["jobs_evaluated_count"] == 1
    store.mark_seen.assert_awaited_once()


@pytest.mark.asyncio
@patch("jobapply.nodes.qualification.get_llm")
async def test_qualification_exclusion_bypasses_llm(mock_get_llm):
    state = _base_state(
        current_job={
            "job_id": "excluded-42",
            "title": "Senior Staff Architect",
            "company": "Acme",
            "location": "Remote",
            "url": "https://example.test/job/excluded-42",
            "description": "Senior architecture role.",
        },
        seen_job_ids=set(),
        jobs_evaluated_count=0,
        qualified_jobs_count=0,
        not_qualified_jobs_count=0,
    )

    result = await qualification_node(state)

    assert result["qualification_result"]["qualified"] is False
    assert "Senior-level position excluded" in result["qualification_result"]["reasoning"]
    assert result["not_qualified_jobs_count"] == 1
    assert "excluded-42" in result["seen_job_ids"]
    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
@patch("jobapply.nodes.qualification.DeduplicationStore")
@patch("jobapply.nodes.qualification.get_llm")
async def test_qualification_failure_still_consumes_job_limit(mock_get_llm, mock_store_class):
    llm = AsyncMock()
    llm.ainvoke.return_value = SimpleNamespace(content="not JSON")
    mock_get_llm.return_value = llm
    state = _base_state(
        current_job={
            "job_id": "bad-42",
            "title": "ML Engineer",
            "company": "Acme",
            "location": "Remote",
            "url": "https://example.test/job/bad-42",
            "description": "Build Python ML systems",
        },
        seen_job_ids=set(),
        jobs_evaluated_count=0,
    )
    result = await qualification_node(state)
    assert result["qualification_result"]["qualified"] is False
    assert result["jobs_evaluated_count"] == 1
    assert len(result["errors"]) == 1
    mock_store_class.assert_not_called()


@pytest.mark.asyncio
@patch("jobapply.nodes.notification.TelegramClient")
async def test_notification_stage_in_isolation(mock_telegram_class):
    telegram = AsyncMock()
    telegram.enqueue_and_deliver.return_value = OutboxDeliveryResult(
        success=True, status=OutboxStatus.SENT, idempotency_key="summary:summary-test"
    )
    telegram.outbox_repo = AsyncMock()
    telegram.outbox_repo.get_pending_and_unknown_counts.return_value = (0, 0)
    mock_telegram_class.return_value = telegram
    state = _base_state(
        run_id="summary-test",
        current_query_index=0,
        search_queries=["ML Engineer"],
        application_outcomes=[
            {
                "status": "dry_run",
                "title": "ML Engineer",
                "company": "Acme",
                "score": 0.8,
                "qa_count": 0,
            }
        ],
    )
    result = await notification_node(state)
    # The notification stage is also the central manual-review flush boundary,
    # so its state update carries the (emptied) durable pending list.
    assert result == {
        "notification_sent": True,
        "outbox_pending_count": 0,
        "outbox_unknown_count": 0,
        "manual_review_queue_pending": [],
    }
    telegram.enqueue_and_deliver.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_http_failure_is_not_silent():
    response = MagicMock()
    fake_token = "123456789:AAFakeTelegramTokenSecretXYZ"
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"400 Client Error for https://api.telegram.org/bot{fake_token}/sendMessage",
        request=httpx.Request("POST", f"https://api.telegram.org/bot{fake_token}/sendMessage"),
        response=httpx.Response(400),
    )
    client = AsyncMock()
    client.post.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = client
    context.__aexit__.return_value = False
    with patch("jobapply.utils.telegram.httpx.AsyncClient", return_value=context):
        with pytest.raises(RuntimeError) as exc_info:
            await TelegramClient()._send_single_message("hello", None)
        assert fake_token not in str(exc_info.value)
        assert "400 Client Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_telegram_poll_failure_redacts_credentials():
    response = MagicMock()
    fake_token = "123456789:AAFakeTelegramTokenSecretXYZ"
    response.raise_for_status.side_effect = httpx.ConnectError(
        f"Connection failed for https://api.telegram.org/bot{fake_token}/getUpdates: connection refused",
        request=httpx.Request("POST", f"https://api.telegram.org/bot{fake_token}/getUpdates"),
    )
    client = AsyncMock()
    client.post.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = client
    context.__aexit__.return_value = False

    mock_update_res = MagicMock()
    mock_update_res.matched_count = 1
    mock_update_res.modified_count = 1

    mock_collection = AsyncMock()
    mock_collection.create_index = AsyncMock(return_value="index_created")
    mock_collection.find_one = AsyncMock(return_value=None)
    mock_collection.update_one = AsyncMock(return_value=mock_update_res)
    mock_collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="1"))
    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    mock_motor = MagicMock()
    mock_motor.__getitem__.return_value = mock_db

    with (
        patch("jobapply.utils.telegram.httpx.AsyncClient", return_value=context),
        patch("jobapply.utils.mongo.AsyncIOMotorClient", return_value=mock_motor),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await TelegramClient().wait_for_correlated_reply("nonce123", timeout=10)
        assert fake_token not in str(exc_info.value)
        assert "bot[REDACTED]" in str(exc_info.value) or "[REDACTED]" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_dedup_store_preserves_exact_raw_job_id_semantics():
    from jobapply.utils.dedup import DeduplicationStore

    mock_collection = AsyncMock()
    mock_collection.find_one.return_value = None

    store = DeduplicationStore()
    store.collection = mock_collection

    raw_ids = ["job/123", "job:456", "job#789", "../job_traversal"]
    for rid in raw_ids:
        await store.is_seen(rid)
        mock_collection.find_one.assert_awaited_with({"job_id": rid})

        await store.mark_seen(rid, {"title": "Engineer", "api_key": "secret"})
        call_args = mock_collection.update_one.call_args[0]
        assert call_args[0] == {"job_id": rid}
        assert call_args[1]["$set"]["job_id"] == rid
        assert call_args[1]["$set"]["api_key"] == "[REDACTED]"
