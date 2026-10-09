"""Choice labels, answer identity, and persistent Telegram answer marks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from jobapply.nodes.execution import get_radio_option_label
from jobapply.execution.controls import clean_choice_option_label, get_role_radio_option_label
from jobapply.execution.telegram_qa import ask_user_for_question, extract_answer_from_reply
from jobapply.models.telegram import CorrelationStatus, CorrelationWaitResult
from jobapply.utils.telegram import TelegramClient


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Are you eligible? Yes", "Yes"),
        ("Are you eligible?*: No", "No"),
        ("Are you eligible?", ""),
        ("Yes", "Yes"),
        ("Yes, subject to sponsorship", "Yes, subject to sponsorship"),
    ],
)
def test_repeated_question_is_removed(label, expected):
    assert clean_choice_option_label("Are you eligible?*", label) == expected


@pytest.mark.asyncio
async def test_native_radio_prefers_visible_label_to_whole_question():
    radio = AsyncMock()
    radio.get_attribute.side_effect = lambda name: {
        "id": "yes",
        "aria-label": "Are you eligible? Yes",
    }.get(name)
    label = AsyncMock()
    label.inner_text.return_value = "Yes"
    fieldset = AsyncMock()
    fieldset.query_selector.return_value = label
    assert await get_radio_option_label(radio, fieldset) == "Yes"


@pytest.mark.asyncio
async def test_role_radio_prefers_visible_choice():
    option = AsyncMock()
    option.inner_text.return_value = "No"
    option.get_attribute.return_value = "Are you eligible? No"
    assert await get_role_radio_option_label(option, "Are you eligible?") == "No"


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["No", "2", "لا"])
async def test_exact_or_numbered_reply_does_not_call_model(reply):
    model = MagicMock(side_effect=AssertionError("model must not be called"))
    assert (
        await extract_answer_from_reply(
            "Eligible?", reply, ["Yes", "No"], ["نعم", "لا"], get_llm_fn=model
        )
        == "No"
    )


async def translate(question, options, language):
    return question, options


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["callback", "text"])
async def test_answer_marks_correct_choice_for_button_and_text(kind):
    telegram = AsyncMock()
    telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED,
        reply_text="No",
        reply_kind=kind,
        reply_option_index=1 if kind == "callback" else None,
    )
    settings = SimpleNamespace(form_qa_timeout_seconds=30, telegram_question_language="English")

    async def extract(*args):
        return "No"
    assert await ask_user_for_question(
        "Eligible?",
        {"job_id": "1"},
        telegram,
        settings,
        options=["Yes", "No"],
        translate_fn=translate,
        extract_fn=extract,
    ) == ("No", False)
    text = telegram.mark_question_answer.await_args.args[1]
    assert "○ Yes" in text and "✅ No" in text


@pytest.mark.asyncio
async def test_unmatched_answer_is_not_marked_as_selected():
    telegram = AsyncMock()
    telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED, reply_text="Maybe"
    )
    async def extract(*args):
        return "Maybe"
    await ask_user_for_question(
        "Eligible?",
        {"job_id": "1"},
        telegram,
        SimpleNamespace(form_qa_timeout_seconds=30, telegram_question_language="English"),
        options=["Yes", "No"],
        translate_fn=translate,
        extract_fn=extract,
    )
    telegram.mark_question_answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_mark_updates_original_prompt_and_removes_buttons(monkeypatch):
    telegram = TelegramClient()
    telegram.telegram_repo = AsyncMock()
    telegram.telegram_repo.get_correlation.return_value = SimpleNamespace(prompt_message_id=42)
    client = AsyncMock()
    client.post.return_value = httpx.Response(
        200, json={"ok": True}, request=httpx.Request("POST", "https://example.test")
    )
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr("jobapply.utils.telegram.httpx.AsyncClient", lambda **kwargs: context)
    assert await telegram.mark_question_answer("question-1", "✅ No")
    payload = client.post.await_args.kwargs["json"]
    assert payload["message_id"] == 42
    assert payload["reply_markup"] == {"inline_keyboard": []}
    assert payload["text"] == "✅ No"
