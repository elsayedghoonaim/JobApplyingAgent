from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jobapply.models.telegram import CorrelationStatus, CorrelationWaitResult
from jobapply.nodes import execution as execution_module
from jobapply.utils.candidate_facts import (
    CandidateFact,
    CandidateFactsRepository,
    classify_candidate_fact,
    compatible_saved_value,
)


def memory_settings():
    return SimpleNamespace(
        mongodb_db="jobapply",
        candidate_facts_collection="candidate_facts",
        candidate_fact_validity_days=365,
        candidate_fact_history_limit=20,
        telegram_question_language="English",
        form_qa_timeout_seconds=30,
    )


def test_fact_classification_is_scoped_and_sensitive_values_are_not_remembered():
    egypt = {"location": "Cairo, Egypt", "title": "ML Engineer"}
    germany = {"location": "Berlin, Germany", "title": "ML Engineer"}

    egypt_auth = classify_candidate_fact(
        "Are you legally authorized to work in this location?", egypt
    )
    germany_auth = classify_candidate_fact(
        "Are you legally authorized to work in this location?", germany
    )
    assert egypt_auth.fact_key == "employment.work_authorization"
    assert egypt_auth.scope != germany_auth.scope

    sensitive = classify_candidate_fact("What is your passport number?", egypt)
    assert sensitive.remember is False


def test_saved_choice_must_match_current_form_options():
    assert compatible_saved_value("yes", ["Yes", "No"]) == "Yes"
    assert compatible_saved_value("Maybe", ["Yes", "No"]) is None
    assert compatible_saved_value("five", None) == "five"


@pytest.mark.asyncio
async def test_repository_records_confirmation_and_one_year_expiry():
    collection = SimpleNamespace(
        create_index=AsyncMock(return_value="idx"),
        update_one=AsyncMock(),
    )
    repository = CandidateFactsRepository(collection=collection, settings=memory_settings())
    identity = classify_candidate_fact(
        "How many years of Python experience do you have?", {}
    )
    now = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)

    await repository.remember(identity, "5", "Python years?", now=now)

    update = collection.update_one.await_args.args[1]
    assert update["$set"]["confirmed_at"] == now
    assert update["$set"]["expires_at"] == now + timedelta(days=365)
    assert update["$push"]["history"]["$slice"] == -20
    assert collection.update_one.await_args.kwargs["upsert"] is True


class _FactRepository:
    def __init__(self, fact=None):
        self.fact = fact
        self.remember = AsyncMock()

    async def get(self, identity):
        return self.fact


def _fact(*, expired: bool) -> CandidateFact:
    now = datetime.now(timezone.utc)
    confirmed = now - timedelta(days=400 if expired else 10)
    return CandidateFact(
        fact_id="fact-id",
        fact_key="employment.willing_to_relocate",
        scope="location:cairo_egypt",
        value="Yes",
        original_question="Relocate?",
        confirmed_at=confirmed,
        expires_at=confirmed + timedelta(days=365),
    )


@pytest.mark.asyncio
async def test_execution_reuses_current_fact_without_telegram(monkeypatch):
    repository = _FactRepository(_fact(expired=False))
    monkeypatch.setattr(
        execution_module, "CandidateFactsRepository", lambda settings: repository
    )
    telegram_ask = AsyncMock()
    monkeypatch.setattr(execution_module, "_ext_ask_user_for_question", telegram_ask)

    answer, timed_out = await execution_module.ask_user_for_question(
        "Are you willing to relocate?",
        {"job_id": "1", "location": "Cairo, Egypt"},
        AsyncMock(),
        memory_settings(),
        options=["Yes", "No"],
    )

    assert (answer, timed_out) == ("Yes", False)
    telegram_ask.assert_not_awaited()


@pytest.mark.asyncio
async def test_execution_reconfirms_expired_fact_and_renews_date(monkeypatch):
    fact = _fact(expired=True)
    repository = _FactRepository(fact)
    monkeypatch.setattr(
        execution_module, "CandidateFactsRepository", lambda settings: repository
    )
    telegram_ask = AsyncMock(return_value=("Yes", False))
    monkeypatch.setattr(execution_module, "_ext_ask_user_for_question", telegram_ask)

    answer, timed_out = await execution_module.ask_user_for_question(
        "Are you willing to relocate?",
        {"job_id": "1", "location": "Cairo, Egypt"},
        AsyncMock(),
        memory_settings(),
        options=["Yes", "No"],
    )

    assert (answer, timed_out) == ("Yes", False)
    assert telegram_ask.await_args.kwargs["previous_answer"] == "Yes"
    assert telegram_ask.await_args.kwargs["previous_confirmed_at"] == fact.confirmed_at
    repository.remember.assert_awaited_once()


@pytest.mark.asyncio
async def test_expiry_prompt_still_correct_button_returns_previous_answer():
    from jobapply.execution.telegram_qa import (
        STILL_CORRECT_ACTION_LABEL,
    )
    from jobapply.execution.telegram_qa import (
        ask_user_for_question as ask_telegram_question,
    )

    telegram = AsyncMock()
    telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED,
        reply_text=STILL_CORRECT_ACTION_LABEL,
        reply_kind="callback",
        reply_option_index=0,
    )

    async def passthrough(question, options, language):
        return question, options or []

    answer, timed_out = await ask_telegram_question(
        "Are you willing to relocate?",
        {"job_id": "1", "title": "Engineer", "company": "Acme"},
        telegram,
        memory_settings(),
        options=["Yes", "No"],
        previous_answer="Yes",
        previous_confirmed_at=datetime(2025, 8, 1, tzinfo=timezone.utc),
        translate_fn=passthrough,
    )

    assert (answer, timed_out) == ("Yes", False)
    prompt = telegram.send_and_wait_for_reply.await_args.kwargs["prompt_text"]
    assert prompt == "<b>Are you willing to relocate?</b>"
    assert telegram.send_and_wait_for_reply.await_args.kwargs["inline_actions"][0] == (
        STILL_CORRECT_ACTION_LABEL
    )
