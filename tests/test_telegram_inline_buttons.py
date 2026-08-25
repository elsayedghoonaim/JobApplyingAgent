"""Deterministic offline tests for Telegram inline buttons and callback durability.

All transport is mocked; no real network is touched. Covers: happy-path button
approval with record→cursor→ack→consume ordering, duplicate-callback
idempotency, wrong chat/nonce/prompt rejection, strict codec validation,
action-list immutability, index-based action identity through consumers,
single-message transport for buttoned prompts (>4000-char regression), crash
recovery at every durability boundary, ordinary text fallback, and fail-closed
delivery.
"""

import asyncio
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from test_telegram_recovery_and_outbox import InMemoryTelegramCollection

from jobapply.models.telegram import CorrelationStatus, CorrelationWaitResult
from jobapply.utils.mongo import MongoClientManager
from jobapply.utils.telegram import MAX_CALLBACK_ANSWER_LENGTH, TelegramClient
from jobapply.utils.telegram_storage import (
    CALLBACK_DATA_MAX_BYTES,
    CALLBACK_DATA_PREFIX,
    TelegramApiDefiniteRejectError,
    TelegramRepository,
    build_inline_keyboard,
    decode_callback_data,
    encode_callback_data,
    sanitize_inline_actions,
)

ACTIONS = ["✅ Approve", "📄 Use Base", "❌ Skip"]


@pytest.fixture(autouse=True)
async def _reset_mongo():
    await MongoClientManager.close()
    try:
        yield
    finally:
        await MongoClientManager.close()


def _shared_collections():
    """One shared in-memory 'deployment' so multiple clients see durable state."""
    return {
        "correlations": InMemoryTelegramCollection(key_field="correlation_key"),
        "cursors": InMemoryTelegramCollection(key_field="bot_chat_key"),
        "outbox": InMemoryTelegramCollection(key_field="idempotency_key"),
    }


def _attach(client, collections):
    client.telegram_repo.correlations_col = collections["correlations"]
    client.telegram_repo.cursors_col = collections["cursors"]
    client.outbox_repo.outbox_col = collections["outbox"]
    return client


def _client_with_memory_collections():
    return _attach(TelegramClient(), _shared_collections())


def make_post_transport(
    *,
    build_updates: Optional[Callable[[str, int], list[dict]]] = None,
    ack_texts: Optional[list[str]] = None,
):
    """Build a mocked httpx.AsyncClient transport for one waiting session.

    ``build_updates(nonce, prompt_message_id)`` constructs updates lazily on the
    first poll so tests can embed the exact durable nonce/prompt IDs.
    """
    holder: dict[str, Any] = {"acks": ack_texts if ack_texts is not None else []}
    state = {"polls": 0}

    def make_client_cls():
        http_cls = MagicMock()
        ctx = AsyncMock()
        http_client = AsyncMock()

        async def post(url, json=None, headers=None):
            resp = MagicMock()
            resp.raise_for_status.return_value = None
            if "answerCallbackQuery" in url:
                holder["acks"].append(str(json.get("text", "")))
                resp.json.return_value = {"ok": True}
                return resp
            state["polls"] += 1
            result: list[dict] = []
            if build_updates is not None and state["polls"] == 1:
                nonce = holder.get("nonce")
                prompt_id = holder.get("prompt_message_id")
                if nonce is not None and isinstance(prompt_id, int):
                    result = build_updates(nonce, prompt_id)
            resp.json.return_value = {"ok": True, "result": result}
            return resp

        http_client.post.side_effect = post
        ctx.__aenter__.return_value = http_client
        ctx.__aexit__.return_value = False
        http_cls.return_value = ctx
        return http_cls

    original_register = TelegramRepository.register_intent
    original_record_prompt = TelegramRepository.record_prompt_sent

    async def capturing_register(self, **kwargs):
        reg = await original_register(self, **kwargs)
        holder["nonce"] = reg.nonce
        holder["registered"] = reg
        return reg

    async def capturing_record_prompt(self, *args, **kwargs):
        ok = await original_record_prompt(self, *args, **kwargs)
        if ok:
            holder["prompt_message_id"] = args[1]
        return ok

    return holder, make_client_cls, capturing_register, capturing_record_prompt


def bind_capture_wrappers(client, cap_reg, cap_prompt):
    client.telegram_repo.register_intent = cap_reg.__get__(client.telegram_repo, TelegramRepository)
    client.telegram_repo.record_prompt_sent = cap_prompt.__get__(
        client.telegram_repo, TelegramRepository
    )


def callback_update(
    update_id: int,
    message_id: int,
    data: str,
    configured_chat_id: str,
    cb_id: str = "cbq1",
) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": cb_id,
            "message": {"message_id": message_id, "chat": {"id": configured_chat_id}},
            "data": data,
        },
    }


# ── Codec, keyboard, strict validation ───────────────────────────────────────


def test_callback_data_is_opaque_bounded_and_nonce_bound():
    data = encode_callback_data("abc123def456", 2)
    assert len(data.encode()) <= CALLBACK_DATA_MAX_BYTES <= 64
    assert decode_callback_data(data) == ("abc123def456", 2)
    assert "Approve" not in data
    assert decode_callback_data("garbage") is None
    assert decode_callback_data("jb|abc|notanumber") is None
    assert decode_callback_data(None) is None


def test_inline_keyboard_layout_and_bounds():
    rows = build_inline_keyboard("nonce12", ACTIONS + ["Skip Job"])
    flat = [btn for row in rows for btn in row]
    assert len(flat) == 4
    for index, btn in enumerate(flat):
        assert btn["callback_data"] == f"jb|nonce12|{index}"
        assert len(btn["text"]) <= 64
    assert all(len(row) <= 3 for row in rows)


def test_sanitize_inline_actions_caps_and_bounds():
    actions = sanitize_inline_actions([" a ", "", "x" * 500] + [f"opt{i}" for i in range(20)])
    assert actions[0] == "a"
    assert len(actions) <= 8
    assert all(len(a) <= 64 for a in actions)


def test_strict_prefix_and_index_validation_in_client_helper():
    client = TelegramClient()
    chat = client.settings.telegram_chat_id
    cbq_template = {
        "id": "cbqV",
        "message": {"message_id": 500, "chat": {"id": chat}},
        "data": "",
    }

    def attempt(data):
        cbq_template["data"] = data
        return client.callback_data_is_for_current_prompt(cbq_template, chat, "nonce01", 500)

    good = attempt("jb|nonce01|3")
    assert good == (True, "cbqV", 3)

    # Wrong prefix / foreign formats are rejected outright.
    assert attempt("xy|nonce01|0")[0] is False
    assert attempt("cb|nonce01|0")[0] is False
    assert attempt("nonce01|0")[0] is False

    # Negative, unbounded, and non-numeric indexes are rejected.
    assert attempt("jb|nonce01|-1")[0] is False
    assert attempt("jb|nonce01|99999")[0] is False
    assert attempt("jb|nonce01|abc")[0] is False

    # Wrong nonce / wrong prompt / wrong chat never satisfy the waiter.
    assert attempt("jb|otherNonce|0")[0] is False
    assert (
        client.callback_data_is_for_current_prompt(
            {**cbq_template, "data": "jb|nonce01|0"},
            chat,
            "nonce01",
            501,
        )[0]
        is False
    )
    assert (
        client.callback_data_is_for_current_prompt(
            {
                "id": "cbqV",
                "message": {"message_id": 500, "chat": {"id": "evil"}},
                "data": "jb|nonce01|0",
            },
            chat,
            "nonce01",
            500,
        )[0]
        is False
    )


# ── Happy path: durable acceptance ordering ──────────────────────────────────


@pytest.mark.asyncio
async def test_button_press_is_recorded_cursor_saved_then_acknowledged_then_consumed():
    order: list[str] = []
    collections = _shared_collections()
    client = _attach(TelegramClient(), collections)
    chat_id = client.settings.telegram_chat_id

    original_record = client.telegram_repo.record_callback_reply
    original_save_cursor = client.telegram_repo.save_cursor
    original_consume = client.telegram_repo.consume_reply

    async def spy_record(*args, **kwargs):
        res = await original_record(*args, **kwargs)
        order.append("record")
        return res

    async def spy_save(*args, **kwargs):
        saved = await original_save_cursor(*args, **kwargs)
        order.append("cursor")
        return saved

    async def spy_ack(cb_query_id: str, text: str = "") -> bool:
        order.append("ack")
        assert cb_query_id == "cbq1"
        assert len(text) <= MAX_CALLBACK_ANSWER_LENGTH + 10
        return True

    async def spy_consume(*args, **kwargs):
        res = await original_consume(*args, **kwargs)
        order.append("consume")
        return res

    client.telegram_repo.record_callback_reply = spy_record
    client.telegram_repo.save_cursor = spy_save
    client._answer_callback_query = spy_ack
    client.telegram_repo.consume_reply = spy_consume

    holder, make_client_cls, cap_reg, cap_prompt = make_post_transport(
        build_updates=lambda nonce, pid: [
            callback_update(99, pid, encode_callback_data(nonce, 0), chat_id)
        ]
    )
    bind_capture_wrappers(client, cap_reg, cap_prompt)

    with patch.object(client, "_send_single_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 500
        with patch("jobapply.utils.telegram.httpx.AsyncClient", make_client_cls()):
            result = await asyncio.wait_for(
                client.send_and_wait_for_reply(
                    correlation_key="approval:r1:j1",
                    purpose="approval",
                    run_id="r1",
                    prompt_text="Urgent edit?",
                    timeout=3,
                    inline_actions=ACTIONS,
                ),
                timeout=15,
            )

    assert result.status == CorrelationStatus.REPLIED, result
    assert result.reply_kind == "callback"
    assert result.reply_option_index == 0
    assert result.reply_text == "✅ Approve"
    assert order == ["record", "cursor", "ack", "consume"]
    cursor = await client.telegram_repo.get_cursor(client.bot_chat_key)
    assert cursor == 99
    corr = await client.telegram_repo.get_correlation("approval:r1:j1")
    assert corr.status == CorrelationStatus.CONSUMED
    assert corr.reply_kind == "callback"
    assert corr.reply_option_index == 0
    assert corr.reply_callback_query_id == "cbq1"


@pytest.mark.asyncio
async def test_buttoned_prompt_is_one_transport_message_with_markup(tmp_path):
    client = _client_with_memory_collections()
    captured_markups: list[dict | None] = []
    captured_texts: list[str] = []

    async def fake_send_single(text, parse_mode=None, reply_markup=None):
        captured_texts.append(text)
        captured_markups.append(reply_markup)
        return 700

    client._send_single_message = fake_send_single
    _, make_client_cls, cap_reg, _ = make_post_transport()
    holder2, make_client_cls2, cap_reg2, cap_prompt2 = make_post_transport()
    bind_capture_wrappers(client, cap_reg2, cap_prompt2)
    make_client_cls = make_client_cls2

    with patch("jobapply.utils.telegram.httpx.AsyncClient", make_client_cls()):
        result = await asyncio.wait_for(
            client.send_and_wait_for_reply(
                correlation_key="approval:markup",
                purpose="approval",
                run_id="r1",
                prompt_text="Choose",
                timeout=1,
                inline_actions=ACTIONS,
            ),
            timeout=15,
        )

    assert result.status == CorrelationStatus.TIMED_OUT
    assert len(captured_markups) == 1 and captured_markups[0] is not None
    buttons = [b for row in captured_markups[0]["inline_keyboard"] for b in row]
    assert [b["text"] for b in buttons] == ACTIONS
    assert len(captured_texts) == 1


@pytest.mark.asyncio
async def test_long_buttoned_prompt_stays_one_message_and_callbacks_match(tmp_path):
    """Regression: a >4000-character buttoned prompt must remain one message."""
    client = _client_with_memory_collections()
    chat_id = client.settings.telegram_chat_id
    long_body = "Detailed context line. " * 400  # ~9200 chars

    calls: list[dict] = []

    async def fake_send_single(text, parse_mode=None, reply_markup=None):
        calls.append({"text": text, "reply_markup": reply_markup})
        return 555

    client._send_single_message = fake_send_single

    def build_updates(nonce, pid):
        assert pid == 555
        return [callback_update(4242, pid, encode_callback_data(nonce, 1), chat_id)]

    holder, make_client_cls, cap_reg, cap_prompt = make_post_transport(build_updates=build_updates)
    bind_capture_wrappers(client, cap_reg, cap_prompt)

    async def spy_ack(cb_query_id, text=""):
        holder["acks"].append(text)
        return True

    client._answer_callback_query = spy_ack

    with patch("jobapply.utils.telegram.httpx.AsyncClient", make_client_cls()):
        result = await asyncio.wait_for(
            client.send_and_wait_for_reply(
                correlation_key="form_qa:long",
                purpose="form_qa",
                run_id="r1",
                prompt_text=long_body,
                timeout=3,
                inline_actions=["Option A", "Option B"],
            ),
            timeout=15,
        )

    # Exactly ONE transport message carrying the keyboard.
    assert len(calls) == 1
    assert calls[0]["reply_markup"] is not None
    assert len(calls[0]["text"]) <= 4000
    assert holder["prompt_message_id"] == 555
    # Callback validated against THAT id and resolved by persisted index.
    assert result.status == CorrelationStatus.REPLIED
    assert result.reply_option_index == 1
    assert result.reply_text == "Option B"


# ── Validation rejections never satisfy the waiter ───────────────────────────


async def _run_rejection_case(correlation_key: str, build_updates):
    client = _client_with_memory_collections()
    holder, make_client_cls, cap_reg, cap_prompt = make_post_transport(build_updates=build_updates)
    bind_capture_wrappers(client, cap_reg, cap_prompt)
    with patch.object(client, "_send_single_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 800
        with patch("jobapply.utils.telegram.httpx.AsyncClient", make_client_cls()):
            result = await asyncio.wait_for(
                client.send_and_wait_for_reply(
                    correlation_key=correlation_key,
                    purpose="approval",
                    run_id="r1",
                    prompt_text="Choose",
                    timeout=2,
                    inline_actions=ACTIONS,
                ),
                timeout=15,
            )
    corr = await client.telegram_repo.get_correlation(correlation_key)
    cursor = await client.telegram_repo.get_cursor(client.bot_chat_key)
    return result, corr, cursor


@pytest.mark.asyncio
async def test_wrong_chat_callback_never_satisfies_waiter_but_advances_cursor():

    def build(nonce, pid):
        return [callback_update(55, pid, encode_callback_data(nonce, 0), "evil-chat", "cbqX")]

    result, corr, cursor = await _run_rejection_case("approval:r1:j9", build)
    assert result.status == CorrelationStatus.TIMED_OUT
    assert cursor == 55
    assert corr.status == CorrelationStatus.TIMED_OUT
    assert corr.reply_kind is None


@pytest.mark.asyncio
async def test_wrong_nonce_wrong_prompt_and_out_of_range_are_ignored():
    chat_id = TelegramClient().settings.telegram_chat_id

    async def case(builder, key):
        return await _run_rejection_case(key, builder)

    bad_nonce = await case(
        lambda n, p: [callback_update(56, p, encode_callback_data("deadbeefcafe", 0), chat_id)],
        "approval:r1:n1",
    )
    wrong_prompt = await case(
        lambda n, p: [callback_update(57, p + 777777, encode_callback_data(n, 0), chat_id)],
        "approval:r1:p1",
    )
    out_of_range = await case(
        lambda n, p: [
            {
                "update_id": 58,
                "callback_query": {
                    "id": "cbqR",
                    "message": {"message_id": p, "chat": {"id": chat_id}},
                    "data": f"{CALLBACK_DATA_PREFIX}|{n}|99",
                },
            }
        ],
        "approval:r1:o1",
    )

    for result, corr, _cursor in (bad_nonce, wrong_prompt, out_of_range):
        assert result.status == CorrelationStatus.TIMED_OUT
        assert corr.status == CorrelationStatus.TIMED_OUT
        assert corr.reply_kind is None


# ── Correlation identity includes the sanitized action list ──────────────────


@pytest.mark.asyncio
async def test_changed_action_list_fails_closed_as_mismatched_payload():
    client = _client_with_memory_collections()
    key = "approval:actions-immutable"

    first = await client.telegram_repo.register_intent(
        correlation_key=key,
        purpose="approval",
        run_id="r1",
        prompt_text="Same body",
        inline_actions=["✅ Approve", "📄 Use Base", "❌ Skip"],
    )
    assert first.registered is True
    assert first.inline_actions == ["✅ Approve", "📄 Use Base", "❌ Skip"]

    conflict = await client.telegram_repo.register_intent(
        correlation_key=key,
        purpose="approval",
        run_id="r1",
        prompt_text="Same body",
        inline_actions=["Yes", "No"],  # changed payload
    )
    assert conflict.registered is False
    assert conflict.reason == "correlation_key_conflict_mismatched_payload"
    # The UI is never shown actions different from those stored durably.
    assert conflict.inline_actions == ["✅ Approve", "📄 Use Base", "❌ Skip"]

    identical = await client.telegram_repo.register_intent(
        correlation_key=key,
        purpose="approval",
        run_id="r1",
        prompt_text="Same body",
        inline_actions=["✅ Approve", "📄 Use Base", "❌ Skip"],
    )
    assert identical.registered is True
    assert identical.reason is None


# ── Crash recovery across every durability boundary ──────────────────────────


async def _drive_to_boundary(collections, key: str, boundary: str) -> dict:
    """Durably advance one callback to just past `boundary`, then 'crash'.

    Boundaries: 'record' (before cursor), 'cursor' (before acknowledgment),
    'ack' (before consume). Returns context (nonce etc.).
    """
    client = _attach(TelegramClient(), collections)
    chat_id = client.settings.telegram_chat_id
    repo = client.telegram_repo

    reg = await repo.register_intent(
        correlation_key=key,
        purpose="approval",
        run_id="r1",
        prompt_text="Recover?",
        inline_actions=ACTIONS,
    )
    waiter = await repo.claim_waiter(key, lease_duration_seconds=60)
    assert waiter.acquired and waiter.lease_id
    assert await repo.mark_prompt_in_flight(key, waiter.lease_id)
    assert await repo.record_prompt_sent(key, 900, waiter.lease_id)

    cbq_id = "cbqCrash"
    rec = await repo.record_callback_reply(
        key,
        encode_callback_data(reg.nonce, 1),
        901,
        900,
        waiter.lease_id,
        callback_query_id=cbq_id,
    )
    assert rec.accepted is True

    if boundary in ("cursor", "ack"):
        poll = await repo.claim_poll_lease(client.bot_chat_key, lease_duration_seconds=60)
        assert poll
        assert await repo.save_cursor(client.bot_chat_key, 901, poll)
        await repo.release_poll_lease(client.bot_chat_key, poll)

    acks: list[str] = []
    if boundary == "ack":
        acks.append("sent-before-crash")

    # Crash: drop leases without consuming.
    await repo.release_waiter(key, waiter.lease_id)
    return {"nonce": reg.nonce, "chat_id": chat_id, "acks": acks, "cbq_id": cbq_id}


@pytest.mark.parametrize("boundary", ["record", "cursor", "ack"])
@pytest.mark.asyncio
async def test_crash_at_each_boundary_recovers_exactly_once(boundary):
    key = f"approval:recover:{boundary}"
    collections = _shared_collections()
    context = await _drive_to_boundary(collections, key, boundary)

    client = _attach(TelegramClient(), collections)
    acks: list[str] = []

    async def spy_ack(cb_query_id, text=""):
        acks.append((cb_query_id, text))
        return True

    client._answer_callback_query = spy_ack

    # No polling transport should even be needed: cached-reply recovery runs
    # under exact waiter/poll leases without contacting getUpdates.
    http_cls = MagicMock()
    ctx = AsyncMock()
    http_client = AsyncMock()

    async def post_fail(url, json=None, headers=None):
        raise AssertionError("recovery must not require network polling")

    http_client.post.side_effect = post_fail
    ctx.__aenter__.return_value = http_client
    ctx.__aexit__.return_value = False
    http_cls.return_value = ctx

    with patch("jobapply.utils.telegram.httpx.AsyncClient", http_cls):
        result = await asyncio.wait_for(
            client.send_and_wait_for_reply(
                correlation_key=key,
                purpose="approval",
                run_id="r1",
                prompt_text="Recover?",
                timeout=2,
                inline_actions=ACTIONS,
            ),
            timeout=15,
        )

    # The action is returned exactly once, identified by its durable index.
    assert result.status == CorrelationStatus.REPLIED
    assert result.reply_kind == "callback"
    assert result.reply_option_index == 1
    assert result.reply_text == "📄 Use Base"

    # Durable cursor reflects the persisted reply update id.
    cursor = await client.telegram_repo.get_cursor(client.bot_chat_key)
    assert cursor == 901

    # Persisted callback-query ID acknowledged during recovery.
    assert acks and acks[-1][0] == context["cbq_id"]
    assert "Recorded" in acks[-1][1]

    corr = await client.telegram_repo.get_correlation(key)
    assert corr.status == CorrelationStatus.CONSUMED


@pytest.mark.asyncio
async def test_recovered_callback_never_binds_to_another_prompt():
    key = "approval:recover:foreign"
    collections = _shared_collections()
    await _drive_to_boundary(collections, key, "record")

    client = _attach(TelegramClient(), collections)
    other_nonce = "ffffffffffff"

    async def spy_ack(cb_query_id, text=""):
        return True

    client._answer_callback_query = spy_ack

    # A callback belonging to another nonce arrives afterwards; it must never
    # satisfy anything or corrupt the recovered correlation.
    chat_id = client.settings.telegram_chat_id
    foreign_update = callback_update(999, 900, encode_callback_data(other_nonce, 0), chat_id)

    http_cls = MagicMock()
    ctx = AsyncMock()
    http_client = AsyncMock()
    state = {"polled": False}

    async def post(url, json=None, headers=None):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "answerCallbackQuery" in url:
            resp.json.return_value = {"ok": True}
            return resp
        if not state["polled"]:
            state["polled"] = True
            resp.json.return_value = {"ok": True, "result": []}  # foreign update withheld
        else:
            resp.json.return_value = {"ok": True, "result": [foreign_update]}
        return resp

    http_client.post.side_effect = post
    ctx.__aenter__.return_value = http_client
    ctx.__aexit__.return_value = False
    http_cls.return_value = ctx

    # Cached recovery consumes first; a subsequent wait sees CONSUMED terminal
    # state regardless of any foreign callbacks.
    with patch("jobapply.utils.telegram.httpx.AsyncClient", http_cls):
        first = await asyncio.wait_for(
            client.send_and_wait_for_reply(
                correlation_key=key,
                purpose="approval",
                run_id="r1",
                prompt_text="Recover?",
                timeout=2,
                inline_actions=ACTIONS,
            ),
            timeout=15,
        )
        second = await client.send_and_wait_for_reply(
            correlation_key=key,
            purpose="approval",
            run_id="r1",
            prompt_text="Recover?",
            timeout=1,
            inline_actions=ACTIONS,
        )

    assert first.status == CorrelationStatus.REPLIED
    assert first.reply_option_index == 1
    assert second.status == CorrelationStatus.CONSUMED


# ── Ordinary text fallback and fail-closed delivery ──────────────────────────


@pytest.mark.asyncio
async def test_plain_text_reply_still_accepted_alongside_buttons():
    client = _client_with_memory_collections()
    chat_id = client.settings.telegram_chat_id

    def build_updates(nonce, pid):
        return [
            {
                "update_id": 651,
                "message": {
                    "message_id": pid + 1,
                    "chat": {"id": chat_id},
                    "text": "✅ approve",
                },
            }
        ]

    holder, make_client_cls, cap_reg, cap_prompt = make_post_transport(build_updates=build_updates)
    bind_capture_wrappers(client, cap_reg, cap_prompt)
    with patch.object(client, "_send_single_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 650
        with patch("jobapply.utils.telegram.httpx.AsyncClient", make_client_cls()):
            result = await asyncio.wait_for(
                client.send_and_wait_for_reply(
                    correlation_key="approval:text-fallback",
                    purpose="approval",
                    run_id="r1",
                    prompt_text="Approve?",
                    timeout=3,
                    inline_actions=ACTIONS,
                ),
                timeout=15,
            )

    assert result.status == CorrelationStatus.REPLIED
    assert result.reply_kind == "text"
    assert result.reply_option_index is None
    assert "approve" in (result.reply_text or "").lower()


@pytest.mark.asyncio
async def test_buttoned_prompt_failure_is_fail_closed_without_resend():
    client = _client_with_memory_collections()
    send_calls = []

    async def failing_send(text, parse_mode=None, reply_markup=None):
        send_calls.append(reply_markup)
        raise TelegramApiDefiniteRejectError("rejected")

    client._send_single_message = failing_send
    holder, make_client_cls, cap_reg, _ = make_post_transport()
    holder2, make_client_cls2, cap_reg2, cap_prompt2 = make_post_transport()
    bind_capture_wrappers(client, cap_reg2, cap_prompt2)
    make_client_cls = make_client_cls2

    with patch("jobapply.utils.telegram.httpx.AsyncClient", make_client_cls()):
        result = await asyncio.wait_for(
            client.send_and_wait_for_reply(
                correlation_key="approval:failclosed",
                purpose="approval",
                run_id="r1",
                prompt_text="Approve?",
                timeout=2,
                inline_actions=ACTIONS,
            ),
            timeout=15,
        )

    assert result.status == CorrelationStatus.PROMPT_FAILED
    assert len(send_calls) == 1  # exactly one attempt; never silently resent


# ── Consumer-side index identity ─────────────────────────────────────────────


class _StubSettings:
    telegram_question_language = "English"
    form_qa_timeout_seconds = 5


def _stub_telegram(wait_result: CorrelationWaitResult):
    tg = MagicMock()
    tg.send_and_wait_for_reply = AsyncMock(return_value=wait_result)
    return tg


@pytest.mark.asyncio
async def test_form_qa_maps_callback_by_persisted_index_not_label():
    from jobapply.execution.telegram_qa import ask_user_for_question

    question = "Preferred language?"
    options = ["Python", "Go", "Rust"]

    # Truncated-label trap: two labels truncating identically must not matter.
    wait = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED,
        reply_text="Go",  # rendered/truncated label only
        reply_kind="callback",
        reply_option_index=2,
    )
    answer, timed_out = await ask_user_for_question(
        question,
        {"job_id": "j1"},
        _stub_telegram(wait),
        _StubSettings(),
        options=options,
        run_id="r1",
        translate_fn=_passthrough_translate,
    )
    assert (answer, timed_out) == ("Rust", False)


@pytest.mark.asyncio
async def test_form_qa_skip_index_raises_user_skipped_directly():
    from jobapply.execution.telegram_qa import (
        SKIP_JOB_ACTION_LABEL,
        UserSkippedJob,
        ask_user_for_question,
    )

    options = ["Python", "Go"]
    wait = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED,
        reply_text=SKIP_JOB_ACTION_LABEL,
        reply_kind="callback",
        reply_option_index=2,  # display_options(2) + Skip Job => index 2
    )
    with pytest.raises(UserSkippedJob):
        await ask_user_for_question(
            "Q?",
            {"job_id": "j1"},
            _stub_telegram(wait),
            _StubSettings(),
            options=options,
            run_id="r1",
            translate_fn=_passthrough_translate,
        )


@pytest.mark.asyncio
async def test_approval_maps_callback_indexes_to_decisions():
    from jobapply.nodes.approval import approval_node

    base_state = {
        "run_id": "r1",
        "dry_run": True,
        "current_job": {"job_id": "j9", "title": "T", "company": "C"},
        "qualification_result": {"score": 0.7},
        "edit_reasoning": "why",
        # No proposed edits keeps every decision branch free of LLM/PDF work.
        "proposed_edits": None,
        "resume_path": None,
        "cover_letter_path": None,
        "application_outcomes": [],
        "skipped_jobs": [],
        "errors": [],
        "logs": [],
    }

    async def decision(index, expected_status):
        wait = CorrelationWaitResult(
            status=CorrelationStatus.REPLIED,
            reply_text="whatever-rendered",
            reply_kind="callback",
            reply_option_index=index,
        )
        with patch("jobapply.nodes.approval.TelegramClient", lambda: _stub_telegram(wait)):
            updates = await approval_node(base_state)
        assert updates["approval_status"] == expected_status
        return updates

    await decision(0, "approved")
    await decision(1, "use_base")

    skip_updates = await decision(2, "skip")
    assert skip_updates["application_status"] == "skipped"
    assert skip_updates["application_outcomes"][-1]["reason"] == "user_skipped"


async def _passthrough_translate(question, options, language):
    return question, list(options or [])
