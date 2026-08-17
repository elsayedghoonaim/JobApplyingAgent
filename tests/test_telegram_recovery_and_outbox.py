"""Deterministic offline unit tests for Telegram recovery, correlated replies, and durable outbox."""

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobapply.models.telegram import (
    CorrelationStatus,
    CorrelationWaitResult,
    OutboxStatus,
)
from jobapply.nodes.approval import approval_node
from jobapply.nodes.execution import (
    FormQaInfrastructureError,
    ask_user_for_question,
)
from jobapply.nodes.notification import notification_node
from jobapply.utils.telegram import TelegramClient
from jobapply.utils.telegram_storage import (
    NotificationOutboxRepository,
    TelegramApiDefiniteRejectError,
    TelegramPersistenceManager,
    TelegramRepository,
    TelegramStorageError,
    TelegramTransportAmbiguousError,
    contains_sensitive_security_challenge,
    make_bot_chat_key,
)


class InMemoryTelegramCollection:
    """Async-safe in-memory MongoDB collection mock with accurate positional updates and leases."""

    def __init__(self, key_field: str = "_id"):
        self.docs: dict[str, dict[str, Any]] = {}
        self.key_field = key_field
        self.lock = asyncio.Lock()
        self.created_indexes: list[Any] = []

    async def create_index(self, key: Any, unique: bool = False, **kwargs):
        self.created_indexes.append(key)
        return "index_created"

    def _matches_elem(self, item: dict[str, Any], elem_query: dict[str, Any]) -> bool:
        for k, v in elem_query.items():
            if item.get(k) != v:
                return False
        return True

    def _matches(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for k, v in query.items():
            if k == "$or":
                if not any(self._matches(doc, branch) for branch in v):
                    return False
                continue
            if k == "$and":
                if not all(self._matches(doc, branch) for branch in v):
                    return False
                continue

            if k == "chunks" and isinstance(v, dict):
                chunks_list = doc.get("chunks", [])
                if "$elemMatch" in v:
                    if not any(self._matches_elem(c, v["$elemMatch"]) for c in chunks_list):
                        return False
                    continue
                if "$not" in v and "$elemMatch" in v["$not"]:
                    if any(self._matches_elem(c, v["$not"]["$elemMatch"]) for c in chunks_list):
                        return False
                    continue

            if "." in k:
                parts = k.split(".")
                curr = doc
                for p in parts:
                    if isinstance(curr, dict):
                        curr = curr.get(p)
                    elif isinstance(curr, list):
                        curr = [item.get(p) if isinstance(item, dict) else None for item in curr]
                    else:
                        curr = None
                if isinstance(curr, list):
                    if v not in curr:
                        return False
                    continue
                else:
                    if curr != v:
                        return False
                    continue

            val = doc.get(k)
            if isinstance(v, dict):
                if "$in" in v and val not in v["$in"]:
                    return False
                if "$ne" in v and val == v["$ne"]:
                    return False
                if "$lt" in v and (val is None or not (val < v["$lt"])):
                    return False
                if "$lte" in v and (val is None or not (val <= v["$lte"])):
                    return False
                if "$gt" in v and (val is None or not (val > v["$gt"])):
                    return False
                if "$gte" in v and (val is None or not (val >= v["$gte"])):
                    return False
            else:
                if val != v:
                    return False
        return True

    async def find_one(self, query: dict[str, Any]) -> Optional[dict[str, Any]]:
        async with self.lock:
            for doc in self.docs.values():
                if self._matches(doc, query):
                    return deepcopy(doc)
            return None

    def find(self, query: dict[str, Any]):
        class QueryCursor:
            def __init__(self, items: list[dict[str, Any]]):
                self.items = items
                self._limit = len(items)

            def limit(self, n: int):
                self._limit = n
                return self

            async def to_list(self, length: int):
                return deepcopy(self.items[: min(length, self._limit)])

            def __aiter__(self):
                self._iter = iter(deepcopy(self.items[: self._limit]))
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    raise StopAsyncIteration

        matched = [doc for doc in self.docs.values() if self._matches(doc, query)]
        return QueryCursor(matched)

    async def insert_one(self, doc: dict[str, Any]):
        async with self.lock:
            key = str(doc.get(self.key_field) or doc.get("_id") or len(self.docs))
            if key in self.docs:
                raise RuntimeError(f"Duplicate key error: {key}")
            self.docs[key] = deepcopy(doc)
            return MagicMock(inserted_id=key)

    async def update_one(self, query: dict[str, Any], update: dict[str, Any], upsert: bool = False):
        async with self.lock:
            for key, doc in self.docs.items():
                if self._matches(doc, query):
                    self._apply_update(doc, update, query)
                    return MagicMock(matched_count=1, modified_count=1)
            if upsert:
                new_doc = deepcopy(query)
                if "$setOnInsert" in update:
                    new_doc.update(update["$setOnInsert"])
                self._apply_update(new_doc, update, query)
                key = str(new_doc.get(self.key_field) or len(self.docs))
                self.docs[key] = new_doc
                return MagicMock(matched_count=0, modified_count=0, upserted_id=key)
            return MagicMock(matched_count=0, modified_count=0)

    async def update_many(self, query: dict[str, Any], update: dict[str, Any]):
        async with self.lock:
            modified = 0
            for doc in self.docs.values():
                if self._matches(doc, query):
                    self._apply_update(doc, update, query)
                    modified += 1
            return MagicMock(matched_count=modified, modified_count=modified)

    async def find_one_and_update(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
        return_document: bool = False,
    ) -> Optional[dict[str, Any]]:
        async with self.lock:
            for key, doc in self.docs.items():
                if self._matches(doc, query):
                    old_doc = deepcopy(doc)
                    self._apply_update(doc, update, query)
                    return deepcopy(doc) if return_document else old_doc
            if upsert:
                new_doc = deepcopy(query)
                if "$setOnInsert" in update:
                    new_doc.update(update["$setOnInsert"])
                self._apply_update(new_doc, update, query)
                key = str(new_doc.get(self.key_field) or len(self.docs))
                self.docs[key] = new_doc
                return deepcopy(new_doc)
            return None

    async def count_documents(self, query: dict[str, Any]) -> int:
        async with self.lock:
            return sum(1 for doc in self.docs.values() if self._matches(doc, query))

    def _apply_update(self, doc: dict[str, Any], update: dict[str, Any], query: dict[str, Any]):
        if "$set" in update:
            for k, v in update["$set"].items():
                if "." in k:
                    parts = k.split(".")
                    if parts[0] == "chunks" and parts[1] == "$" and len(parts) == 3:
                        field_name = parts[2]
                        target_chunk_index = None
                        if (
                            "chunks" in query
                            and isinstance(query["chunks"], dict)
                            and "$elemMatch" in query["chunks"]
                        ):
                            target_chunk_index = query["chunks"]["$elemMatch"].get("chunk_index")
                        elif "chunks.chunk_index" in query:
                            target_chunk_index = query["chunks.chunk_index"]

                        for c in doc.get("chunks", []):
                            if (
                                target_chunk_index is None
                                or c.get("chunk_index") == target_chunk_index
                            ):
                                c[field_name] = v
                                break
                    else:
                        doc[k] = v
                else:
                    doc[k] = v
        if "$max" in update:
            for k, v in update["$max"].items():
                if k not in doc or doc[k] < v:
                    doc[k] = v
        if "$inc" in update:
            for k, v in update["$inc"].items():
                doc[k] = doc.get(k, 0) + v


@pytest.fixture(autouse=True)
def reset_telegram_repositories():
    TelegramPersistenceManager._client = None
    TelegramPersistenceManager._correlations_init_future = None
    TelegramPersistenceManager._correlations_index_succeeded = False
    TelegramPersistenceManager._correlations_index_error = None
    TelegramPersistenceManager._outbox_init_future = None
    TelegramPersistenceManager._outbox_index_succeeded = False
    TelegramPersistenceManager._outbox_index_error = None
    yield


@pytest.mark.asyncio
async def test_separate_index_initialization_orders():
    t_repo = TelegramRepository()
    t_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    t_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    o_repo = NotificationOutboxRepository()
    o_repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")

    # Initialize TelegramRepository first
    await t_repo.ensure_indexes()
    assert TelegramPersistenceManager._correlations_index_succeeded is True
    assert TelegramPersistenceManager._outbox_index_succeeded is False
    assert len(t_repo.correlations_col.created_indexes) > 0
    assert len(o_repo.outbox_col.created_indexes) == 0

    # Now initialize OutboxRepository
    await o_repo.ensure_indexes()
    assert TelegramPersistenceManager._outbox_index_succeeded is True
    assert len(o_repo.outbox_col.created_indexes) > 0

    # Reset via close
    await TelegramPersistenceManager.close()
    assert TelegramPersistenceManager._correlations_index_succeeded is False
    assert TelegramPersistenceManager._outbox_index_succeeded is False


@pytest.mark.asyncio
async def test_make_bot_chat_key_hashes_token_and_chat():
    key1 = make_bot_chat_key("secret_token_123", "999888")
    assert "secret_token_123" not in key1
    assert "999888" not in key1
    assert key1.startswith("b_") and "_c_" in key1


@pytest.mark.asyncio
async def test_sensitive_security_challenge_detection():
    assert contains_sensitive_security_challenge("Please send your OTP code")
    assert contains_sensitive_security_challenge("Enter your 2FA verification-code")
    assert contains_sensitive_security_challenge("Solve this CAPTCHA puzzle")
    assert contains_sensitive_security_challenge("Enter your account password")
    assert not contains_sensitive_security_challenge("Are you willing to relocate to NYC?")


@pytest.mark.asyncio
async def test_sensitive_prompt_fails_closed():
    repo = TelegramRepository()
    repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    with pytest.raises(TelegramStorageError) as exc_info:
        await repo.register_intent(
            correlation_key="test:sensitive",
            purpose="test",
            run_id="run_1",
            prompt_text="Please enter your 2FA verification code",
        )
    assert "Sensitive security challenge detected" in str(exc_info.value)


@pytest.mark.asyncio
async def test_correlation_key_payload_conflict():
    repo = TelegramRepository()
    repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    res1 = await repo.register_intent(
        correlation_key="approval:run1:jobA",
        purpose="approval",
        run_id="run1",
        job_id="jobA",
        prompt_text="Approve job A edits?",
    )
    assert res1.registered is True
    assert res1.is_new is True

    res2 = await repo.register_intent(
        correlation_key="approval:run1:jobA",
        purpose="approval",
        run_id="run1",
        job_id="jobA",
        prompt_text="Different prompt text entirely!",
    )
    assert res2.registered is False
    assert res2.reason == "correlation_key_conflict_mismatched_payload"


@pytest.mark.asyncio
async def test_restart_uses_persisted_nonce_and_prompt_once():
    repo = TelegramRepository()
    repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    res1 = await repo.register_intent(
        correlation_key="approval:run1:jobA",
        purpose="approval",
        run_id="run1",
        job_id="jobA",
        prompt_text="Approve job A edits?",
    )
    assert res1.registered is True
    original_nonce = res1.nonce
    original_prompt = res1.prompt_text

    res2 = await repo.register_intent(
        correlation_key="approval:run1:jobA",
        purpose="approval",
        run_id="run1",
        job_id="jobA",
        prompt_text="Approve job A edits?",
    )
    assert res2.registered is True
    assert res2.is_new is False
    assert res2.nonce == original_nonce
    assert res2.prompt_text == original_prompt


@pytest.mark.asyncio
async def test_prompt_send_success_then_persistence_failure_marks_delivery_unknown():
    client = TelegramClient()
    client.telegram_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    client.telegram_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 501
        with patch.object(
            client.telegram_repo, "record_prompt_sent", new_callable=AsyncMock
        ) as mock_rec_prompt:
            mock_rec_prompt.return_value = False
            res = await client.send_and_wait_for_reply(
                correlation_key="form_qa:run1:job1:0:hash1",
                purpose="form_qa",
                run_id="run1",
                prompt_text="Relocate?",
                timeout=10,
            )
            assert res.status == CorrelationStatus.PROMPT_DELIVERY_UNKNOWN


@pytest.mark.asyncio
async def test_ambiguous_prompt_send_marks_delivery_unknown():
    client = TelegramClient()
    client.telegram_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    client.telegram_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.side_effect = TelegramTransportAmbiguousError(
            "Network timed out waiting for header"
        )
        res = await client.send_and_wait_for_reply(
            correlation_key="form_qa:run1:job1:0:hash_ambig",
            purpose="form_qa",
            run_id="run1",
            prompt_text="Relocate?",
            timeout=10,
        )
        assert res.status == CorrelationStatus.PROMPT_DELIVERY_UNKNOWN


@pytest.mark.asyncio
async def test_definite_api_reject_marks_prompt_failed_and_terminates():
    client = TelegramClient()
    client.telegram_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    client.telegram_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.side_effect = TelegramApiDefiniteRejectError("Chat not found")
        res = await client.send_and_wait_for_reply(
            correlation_key="form_qa:run1:job1:0:hash_rej",
            purpose="form_qa",
            run_id="run1",
            prompt_text="Relocate?",
            timeout=10,
        )
        assert res.status == CorrelationStatus.PROMPT_FAILED

        # Re-running immediately recognizes terminal PROMPT_FAILED status
        res2 = await client.send_and_wait_for_reply(
            correlation_key="form_qa:run1:job1:0:hash_rej",
            purpose="form_qa",
            run_id="run1",
            prompt_text="Relocate?",
            timeout=10,
        )
        assert res2.status == CorrelationStatus.PROMPT_FAILED


@pytest.mark.asyncio
async def test_waiter_lease_and_reply_consumption_clears_reply_text():
    repo = TelegramRepository()
    repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    await repo.register_intent(
        correlation_key="corr:1",
        purpose="approval",
        run_id="r1",
        prompt_text="Approve?",
    )

    lease = await repo.claim_waiter("corr:1", lease_duration_seconds=30)
    assert lease.acquired is True
    assert lease.lease_id is not None

    in_flight_ok = await repo.mark_prompt_in_flight("corr:1", lease.lease_id)
    assert in_flight_ok is True

    sent_ok = await repo.record_prompt_sent(
        "corr:1", prompt_message_id=100, lease_id=lease.lease_id
    )
    assert sent_ok is True

    rep_res = await repo.record_reply(
        correlation_key="corr:1",
        reply_text="approve this now",
        update_id=50,
        message_id=101,
        lease_id=lease.lease_id,
    )
    assert rep_res.accepted is True
    assert rep_res.status == CorrelationStatus.REPLIED

    cons_res = await repo.consume_reply("corr:1", lease_id=lease.lease_id)
    assert cons_res.consumed is True
    assert cons_res.reply_text == "approve this now"

    # Verify reply_text cleared in DB
    raw_doc = await repo.get_correlation("corr:1")
    assert raw_doc.status == CorrelationStatus.CONSUMED
    assert raw_doc.reply_text is None


@pytest.mark.asyncio
async def test_single_active_polling_waiter_per_bot_chat():

    repo = TelegramRepository()
    repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")
    key = make_bot_chat_key("tok", "chat")

    lease1 = await repo.claim_poll_lease(key, lease_duration_seconds=30)
    assert lease1 is not None

    # Second concurrent claim must return None
    lease2 = await repo.claim_poll_lease(key, lease_duration_seconds=30)
    assert lease2 is None

    released = await repo.release_poll_lease(key, lease1)
    assert released is True

    lease3 = await repo.claim_poll_lease(key, lease_duration_seconds=30)
    assert lease3 is not None


@pytest.mark.asyncio
async def test_long_wait_retains_and_renews_poll_lease():
    client = TelegramClient()
    client.telegram_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    client.telegram_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 200

        with patch("jobapply.utils.telegram.httpx.AsyncClient") as mock_http_cls:
            mock_ctx = AsyncMock()
            mock_http_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"ok": True, "result": []}
            mock_resp.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_resp
            mock_ctx.__aenter__.return_value = mock_http_client
            mock_ctx.__aexit__.return_value = False
            mock_http_cls.return_value = mock_ctx

            # Run with short timeout
            res = await client.send_and_wait_for_reply(
                correlation_key="poll:renew",
                purpose="approval",
                run_id="r1",
                prompt_text="Approve?",
                timeout=1,
            )
            assert res.status == CorrelationStatus.TIMED_OUT


@pytest.mark.asyncio
async def test_cursor_insert_infrastructure_failure_surfaces_error():
    repo = TelegramRepository()
    repo.cursors_col = AsyncMock()
    repo.cursors_col.insert_one.side_effect = RuntimeError("DB write error")
    repo.cursors_col.find_one.return_value = None

    with pytest.raises(TelegramStorageError) as exc_info:
        await repo.claim_poll_lease("bot_chat_err")
    assert "Cursor document initialization failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_save_cursor_requires_exact_lease_and_does_not_upsert():
    repo = TelegramRepository()
    repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")
    key = make_bot_chat_key("tok", "chat")

    lease1 = await repo.claim_poll_lease(key, lease_duration_seconds=30)
    assert lease1 is not None

    # Wrong lease cannot update cursor
    wrong_save = await repo.save_cursor(key, 100, lease_id="wrong_lease")
    assert wrong_save is False

    # Exact lease updates cursor
    exact_save = await repo.save_cursor(key, 100, lease_id=lease1)
    assert exact_save is True

    curr = await repo.get_cursor(key)
    assert curr == 100


@pytest.mark.asyncio
async def test_cursor_failure_leaves_reply_unconsumed_for_recovery():
    client = TelegramClient()
    client.telegram_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    client.telegram_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 200

        with patch("jobapply.utils.telegram.httpx.AsyncClient") as mock_http_cls:
            mock_ctx = AsyncMock()
            mock_http_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "ok": True,
                "result": [
                    {
                        "update_id": 99,
                        "message": {
                            "message_id": 201,
                            "chat": {"id": client.settings.telegram_chat_id},
                            "text": "approve",
                        },
                    }
                ],
            }
            mock_resp.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_resp
            mock_ctx.__aenter__.return_value = mock_http_client
            mock_ctx.__aexit__.return_value = False
            mock_http_cls.return_value = mock_ctx

            with patch.object(
                client.telegram_repo, "save_cursor", new_callable=AsyncMock
            ) as mock_save_cur:
                mock_save_cur.return_value = False
                res = await client.send_and_wait_for_reply(
                    correlation_key="appr:r1:j1",
                    purpose="approval",
                    run_id="r1",
                    prompt_text="Approve?",
                    timeout=5,
                )
                assert res.status == CorrelationStatus.STORAGE_ERROR
                assert "cursor_save_failed" in (res.error_reason or "")

                # Document remains in REPLIED state in DB for recovery
                doc = await client.telegram_repo.get_correlation("appr:r1:j1")
                assert doc.status == CorrelationStatus.REPLIED


@pytest.mark.asyncio
async def test_nonmatching_save_cursor_false_does_not_advance_in_memory_cursor():
    client = TelegramClient()
    client.telegram_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    client.telegram_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 200

        with patch("jobapply.utils.telegram.httpx.AsyncClient") as mock_http_cls:
            mock_ctx = AsyncMock()
            mock_http_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "ok": True,
                "result": [
                    {
                        "update_id": 99,
                        "message": {
                            "message_id": 201,
                            "chat": {"id": "wrong_chat"},
                            "text": "ignore me",
                        },
                    }
                ],
            }
            mock_resp.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_resp
            mock_ctx.__aenter__.return_value = mock_http_client
            mock_ctx.__aexit__.return_value = False
            mock_http_cls.return_value = mock_ctx

            with patch.object(
                client.telegram_repo, "save_cursor", new_callable=AsyncMock
            ) as mock_save_cur:
                mock_save_cur.return_value = False
                res = await client.send_and_wait_for_reply(
                    correlation_key="appr:nonmatching",
                    purpose="approval",
                    run_id="r1",
                    prompt_text="Approve?",
                    timeout=5,
                )
                assert res.status == CorrelationStatus.STORAGE_ERROR
                assert client._last_update_id != 99


@pytest.mark.asyncio
async def test_timeout_persistence_failure_returns_storage_error():
    client = TelegramClient()
    client.telegram_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    client.telegram_repo.cursors_col = InMemoryTelegramCollection(key_field="bot_chat_key")

    with patch.object(client, "send_message", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = 200

        with patch("jobapply.utils.telegram.httpx.AsyncClient") as mock_http_cls:
            mock_ctx = AsyncMock()
            mock_http_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"ok": True, "result": []}
            mock_resp.raise_for_status.return_value = None
            mock_http_client.post.return_value = mock_resp
            mock_ctx.__aenter__.return_value = mock_http_client
            mock_ctx.__aexit__.return_value = False
            mock_http_cls.return_value = mock_ctx

            with patch.object(
                client.telegram_repo, "mark_timed_out", new_callable=AsyncMock
            ) as mock_timeout:
                mock_timeout.return_value = False
                res = await client.send_and_wait_for_reply(
                    correlation_key="timeout:fail",
                    purpose="approval",
                    run_id="r1",
                    prompt_text="Approve?",
                    timeout=1,
                )
                assert res.status == CorrelationStatus.STORAGE_ERROR


@pytest.mark.asyncio
async def test_consumed_status_does_not_return_timeout():
    import hashlib

    from jobapply.utils.dedup import canonicalize_job_id

    question_text = "Q"
    run_id = "r1"
    job_id = "j1"
    ordinal = 0
    safe_run_id = str(run_id)
    canonical_job_id = canonicalize_job_id(job_id) or "unknown"
    q_hash = hashlib.sha256(question_text.strip().lower().encode("utf-8")).hexdigest()[:16]
    corr_key = f"form_qa:{safe_run_id}:{canonical_job_id}:{ordinal}:{q_hash}"

    repo = TelegramRepository()
    repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    repo.correlations_col.docs[corr_key] = {
        "correlation_key": corr_key,
        "purpose": "form_qa",
        "run_id": safe_run_id,
        "job_id": canonical_job_id,
        "prompt_hash": "hash",
        "nonce": "nonce1",
        "prompt_text": question_text,
        "prompt_in_flight": False,
        "prompt_message_id": 1,
        "status": CorrelationStatus.CONSUMED.value,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    client = TelegramClient()
    client.telegram_repo = repo

    with pytest.raises(FormQaInfrastructureError):
        await ask_user_for_question(
            question_text=question_text,
            job={"job_id": job_id},
            telegram=client,
            settings=MagicMock(form_qa_timeout_seconds=10, telegram_question_language="English"),
            run_id=run_id,
            ordinal=ordinal,
        )


@pytest.mark.asyncio
async def test_malformed_documents_fail_closed():

    t_repo = TelegramRepository()
    t_repo.correlations_col = InMemoryTelegramCollection(key_field="correlation_key")
    t_repo.correlations_col.docs["bad_corr"] = {
        "correlation_key": "bad_corr",
        "status": "invalid_status_value",
    }

    with pytest.raises(TelegramStorageError):
        await t_repo.get_correlation("bad_corr")

    o_repo = NotificationOutboxRepository()
    o_repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")
    o_repo.outbox_col.docs["bad_out"] = {"idempotency_key": "bad_out", "status": "not_real"}

    with pytest.raises(TelegramStorageError):
        await o_repo.get_record("bad_out")


@pytest.mark.asyncio
async def test_outbox_due_predicate_repeated_in_cas():
    repo = NotificationOutboxRepository()
    repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")
    await repo.enqueue_notification(
        idempotency_key="summary:due_cas",
        notification_type="session_summary",
        run_id="run_cas",
        payload_text="Summary text",
    )

    # Change next_attempt_at into future after discovery
    doc = repo.outbox_col.docs["summary:due_cas"]
    doc["next_attempt_at"] = datetime.now(timezone.utc) + timedelta(hours=1)

    claim = await repo.claim_due_records()
    assert claim.claimed is False


@pytest.mark.asyncio
async def test_outbox_configured_lease_is_honored():
    repo = NotificationOutboxRepository()
    repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")
    await repo.enqueue_notification(
        idempotency_key="summary:lease_test",
        notification_type="session_summary",
        run_id="run_lease",
        payload_text="Summary text",
    )

    claim = await repo.claim_due_records()
    assert claim.claimed is True
    rec = claim.records[0]
    assert rec.lease_expires_at is not None
    # Verify lease is roughly settings.outbox_lease_seconds (default 60s)
    remaining = (rec.lease_expires_at - datetime.now(timezone.utc)).total_seconds()
    assert 50 <= remaining <= 70


@pytest.mark.asyncio
async def test_outbox_mark_chunk_in_flight_checked_before_send():
    client = TelegramClient()
    client.outbox_repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")
    await client.outbox_repo.enqueue_notification(
        idempotency_key="summary:inflight_test",
        notification_type="session_summary",
        run_id="run_inf",
        payload_text="Summary text",
    )

    with patch.object(
        client.outbox_repo, "mark_chunk_in_flight", new_callable=AsyncMock
    ) as mock_inf:
        mock_inf.return_value = False
        with patch.object(client, "_send_single_message", new_callable=AsyncMock) as mock_send:
            drain = await client.drain_outbox()
            mock_send.assert_not_called()
            assert drain.sent_count == 0
            assert drain.unknown_count == 0


@pytest.mark.asyncio
async def test_outbox_one_unsent_chunk_blocks_mark_record_sent():
    repo = NotificationOutboxRepository()
    repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")

    long_payload = "A" * 5000
    await repo.enqueue_notification(
        idempotency_key="summary:two_chunks",
        notification_type="session_summary",
        run_id="run1",
        payload_text=long_payload,
    )
    claimed = await repo.claim_due_records()
    lease = claimed.records[0].lease_id

    # Mark ONLY chunk 0 sent
    await repo.mark_chunk_in_flight("summary:two_chunks", 0, lease)
    await repo.mark_chunk_sent("summary:two_chunks", 0, message_id=10, lease_id=lease)

    # Attempting to mark record sent must fail because chunk 1 is unsent
    sent_res = await repo.mark_record_sent("summary:two_chunks", lease_id=lease)
    assert sent_res is False

    rec = await repo.get_record("summary:two_chunks")
    assert rec.status == OutboxStatus.SENDING


@pytest.mark.asyncio
async def test_outbox_expired_sending_moves_to_delivery_unknown():
    repo = NotificationOutboxRepository()
    repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")
    await repo.enqueue_notification(
        idempotency_key="summary:expired",
        notification_type="session_summary",
        run_id="run_exp",
        payload_text="Summary text",
    )
    claimed = await repo.claim_due_records(lease_duration_seconds=10)
    assert claimed.claimed is True

    # Simulate expired lease in the past
    doc = repo.outbox_col.docs["summary:expired"]
    doc["lease_expires_at"] = datetime.now(timezone.utc) - timedelta(seconds=60)

    # Next claim pass reconciles expired SENDING records to DELIVERY_UNKNOWN
    await repo.claim_due_records()
    rec = await repo.get_record("summary:expired")
    assert rec.status == OutboxStatus.DELIVERY_UNKNOWN
    assert "lease_expired_during_send" in (rec.error_reason or "")


@pytest.mark.asyncio
async def test_outbox_retry_increments_attempts_atomically_on_definite_rejection():

    repo = NotificationOutboxRepository()
    repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")
    await repo.enqueue_notification(
        idempotency_key="summary:retry",
        notification_type="session_summary",
        run_id="run_retry",
        payload_text="Summary text",
    )
    claim1 = await repo.claim_due_records()
    lease = claim1.records[0].lease_id

    # Mark in-flight before reset
    await repo.mark_chunk_in_flight("summary:retry", 0, lease)

    failed_ok = await repo.mark_record_failed(
        "summary:retry",
        "chat not found",
        lease_id=lease,
        current_attempts=0,
        reset_in_flight_chunk=0,
    )
    assert failed_ok is True

    rec = await repo.get_record("summary:retry")
    assert rec.attempts == 1
    assert rec.status == OutboxStatus.PENDING
    assert rec.next_attempt_at is not None
    assert rec.chunks[0].in_flight is False


@pytest.mark.asyncio
async def test_outbox_ambiguous_transport_failure_moves_to_delivery_unknown():
    client = TelegramClient()
    client.outbox_repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")

    await client.outbox_repo.enqueue_notification(
        idempotency_key="summary:ambig",
        notification_type="session_summary",
        run_id="run_ambig",
        payload_text="Summary text",
    )

    with patch.object(client, "_send_single_message", new_callable=AsyncMock) as mock_send_single:
        mock_send_single.side_effect = TelegramTransportAmbiguousError("Connection reset by peer")
        drain_res = await client.drain_outbox()
        assert drain_res.unknown_count == 1
        assert drain_res.sent_count == 0

        rec = await client.outbox_repo.get_record("summary:ambig")
        assert rec.status == OutboxStatus.DELIVERY_UNKNOWN


@pytest.mark.asyncio
async def test_outbox_false_retry_or_unknown_transition_not_counted():
    client = TelegramClient()
    client.outbox_repo.outbox_col = InMemoryTelegramCollection(key_field="idempotency_key")
    await client.outbox_repo.enqueue_notification(
        idempotency_key="summary:false_trans",
        notification_type="session_summary",
        run_id="run_ft",
        payload_text="Summary text",
    )

    with patch.object(client, "_send_single_message", new_callable=AsyncMock) as mock_send_single:
        mock_send_single.side_effect = TelegramApiDefiniteRejectError("Rejected")
        with patch.object(
            client.outbox_repo, "mark_record_failed", new_callable=AsyncMock
        ) as mock_fail:
            mock_fail.return_value = False
            drain_res = await client.drain_outbox()
            assert drain_res.failed_count == 0


@pytest.mark.asyncio
async def test_approval_node_uses_durable_correlation_without_mock_fallbacks():
    state = {
        "run_id": "run_approval_101",
        "current_job": {"job_id": "job_123", "title": "ML Engineer", "company": "Acme"},
        "qualification_result": {"score": 0.9},
        "errors": [],
    }

    with patch("jobapply.nodes.approval.TelegramClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.send_and_wait_for_reply.return_value = CorrelationWaitResult(
            status=CorrelationStatus.REPLIED,
            reply_text="approve",
            nonce="appr_nonce",
        )
        mock_client_cls.return_value = mock_client

        with (
            patch("jobapply.nodes.approval.open", new_callable=MagicMock) as mock_open,
            patch("jobapply.nodes.approval.get_llm") as mock_llm_fn,
            patch("jobapply.nodes.approval.markdown_to_pdf", new_callable=AsyncMock),
            patch("jobapply.nodes.approval.os.path.exists", return_value=True),
        ):
            mock_file = MagicMock()
            mock_file.read.return_value = "Resume content"
            mock_open.return_value.__enter__.return_value = mock_file
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = MagicMock(content="Edited resume")
            mock_llm_fn.return_value = mock_llm

            state["proposed_edits"] = "Update skills"
            res = await approval_node(state)

            assert res["approval_status"] == "approved"
            assert res["approval_nonce"] == "appr_nonce"
            mock_client.send_and_wait_for_reply.assert_awaited_once()


@pytest.mark.asyncio
async def test_form_qa_uses_real_run_id_and_question_ordinal():
    telegram = AsyncMock()
    telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.REPLIED,
        reply_text="3 years",
    )
    settings = MagicMock(form_qa_timeout_seconds=30, telegram_question_language="English")

    with patch(
        "jobapply.nodes.execution.translate_question_for_telegram",
        new_callable=AsyncMock,
        return_value=("Years of Python experience?", []),
    ):
        with patch(
            "jobapply.nodes.execution.extract_answer_from_reply", new_callable=AsyncMock
        ) as mock_ext:
            mock_ext.return_value = "3"
            ans, timed_out = await ask_user_for_question(
                question_text="Years of Python experience?",
                job={"job_id": "job_456"},
                telegram=telegram,
                settings=settings,
                run_id="run_live_202",
                ordinal=2,
            )
            assert ans == "3"
            assert timed_out is False

            call_kwargs = telegram.send_and_wait_for_reply.call_args.kwargs
            assert "form_qa:run_live_202:job_456:2:" in call_kwargs["correlation_key"]


@pytest.mark.asyncio
async def test_form_qa_infrastructure_error_triggers_manual_review():
    telegram = AsyncMock()
    telegram.send_and_wait_for_reply.return_value = CorrelationWaitResult(
        status=CorrelationStatus.STORAGE_ERROR,
        error_reason="database_unavailable",
    )
    settings = MagicMock(form_qa_timeout_seconds=30, telegram_question_language="English")

    with pytest.raises(FormQaInfrastructureError):
        await ask_user_for_question(
            question_text="Relocate?",
            job={"job_id": "job_789"},
            telegram=telegram,
            settings=settings,
            run_id="run_infra",
            ordinal=0,
        )


@pytest.mark.asyncio
async def test_notification_node_reports_truthful_outbox_counts():
    state = {
        "run_id": "run_notif_303",
        "current_query_index": 0,
        "search_queries": ["ML Engineer"],
        "application_outcomes": [{"status": "submitted", "title": "SWE", "company": "Co"}],
        "account_safety_paused": True,
        "account_safety_barrier_type": "CAPTCHA",
        "account_safety_stage": "execution_pre_submit",
        "account_safety_reason": "Captcha challenge",
        "account_safety_url": "https://linkedin.com",
    }

    with patch("jobapply.nodes.notification.TelegramClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.enqueue_and_deliver.return_value = MagicMock(
            success=True,
            status=OutboxStatus.SENT,
        )
        mock_client.outbox_repo = AsyncMock()
        mock_client.outbox_repo.get_pending_and_unknown_counts.return_value = (0, 0)
        mock_client_cls.return_value = mock_client

        res = await notification_node(state)
        assert res["notification_sent"] is True
        assert res["outbox_pending_count"] == 0
        assert res["outbox_unknown_count"] == 0
        assert "account_safety_paused" not in res


@pytest.mark.asyncio
async def test_notification_node_handles_count_db_failure_honestly():
    state = {
        "run_id": "run_notif_404",
        "current_query_index": 0,
        "search_queries": ["ML Engineer"],
        "application_outcomes": [],
    }

    with patch("jobapply.nodes.notification.TelegramClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.enqueue_and_deliver.side_effect = RuntimeError("MongoDB connection lost")
        mock_client_cls.return_value = mock_client

        res = await notification_node(state)
        assert res["notification_sent"] is False
        assert "outbox_pending_count" not in res
        assert "outbox_unknown_count" not in res
        assert len(res["errors"]) == 1
        assert (
            "MongoDB connection lost" in res["errors"][0]
            or "Telegram summary notification persistence failure" in res["errors"][0]
        )
