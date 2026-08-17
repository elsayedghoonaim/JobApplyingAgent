"""Strictly offline unit tests for idempotent live-application attempts and atomic quota reservations."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobapply.models.application import (
    AttemptStatus,
    AttemptTransitionResult,
)
from jobapply.nodes.execution import execution_node
from jobapply.state import JobApplyState
from jobapply.utils.attempts import (
    AttemptError,
    AttemptRepository,
    AttemptTransitionError,
    QuotaRepository,
    QuotaReservationResult,
    get_utc_date_str,
    validate_attempt_transition,
)


@asynccontextmanager
async def mock_managed_browser_cm(browser, context):
    yield browser, context


def create_mock_page(page_close_side_effect: Optional[Exception] = None):
    """Create a fully-mocked Playwright page and modal hierarchy for execution tests."""
    page = MagicMock()
    page.is_closed = MagicMock(return_value=False)
    if page_close_side_effect:
        page.close = AsyncMock(side_effect=page_close_side_effect)
    else:
        page.close = AsyncMock()

    page.goto = AsyncMock(return_value=MagicMock(status=200))
    page.wait_for_selector = AsyncMock()
    page.evaluate = AsyncMock(return_value="")
    page.query_selector_all = AsyncMock(return_value=[])

    easy_apply_locator = MagicMock()
    easy_apply_locator.filter.return_value = easy_apply_locator
    easy_apply_locator.first = easy_apply_locator
    easy_apply_locator.click = AsyncMock()
    page.locator = MagicMock(return_value=easy_apply_locator)

    submit_button = MagicMock()
    submit_button.is_visible = AsyncMock(return_value=True)
    submit_button.is_enabled = AsyncMock(return_value=True)
    submit_button.inner_text = AsyncMock(return_value="Submit application")
    submit_button.get_attribute = AsyncMock(
        side_effect=lambda a: "Submit application" if a in ("aria-label", "name") else None
    )
    submit_button.click = AsyncMock()

    modal = MagicMock()
    modal.inner_text = AsyncMock(return_value="Easy Apply Form")

    def modal_qsa(selector: str):
        if "button" in selector:
            return [submit_button]
        return []

    modal.query_selector_all = AsyncMock(side_effect=modal_qsa)
    modal.query_selector = AsyncMock(return_value=None)
    page.query_selector = AsyncMock(return_value=modal)

    return page, submit_button, modal, easy_apply_locator


class InMemoryMongoCollection:
    """Thread/async-safe in-memory MongoDB collection mock for attempts and quotas."""

    def __init__(self):
        self.docs: dict[str, dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def create_index(self, key: Any, unique: bool = False):
        return "index_created"

    async def find_one(
        self, query: dict[str, Any], projection: Optional[dict[str, Any]] = None
    ) -> Optional[dict[str, Any]]:
        async with self.lock:
            for doc in self.docs.values():
                if self._matches(doc, query):
                    return self._project(dict(doc), projection)
            return None

    async def count_documents(self, query: dict[str, Any]) -> int:
        async with self.lock:
            count = 0
            for doc in self.docs.values():
                if self._matches(doc, query):
                    count += 1
            return count

    async def insert_one(self, doc: dict[str, Any]):
        async with self.lock:
            if "job_id" in doc:
                key = f"job_{doc['job_id']}"
                if key in self.docs:
                    raise RuntimeError("DuplicateKeyError: job_id already exists")
            elif "date" in doc:
                key = f"date_{doc['date']}"
                if key in self.docs:
                    raise RuntimeError("DuplicateKeyError: date already exists")
            else:
                key = f"doc_{len(self.docs) + 1}"

            inserted = self._deep_copy(doc)
            self.docs[key] = inserted
            mock_res = MagicMock()
            mock_res.inserted_id = key
            return mock_res

    async def update_one(
        self,
        filter_query: dict[str, Any],
        update_doc: dict[str, Any],
        upsert: bool = False,
    ):
        async with self.lock:
            matched_key = None
            for key, doc in self.docs.items():
                if self._matches(doc, filter_query):
                    matched_key = key
                    break

            res = MagicMock()
            if matched_key is not None:
                doc = self.docs[matched_key]
                self._apply_update(doc, update_doc)
                res.matched_count = 1
                res.modified_count = 1
                res.upserted_id = None
                return res

            if upsert:
                if "date" in filter_query:
                    date_val = filter_query["date"]
                    for d in self.docs.values():
                        if d.get("date") == date_val:
                            raise RuntimeError(f"DuplicateKeyError: date {date_val} already exists")

                new_doc: dict[str, Any] = {}
                if "date" in filter_query and isinstance(filter_query["date"], str):
                    new_doc["date"] = filter_query["date"]
                if "job_id" in filter_query and isinstance(filter_query["job_id"], str):
                    new_doc["job_id"] = filter_query["job_id"]

                if "$setOnInsert" in update_doc:
                    for k, v in update_doc["$setOnInsert"].items():
                        self._set_nested(new_doc, k, self._deep_copy(v))

                self._apply_update(new_doc, update_doc)
                key = f"doc_{len(self.docs) + 1}"
                self.docs[key] = new_doc
                res.matched_count = 0
                res.modified_count = 0
                res.upserted_id = key
                return res

            res.matched_count = 0
            res.modified_count = 0
            res.upserted_id = None
            return res

    def _matches(self, doc: dict[str, Any], query: dict[str, Any]) -> bool:
        for k, v in query.items():
            if k == "$and":
                if not all(self._matches(doc, subq) for subq in v):
                    return False
                continue
            if k == "$or":
                if not any(self._matches(doc, subq) for subq in v):
                    return False
                continue

            doc_val = self._get_nested(doc, k)
            if isinstance(v, dict):
                for op, expected in v.items():
                    if op == "$exists":
                        exists = doc_val is not None
                        if exists != expected:
                            return False
                    elif op == "$lt":
                        if doc_val is None or doc_val >= expected:
                            return False
                    elif op == "$lte":
                        if doc_val is None or doc_val > expected:
                            return False
                    elif op == "$gt":
                        if doc_val is None or doc_val <= expected:
                            return False
                    elif op == "$gte":
                        if doc_val is None or doc_val < expected:
                            return False
                    elif op == "$ne":
                        if doc_val == expected:
                            return False
                    elif op == "$in":
                        if doc_val not in expected:
                            return False
                    elif op == "$nin":
                        if doc_val in expected:
                            return False
            else:
                if doc_val != v:
                    return False
        return True

    def _apply_update(self, doc: dict[str, Any], update: dict[str, Any]):
        if "$set" in update:
            for k, v in update["$set"].items():
                self._set_nested(doc, k, self._deep_copy(v))
        if "$inc" in update:
            for k, v in update["$inc"].items():
                curr = self._get_nested(doc, k) or 0
                self._set_nested(doc, k, curr + v)

    def _get_nested(self, doc: dict[str, Any], path: str) -> Any:
        parts = path.split(".")
        curr = doc
        for part in parts:
            if not isinstance(curr, dict) or part not in curr:
                return None
            curr = curr[part]
        return curr

    def _set_nested(self, doc: dict[str, Any], path: str, val: Any):
        parts = path.split(".")
        curr = doc
        for part in parts[:-1]:
            if part not in curr or not isinstance(curr[part], dict):
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = val

    def _deep_copy(self, obj: Any) -> Any:
        import copy

        return copy.deepcopy(obj)

    def _project(self, doc: dict[str, Any], projection: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not projection:
            return doc
        res = {}
        for k in projection:
            if k in doc:
                res[k] = doc[k]
        return res


@pytest.fixture(autouse=True)
async def reset_repositories():
    await AttemptRepository.close()
    await QuotaRepository.close()
    yield
    await AttemptRepository.close()
    await QuotaRepository.close()


# =============================================================================
# 1. State Machine & Legal Transition Tests
# =============================================================================


def test_attempt_state_machine_transitions():
    """Verify all legal and illegal attempt transitions."""
    validate_attempt_transition(AttemptStatus.CREATED, AttemptStatus.QUOTA_RESERVED)
    validate_attempt_transition(AttemptStatus.CREATED, AttemptStatus.RELEASED)
    validate_attempt_transition(AttemptStatus.QUOTA_RESERVED, AttemptStatus.SUBMISSION_UNKNOWN)
    validate_attempt_transition(AttemptStatus.QUOTA_RESERVED, AttemptStatus.RELEASED)
    validate_attempt_transition(AttemptStatus.SUBMISSION_UNKNOWN, AttemptStatus.SUBMITTED)
    validate_attempt_transition(AttemptStatus.RELEASED, AttemptStatus.CREATED)

    validate_attempt_transition(AttemptStatus.CREATED, AttemptStatus.CREATED)
    validate_attempt_transition(AttemptStatus.SUBMISSION_UNKNOWN, AttemptStatus.SUBMISSION_UNKNOWN)
    validate_attempt_transition(AttemptStatus.SUBMITTED, AttemptStatus.SUBMITTED)

    # Illegal: CREATED -> SUBMISSION_UNKNOWN (must go through QUOTA_RESERVED)
    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.CREATED, AttemptStatus.SUBMISSION_UNKNOWN)

    # Illegal regressions
    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.SUBMISSION_UNKNOWN, AttemptStatus.CREATED)
    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.SUBMISSION_UNKNOWN, AttemptStatus.QUOTA_RESERVED)
    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.SUBMISSION_UNKNOWN, AttemptStatus.RELEASED)

    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.SUBMITTED, AttemptStatus.CREATED)
    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.SUBMITTED, AttemptStatus.QUOTA_RESERVED)
    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.SUBMITTED, AttemptStatus.SUBMISSION_UNKNOWN)
    with pytest.raises(AttemptTransitionError):
        validate_attempt_transition(AttemptStatus.SUBMITTED, AttemptStatus.RELEASED)


@pytest.mark.asyncio
async def test_created_to_mark_unknown_rejected_by_repository():
    """AttemptRepository.mark_unknown must reject attempts in CREATED status."""
    mock_coll = InMemoryMongoCollection()
    repo = AttemptRepository()
    repo.collection = mock_coll
    repo.__class__._index_succeeded = True

    claim = await repo.begin_attempt("job_created_direct", "run_1")
    assert claim.claimed is True
    assert claim.status == AttemptStatus.CREATED

    res = await repo.mark_unknown("job_created_direct", claim.attempt_id)
    assert res is False


@pytest.mark.asyncio
async def test_terminal_checks_reject_wrong_attempt_tokens():
    """mark_submitted and mark_released must reject wrong attempt tokens even on terminal records."""
    mock_coll = InMemoryMongoCollection()
    repo = AttemptRepository()
    repo.collection = mock_coll
    repo.__class__._index_succeeded = True

    job_id = "job_token_check"
    claim = await repo.begin_attempt(job_id, "run_1")
    assert claim.claimed is True
    await repo.reserve_quota_state(job_id, claim.attempt_id)
    await repo.mark_unknown(job_id, claim.attempt_id)
    sub = await repo.mark_submitted(job_id, claim.attempt_id)
    assert sub.success is True
    assert sub.changed is True

    # Duplicate call with SAME token -> success=True, changed=False
    sub_same = await repo.mark_submitted(job_id, claim.attempt_id)
    assert sub_same.success is True
    assert sub_same.changed is False

    # Call with WRONG token -> success=False, reason="attempt_token_mismatch"
    sub_wrong = await repo.mark_submitted(job_id, "att_wrong_token_999")
    assert sub_wrong.success is False
    assert sub_wrong.reason == "attempt_token_mismatch"


# =============================================================================
# 2. Missing-Day High-Concurrency Quota Reservation & _ensure_day_doc Tests
# =============================================================================


@pytest.mark.asyncio
async def test_missing_day_high_concurrency_reservation_reaches_exact_caps():
    """N concurrent workers reserving against a missing day document all succeed up to caps."""
    mock_coll = InMemoryMongoCollection()
    quota_repo = QuotaRepository()
    quota_repo.collection = mock_coll
    quota_repo.__class__._index_succeeded = True

    date_str = "2026-08-17"
    daily_cap = 10
    session_cap = 15
    total_workers = 20

    async def reserve_worker(idx: int) -> QuotaReservationResult:
        attempt_id = f"att_worker_{idx}"
        job_id = f"job_concurrent_{idx}"
        run_id = f"run_{idx % 3}"
        return await quota_repo.reserve_quota(
            attempt_id=attempt_id,
            job_id=job_id,
            run_id=run_id,
            daily_cap=daily_cap,
            session_cap=session_cap,
            date_str=date_str,
        )

    results = await asyncio.gather(*(reserve_worker(i) for i in range(total_workers)))

    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]

    assert len(successes) == daily_cap
    assert len(failures) == total_workers - daily_cap
    assert all(f.reason == "daily_cap_reached" for f in failures)
    assert await quota_repo.get_daily_consumed(date_str) == daily_cap


@pytest.mark.asyncio
async def test_ensure_day_doc_race_vs_infrastructure_failure():
    """_ensure_day_doc safely absorbs insert race if doc exists on re-read, but raises on real failure."""
    mock_coll = InMemoryMongoCollection()
    quota_repo = QuotaRepository()
    quota_repo.collection = mock_coll
    quota_repo.__class__._index_succeeded = True

    date_str = "2026-08-17"

    # 1. Simulated race: insert raises duplicate error, but document exists on re-read
    mock_coll.docs[f"date_{date_str}"] = {"date": date_str, "daily_reserved": 0}
    with patch.object(mock_coll, "update_one", side_effect=RuntimeError("DuplicateKeyError")):
        # Must not raise because day doc exists on re-read
        await quota_repo._ensure_day_doc(date_str)

    # 2. Simulated infra failure: update_one raises and document does NOT exist on re-read
    mock_coll.docs.clear()
    with patch.object(mock_coll, "update_one", side_effect=RuntimeError("MongoNetworkTimeout")):
        with pytest.raises(AttemptError) as exc_info:
            await quota_repo._ensure_day_doc(date_str)
        assert "Quota day initialization failed (RuntimeError)" in str(exc_info.value)


# =============================================================================
# 3. Quota Idempotency: Owner Verification, Duplicate Commit/Release
# =============================================================================


@pytest.mark.asyncio
async def test_quota_owner_verification_and_idempotent_commit_release():
    """Verify owner matching on pre-read and safe idempotent commit/release without duplicate counts."""
    mock_coll = InMemoryMongoCollection()
    quota_repo = QuotaRepository()
    quota_repo.collection = mock_coll
    quota_repo.__class__._index_succeeded = True

    date_str = "2026-08-17"
    attempt_id = "att_owner_test"
    correct_run = "run_owner_1"
    wrong_run = "run_owner_2"

    res = await quota_repo.reserve_quota(
        attempt_id=attempt_id,
        job_id="job_owner_1",
        run_id=correct_run,
        daily_cap=5,
        session_cap=5,
        date_str=date_str,
    )
    assert res.success is True

    # Wrong owner reserve pre-read fails
    res_wrong = await quota_repo.reserve_quota(
        attempt_id=attempt_id,
        job_id="job_owner_1",
        run_id=wrong_run,
        daily_cap=5,
        session_cap=5,
        date_str=date_str,
    )
    assert res_wrong.success is False
    assert res_wrong.reason == "reservation_owner_mismatch"

    # Wrong owner commit fails
    com_wrong = await quota_repo.commit_quota(attempt_id, wrong_run, date_str=date_str)
    assert com_wrong.success is False

    # Wrong owner release fails
    rel_wrong = await quota_repo.release_quota(attempt_id, wrong_run, date_str=date_str)
    assert rel_wrong.success is False

    # Correct commit succeeds once
    com1 = await quota_repo.commit_quota(attempt_id, correct_run, date_str=date_str)
    assert com1.success is True
    assert com1.changed is True

    # Duplicate commit is idempotent and does not increment again
    com2 = await quota_repo.commit_quota(attempt_id, correct_run, date_str=date_str)
    assert com2.success is True
    assert com2.changed is False
    assert com2.already_committed is True


# =============================================================================
# 4. Pre-Click Failure Cleanup & Outer Exception Net
# =============================================================================


@pytest.mark.asyncio
async def test_unexpected_exception_after_claim_triggers_exact_cleanup():
    """Unexpected exception after claim but before mark_unknown triggers exact cleanup and returns manual review."""
    state: JobApplyState = {
        "run_id": "test_run_unhandled_pre_click",
        "dry_run": False,
        "current_job": {
            "job_id": "job_unhandled_pre",
            "title": "Site Reliability Engineer",
            "company": "CloudCorp",
            "url": "https://www.linkedin.com/jobs/view/333",
        },
        "resume_path": __file__,
        "daily_application_cap": 50,
        "max_applications": 25,
        "applications_count": 0,
        "daily_applications_count": 0,
        "seen_job_ids": set(),
        "search_queries": ["SRE"],
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": 1,
        "job_listings": [],
        "current_job_index": 0,
        "search_failed": False,
        "qualification_result": None,
        "edits_urgent": False,
        "proposed_edits": None,
        "edit_reasoning": None,
        "cover_letter_text": None,
        "cover_letter_path": None,
        "approval_status": None,
        "approval_nonce": None,
        "application_status": None,
        "application_error": None,
        "form_qa_exchanges": [],
        "notification_sent": False,
        "account_safety_paused": False,
        "account_safety_barrier_type": None,
        "account_safety_reason": None,
        "account_safety_stage": None,
        "account_safety_url": None,
        "account_safety_detected_at": None,
        "account_safety_evidence": None,
        "account_safety_resume_instructions": None,
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
    }

    mock_attempt_coll = InMemoryMongoCollection()
    mock_quota_coll = InMemoryMongoCollection()

    page, submit_button, _, _ = create_mock_page()
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser = MagicMock()

    with (
        patch(
            "jobapply.nodes.execution.managed_browser",
            return_value=mock_managed_browser_cm(browser, context),
        ),
        patch("jobapply.nodes.execution.TelegramClient") as mock_tg_cls,
        patch("jobapply.nodes.execution.guard_page_account_safety", AsyncMock(return_value=None)),
        patch("jobapply.nodes.execution.asyncio.sleep", AsyncMock()),
        patch("jobapply.nodes.execution.get_randomized_delay", MagicMock(return_value=0)),
        patch.object(AttemptRepository, "ensure_indexes", AsyncMock(return_value=True)),
        patch.object(QuotaRepository, "ensure_indexes", AsyncMock(return_value=True)),
        # Simulate unexpected crash inside reserve_quota_state
        patch.object(
            AttemptRepository,
            "reserve_quota_state",
            AsyncMock(side_effect=RuntimeError("UnexpectedMongoMemoryError")),
        ),
    ):
        mock_tg = MagicMock()
        mock_tg.send_message = AsyncMock(return_value=123)
        mock_tg_cls.return_value = mock_tg

        with (
            patch.object(
                AttemptRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_attempt_coll) or setattr(self, "client", None)
                ),
            ),
            patch.object(
                QuotaRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_quota_coll) or setattr(self, "client", None)
                ),
            ),
        ):
            update = await execution_node(state)

    assert update["application_status"] == "needs_manual_review"
    submit_button.click.assert_not_called()

    # Attempt and quota were cleanly released
    doc = list(mock_attempt_coll.docs.values())[0]
    assert doc["status"] == AttemptStatus.RELEASED.value
    quota_repo = QuotaRepository()
    quota_repo.collection = mock_quota_coll
    assert await quota_repo.get_daily_consumed(get_utc_date_str()) == 0


@pytest.mark.asyncio
async def test_cleanup_failure_and_page_close_failure_reports_cleanup_required():
    """If cleanup fails and page.close fails, manual review is returned with bounded class names."""
    state: JobApplyState = {
        "run_id": "test_run_cleanup_and_close_fail",
        "dry_run": False,
        "current_job": {
            "job_id": "job_cleanup_and_close_fail",
            "title": "Security Engineer",
            "company": "SecureCorp",
            "url": "https://www.linkedin.com/jobs/view/444",
        },
        "resume_path": __file__,
        "daily_application_cap": 50,
        "max_applications": 25,
        "applications_count": 0,
        "daily_applications_count": 0,
        "seen_job_ids": set(),
        "search_queries": ["Sec"],
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": 1,
        "job_listings": [],
        "current_job_index": 0,
        "search_failed": False,
        "qualification_result": None,
        "edits_urgent": False,
        "proposed_edits": None,
        "edit_reasoning": None,
        "cover_letter_text": None,
        "cover_letter_path": None,
        "approval_status": None,
        "approval_nonce": None,
        "application_status": None,
        "application_error": None,
        "form_qa_exchanges": [],
        "notification_sent": False,
        "account_safety_paused": False,
        "account_safety_barrier_type": None,
        "account_safety_reason": None,
        "account_safety_stage": None,
        "account_safety_url": None,
        "account_safety_detected_at": None,
        "account_safety_evidence": None,
        "account_safety_resume_instructions": None,
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
    }

    mock_attempt_coll = InMemoryMongoCollection()
    mock_quota_coll = InMemoryMongoCollection()

    page, _, _, _ = create_mock_page(page_close_side_effect=RuntimeError("PageCloseNetworkDrop"))
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser = MagicMock()

    with (
        patch(
            "jobapply.nodes.execution.managed_browser",
            return_value=mock_managed_browser_cm(browser, context),
        ),
        patch("jobapply.nodes.execution.TelegramClient") as mock_tg_cls,
        patch("jobapply.nodes.execution.guard_page_account_safety", AsyncMock(return_value=None)),
        patch("jobapply.nodes.execution.asyncio.sleep", AsyncMock()),
        patch("jobapply.nodes.execution.get_randomized_delay", MagicMock(return_value=0)),
        patch.object(AttemptRepository, "ensure_indexes", AsyncMock(return_value=True)),
        patch.object(QuotaRepository, "ensure_indexes", AsyncMock(return_value=True)),
        patch.object(AttemptRepository, "reserve_quota_state", AsyncMock(return_value=False)),
        patch.object(
            AttemptRepository,
            "mark_released",
            AsyncMock(side_effect=RuntimeError("AttemptReleaseTimeout")),
        ),
    ):
        mock_tg = MagicMock()
        mock_tg.send_message = AsyncMock(return_value=123)
        mock_tg_cls.return_value = mock_tg

        with (
            patch.object(
                AttemptRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_attempt_coll) or setattr(self, "client", None)
                ),
            ),
            patch.object(
                QuotaRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_quota_coll) or setattr(self, "client", None)
                ),
            ),
        ):
            update = await execution_node(state)

    assert update["application_status"] == "needs_manual_review"
    assert "Cleanup required" in update["application_error"]
    assert "PageCloseError (RuntimeError)" in update["application_error"]


# =============================================================================
# 5. Already-Submitted (changed=False) and Post-Submit Truth Preservation
# =============================================================================


@pytest.mark.asyncio
async def test_changed_false_plus_close_failure_remains_skipped_already_submitted():
    """When mark_submitted returns changed=False and page.close raises, outcome remains skipped already_submitted."""
    state: JobApplyState = {
        "run_id": "test_run_changed_false_close_fail",
        "dry_run": False,
        "current_job": {
            "job_id": "job_changed_false",
            "title": "Lead Architect",
            "company": "ArcCorp",
            "url": "https://www.linkedin.com/jobs/view/555",
        },
        "resume_path": __file__,
        "daily_application_cap": 50,
        "max_applications": 25,
        "applications_count": 0,
        "daily_applications_count": 0,
        "seen_job_ids": set(),
        "search_queries": ["Arc"],
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": 1,
        "job_listings": [],
        "current_job_index": 0,
        "search_failed": False,
        "qualification_result": None,
        "edits_urgent": False,
        "proposed_edits": None,
        "edit_reasoning": None,
        "cover_letter_text": None,
        "cover_letter_path": None,
        "approval_status": None,
        "approval_nonce": None,
        "application_status": None,
        "application_error": None,
        "form_qa_exchanges": [],
        "notification_sent": False,
        "account_safety_paused": False,
        "account_safety_barrier_type": None,
        "account_safety_reason": None,
        "account_safety_stage": None,
        "account_safety_url": None,
        "account_safety_detected_at": None,
        "account_safety_evidence": None,
        "account_safety_resume_instructions": None,
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": [],
    }

    mock_attempt_coll = InMemoryMongoCollection()
    mock_quota_coll = InMemoryMongoCollection()

    page, _, _, _ = create_mock_page(page_close_side_effect=RuntimeError("BrowserCloseHang"))
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser = MagicMock()

    with (
        patch(
            "jobapply.nodes.execution.managed_browser",
            return_value=mock_managed_browser_cm(browser, context),
        ),
        patch("jobapply.nodes.execution.TelegramClient") as mock_tg_cls,
        patch("jobapply.nodes.execution.guard_page_account_safety", AsyncMock(return_value=None)),
        patch("jobapply.nodes.execution.asyncio.sleep", AsyncMock()),
        patch("jobapply.nodes.execution.get_randomized_delay", MagicMock(return_value=0)),
        patch(
            "jobapply.nodes.execution.wait_for_submission_or_safety",
            AsyncMock(return_value=(True, None)),
        ),
        patch.object(AttemptRepository, "ensure_indexes", AsyncMock(return_value=True)),
        patch.object(QuotaRepository, "ensure_indexes", AsyncMock(return_value=True)),
        patch.object(
            AttemptRepository,
            "mark_submitted",
            AsyncMock(
                return_value=AttemptTransitionResult(
                    success=True,
                    changed=False,
                    status=AttemptStatus.SUBMITTED,
                    attempt_id="att_test_changed_false",
                )
            ),
        ),
    ):
        mock_tg = MagicMock()
        mock_tg.send_message = AsyncMock(return_value=123)
        mock_tg_cls.return_value = mock_tg

        with (
            patch.object(
                AttemptRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_attempt_coll) or setattr(self, "client", None)
                ),
            ),
            patch.object(
                QuotaRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_quota_coll) or setattr(self, "client", None)
                ),
            ),
        ):
            update = await execution_node(state)

    assert update["application_status"] == "skipped"
    assert update["application_outcomes"][-1]["reason"] == "already_submitted"
    assert "applications_count" not in update
    assert any("PageCloseError" in err for err in update["errors"])


@pytest.mark.asyncio
async def test_primary_submitted_merges_earlier_and_later_secondary_errors():
    """Primary submitted plus earlier secondary error plus later close/trace error preserves all secondary errors."""
    state: JobApplyState = {
        "run_id": "test_run_merge_secondary",
        "dry_run": False,
        "current_job": {
            "job_id": "job_merge_sec",
            "title": "Principal Architect",
            "company": "EnterpriseCorp",
            "url": "https://www.linkedin.com/jobs/view/666",
        },
        "resume_path": __file__,
        "daily_application_cap": 50,
        "max_applications": 25,
        "applications_count": 0,
        "daily_applications_count": 0,
        "seen_job_ids": set(),
        "search_queries": ["Architect"],
        "current_query_index": 0,
        "current_page": 1,
        "pages_per_query": 1,
        "job_listings": [],
        "current_job_index": 0,
        "search_failed": False,
        "qualification_result": None,
        "edits_urgent": False,
        "proposed_edits": None,
        "edit_reasoning": None,
        "cover_letter_text": None,
        "cover_letter_path": None,
        "approval_status": None,
        "approval_nonce": None,
        "application_status": None,
        "application_error": None,
        "form_qa_exchanges": [],
        "notification_sent": False,
        "account_safety_paused": False,
        "account_safety_barrier_type": None,
        "account_safety_reason": None,
        "account_safety_stage": None,
        "account_safety_url": None,
        "account_safety_detected_at": None,
        "account_safety_evidence": None,
        "account_safety_resume_instructions": None,
        "max_jobs_to_evaluate": None,
        "jobs_evaluated_count": 0,
        "qualified_jobs_count": 0,
        "not_qualified_jobs_count": 0,
        "application_outcomes": [],
        "applied_jobs": [],
        "skipped_jobs": [],
        "logs": [],
        "errors": ["prior_session_error"],
    }

    mock_attempt_coll = InMemoryMongoCollection()
    mock_quota_coll = InMemoryMongoCollection()

    page, _, _, _ = create_mock_page(page_close_side_effect=RuntimeError("BrowserCloseDeadlock"))
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    browser = MagicMock()

    with (
        patch(
            "jobapply.nodes.execution.managed_browser",
            return_value=mock_managed_browser_cm(browser, context),
        ),
        patch("jobapply.nodes.execution.TelegramClient") as mock_tg_cls,
        patch("jobapply.nodes.execution.guard_page_account_safety", AsyncMock(return_value=None)),
        patch("jobapply.nodes.execution.asyncio.sleep", AsyncMock()),
        patch("jobapply.nodes.execution.get_randomized_delay", MagicMock(return_value=0)),
        patch(
            "jobapply.nodes.execution.wait_for_submission_or_safety",
            AsyncMock(return_value=(True, None)),
        ),
        patch.object(AttemptRepository, "ensure_indexes", AsyncMock(return_value=True)),
        patch.object(QuotaRepository, "ensure_indexes", AsyncMock(return_value=True)),
        # Quota commit fails as secondary error
        patch.object(
            QuotaRepository,
            "commit_quota",
            AsyncMock(side_effect=RuntimeError("SecondaryQuotaCommitTimeout")),
        ),
    ):
        mock_tg = MagicMock()
        mock_tg.send_message = AsyncMock(return_value=123)
        mock_tg_cls.return_value = mock_tg

        with (
            patch.object(
                AttemptRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_attempt_coll) or setattr(self, "client", None)
                ),
            ),
            patch.object(
                QuotaRepository,
                "__init__",
                lambda self: (
                    setattr(self, "collection", mock_quota_coll) or setattr(self, "client", None)
                ),
            ),
        ):
            update = await execution_node(state)

    assert update["application_status"] == "submitted"
    assert update["applications_count"] == 1
    # Check that prior errors, secondary quota commit errors, and page close errors are all preserved
    assert "prior_session_error" in update["errors"]
    assert any("Secondary quota commit error" in err for err in update["errors"])
    assert any("Post-submit error" in err or "PageCloseError" in err for err in update["errors"])
