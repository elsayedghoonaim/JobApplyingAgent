from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobapply.graph import (
    route_after_approval,
    route_after_generation,
    route_after_qualification,
    route_post_processing,
    route_select_next_job,
    route_start,
)
from jobapply.models.application import ApplicationStatus
from jobapply.nodes.execution import (
    execution_node,
    wait_for_submission_confirmation,
    wait_for_submission_or_safety,
)
from jobapply.nodes.notification import format_session_summary, notification_node
from jobapply.nodes.search import search_node
from jobapply.nodes.select_next_job import select_next_job_node
from jobapply.utils.account_safety import (
    AccountSafetyBarrierError,
    AccountSafetyBarrierType,
    AccountSafetyDetection,
    classify_http_status,
    classify_platform_message,
    classify_title,
    classify_url,
    format_account_safety_notification,
    inspect_page_account_safety,
    sanitize_evidence_string,
    sanitize_url_for_evidence,
)


@asynccontextmanager
async def mock_managed_browser_cm(browser, context):
    yield browser, context


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pure Classifier Matrix & False-Positive Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_classify_url_origin_and_path_aware():
    # True positives: LinkedIn paths on linkedin.com
    assert (
        classify_url("https://www.linkedin.com/checkpoint/challenge/captcha")[0]
        == AccountSafetyBarrierType.CAPTCHA
    )
    assert (
        classify_url("https://www.linkedin.com/checkpoint/challenge/totp")[0]
        == AccountSafetyBarrierType.TWO_FACTOR
    )
    assert (
        classify_url("https://www.linkedin.com/checkpoint/challenge/123")[0]
        == AccountSafetyBarrierType.CHECKPOINT
    )
    assert (
        classify_url("https://www.linkedin.com/identity/restricted")[0]
        == AccountSafetyBarrierType.RESTRICTION
    )
    assert classify_url("https://www.linkedin.com/authwall")[0] == AccountSafetyBarrierType.AUTHWALL
    assert classify_url("https://www.linkedin.com/429")[0] == AccountSafetyBarrierType.RATE_LIMIT

    # True positives: External approved CAPTCHA providers
    assert (
        classify_url("https://client-api.arkoselabs.com/fc/api/")[0]
        == AccountSafetyBarrierType.CAPTCHA
    )
    assert (
        classify_url("https://www.recaptcha.net/recaptcha/api.js")[0]
        == AccountSafetyBarrierType.CAPTCHA
    )
    assert (
        classify_url("https://www.google.com/recaptcha/enterprise.js")[0]
        == AccountSafetyBarrierType.CAPTCHA
    )
    assert classify_url("https://hcaptcha.com/1/api.js")[0] == AccountSafetyBarrierType.CAPTCHA

    # False-positive resistance: Segment-boundary safe (jobs containing security words must NOT trigger)
    assert classify_url("https://www.linkedin.com/jobs/view/login-engineer-12345") is None
    assert classify_url("https://www.linkedin.com/jobs/view/checkpoint-specialist-67890") is None
    assert classify_url("https://www.linkedin.com/jobs/view/rate-limit-analyst-11111") is None
    assert classify_url("https://www.linkedin.com/jobs/view/captcha-solver-developer") is None
    assert classify_url("https://www.linkedin.com/jobs/view/restriction-policy-lead") is None

    # False-positive resistance: Query parameters containing security terms must NOT trigger
    assert (
        classify_url("https://www.linkedin.com/jobs/search/?keywords=recaptcha&query=/login")
        is None
    )
    assert (
        classify_url("https://www.linkedin.com/jobs/view/12345/?trackingId=checkpoint_429") is None
    )
    assert classify_url("https://www.linkedin.com/jobs/collections/?q=totp_challenge") is None

    # False-positive resistance: Foreign origins containing path markers must NOT trigger
    assert classify_url("https://example.com/login") is None
    assert classify_url("https://example.com/checkpoint/challenge") is None
    assert classify_url("https://github.com/login") is None
    assert classify_url("https://gitlab.com/identity/restricted") is None

    # Invalid / non-http schemes
    assert classify_url("ftp://www.linkedin.com/checkpoint/") is None
    assert classify_url("") is None


def test_classify_http_status_semantics():
    # 429 rate limit is detected unconditionally
    res_429 = classify_http_status(429, "https://www.linkedin.com/jobs/search")
    assert res_429 is not None
    assert res_429[0] == AccountSafetyBarrierType.RATE_LIMIT

    # Plain 403 on a normal LinkedIn jobs page is NOT by itself proof of account restriction
    assert classify_http_status(403, "https://www.linkedin.com/jobs/view/12345") is None
    assert classify_http_status(403, "https://www.linkedin.com/feed") is None

    # 403 on a known restriction/checkpoint route IS classified
    res_403_route = classify_http_status(403, "https://www.linkedin.com/identity/restricted")
    assert res_403_route is not None
    assert res_403_route[0] == AccountSafetyBarrierType.RESTRICTION

    # 403 with strong restriction/challenge evidence text IS classified
    res_403_ev = classify_http_status(
        403,
        url="https://www.linkedin.com/jobs/view/123",
        page_evidence="Your account has been temporarily restricted",
    )
    assert res_403_ev is not None
    assert res_403_ev[0] == AccountSafetyBarrierType.RESTRICTION

    # Ordinary 4xx / 2xx statuses must NOT trigger
    assert classify_http_status(404, "https://www.linkedin.com/jobs/view/999") is None
    assert classify_http_status(410, "https://www.linkedin.com/jobs/view/999") is None
    assert classify_http_status(200, "https://www.linkedin.com/feed") is None
    assert classify_http_status(500, "https://www.linkedin.com/jobs") is None
    assert classify_http_status(None) is None


def test_classify_title():
    assert classify_title("Security Challenge | LinkedIn")[0] == AccountSafetyBarrierType.CAPTCHA
    assert classify_title("Two-Step Verification")[0] == AccountSafetyBarrierType.TWO_FACTOR
    assert (
        classify_title("Account Restricted | LinkedIn")[0] == AccountSafetyBarrierType.RESTRICTION
    )
    assert classify_title("429 Too Many Requests")[0] == AccountSafetyBarrierType.RATE_LIMIT
    assert classify_title("Sign In | LinkedIn")[0] == AccountSafetyBarrierType.AUTHWALL

    # False positive: Job title with security keywords
    assert classify_title("Cybersecurity Engineer | TechCorp | LinkedIn") is None
    assert classify_title("Information Security Specialist - Remote | LinkedIn") is None


def test_classify_platform_message():
    assert (
        classify_platform_message("Please solve this challenge to continue")[0]
        == AccountSafetyBarrierType.CAPTCHA
    )
    assert (
        classify_platform_message("Enter the 6-digit code sent to your phone")[0]
        == AccountSafetyBarrierType.TWO_FACTOR
    )
    assert (
        classify_platform_message("Your account has been temporarily restricted")[0]
        == AccountSafetyBarrierType.RESTRICTION
    )
    assert (
        classify_platform_message("HTTP 429 Too Many Requests")[0]
        == AccountSafetyBarrierType.RATE_LIMIT
    )
    assert (
        classify_platform_message("Please sign in to view this page")[0]
        == AccountSafetyBarrierType.AUTHWALL
    )

    # Job description text must not match
    assert (
        classify_platform_message("We are hiring a Senior Python Developer with AWS experience")
        is None
    )


def test_sanitize_url_for_evidence_masks_opaque_payloads():
    # Strip user credentials, query strings, fragments, AND mask opaque challenge payload / OTP / secrets
    dirty = (
        "https://user:pass@www.linkedin.com/checkpoint/challenge/totp/SECRET12345/928374?token=x#y"
    )
    sanitized = sanitize_url_for_evidence(dirty)
    assert "user:pass" not in sanitized
    assert "SECRET12345" not in sanitized
    assert "928374" not in sanitized
    assert "token=x" not in sanitized
    assert "#y" not in sanitized
    assert sanitized == "https://www.linkedin.com/checkpoint/challenge/totp/[REDACTED]"

    # Length bounded
    long_url = "https://www.linkedin.com/" + ("a" * 300)
    assert len(sanitize_url_for_evidence(long_url, max_length=50)) <= 50

    # Non-http fallback
    assert sanitize_url_for_evidence("javascript:alert(1)") == "https://www.linkedin.com"


def test_sanitize_evidence_string():
    raw = "Your one-time code is 849201 for verification."
    sanitized = sanitize_evidence_string(raw)
    assert "[REDACTED_CODE]" in sanitized
    assert "849201" not in sanitized
    assert len(sanitize_evidence_string("a" * 200, max_length=50)) <= 50


# ─────────────────────────────────────────────────────────────────────────────
# 2. Async Inspector & Centralized Guard Fail-Closed Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inspect_page_account_safety_fails_closed():
    mock_page = AsyncMock()
    mock_page.url = "https://www.linkedin.com/jobs/view/123"
    mock_page.title.side_effect = RuntimeError("Page crashed / disconnected")
    mock_page.query_selector = AsyncMock(return_value=None)
    mock_page.evaluate = AsyncMock(side_effect=RuntimeError("Page crashed / disconnected"))

    # When fail_closed=True (default), critical inspection unavailable fails closed with INSPECTION_UNAVAILABLE
    detection = await inspect_page_account_safety(
        mock_page, stage="execution_pre_submit", fail_closed=True
    )
    assert detection.detected is True
    assert detection.barrier_type == AccountSafetyBarrierType.INSPECTION_UNAVAILABLE
    assert "inspection unavailable" in (detection.reason or "").lower()

    # When fail_closed=False (e.g. interactive preflight polling), returns detected=False
    detection_open = await inspect_page_account_safety(
        mock_page, stage="preflight_signin", fail_closed=False
    )
    assert detection_open.detected is False


@pytest.mark.asyncio
async def test_guard_page_account_safety_raises_barrier_error():
    from jobapply.utils.account_safety import guard_page_account_safety

    mock_page = AsyncMock()
    mock_page.url = "https://www.linkedin.com/checkpoint/challenge/captcha"

    with pytest.raises(AccountSafetyBarrierError) as exc_info:
        await guard_page_account_safety(mock_page, stage="execution_pre_easy_apply_click")

    assert exc_info.value.detection.detected is True
    assert exc_info.value.detection.barrier_type == AccountSafetyBarrierType.CAPTCHA


@pytest.mark.asyncio
async def test_inspect_page_account_safety_http_status_detected():
    mock_page = AsyncMock()
    mock_page.url = "https://www.linkedin.com/jobs/search"
    mock_page.title.return_value = "Search | LinkedIn"

    detection = await inspect_page_account_safety(
        mock_page, stage="search_navigation", http_status=429
    )
    assert detection.detected is True
    assert detection.barrier_type == AccountSafetyBarrierType.RATE_LIMIT


# ─────────────────────────────────────────────────────────────────────────────
# 3. Search Node Pre-Action & Navigation Guard Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("jobapply.nodes.search.managed_browser")
async def test_search_node_pauses_immediately_on_navigation_barrier(mock_managed_browser):
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)

    mock_response = MagicMock()
    mock_response.status = 429
    mock_page.goto.return_value = mock_response
    mock_page.url = "https://www.linkedin.com/429"
    mock_page.title.return_value = "429 Too Many Requests"
    mock_context.new_page.return_value = mock_page

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "current_query_index": 0,
        "current_page": 1,
        "search_queries": ["python developer"],
        "pages_per_query": 1,
        "seen_job_ids": set(),
        "job_listings": [],
        "logs": [],
        "errors": [],
    }

    update = await search_node(state)

    assert update["account_safety_paused"] is True
    assert update["account_safety_barrier_type"] == AccountSafetyBarrierType.RATE_LIMIT.value
    assert update["search_failed"] is False
    assert update["job_listings"] == []
    mock_page.close.assert_awaited()


@pytest.mark.asyncio
@patch("jobapply.nodes.search.DeduplicationStore")
@patch("jobapply.nodes.search.managed_browser")
async def test_search_node_pauses_before_card_click_and_never_clicks(
    mock_managed_browser, mock_dedup_cls
):
    mock_store = AsyncMock()
    mock_store.find_seen_ids.return_value = set()
    mock_dedup_cls.return_value = mock_store

    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)

    mock_response = MagicMock()
    mock_response.status = 200
    mock_page.goto.return_value = mock_response
    mock_page.url = "https://www.linkedin.com/jobs/search"
    mock_page.title.return_value = "Jobs | LinkedIn"
    mock_context.new_page.return_value = mock_page

    mock_card = AsyncMock()
    mock_card.get_attribute = AsyncMock(return_value="job_123")
    mock_company_elem = AsyncMock()
    mock_company_elem.inner_text = AsyncMock(return_value="Acme Corp")
    mock_location_elem = AsyncMock()
    mock_location_elem.inner_text = AsyncMock(return_value="Remote")
    mock_link_elem = AsyncMock()
    mock_link_elem.inner_text = AsyncMock(return_value="Python Developer")

    async def link_get_attribute(attr):
        if attr == "aria-label":
            return "Python Developer"
        return "/jobs/view/123"

    mock_link_elem.get_attribute = link_get_attribute
    mock_link_elem.click = AsyncMock()

    async def card_query_selector(sel):
        if "job-card-container__link" in sel:
            return mock_link_elem
        if "artdeco-entity-lockup__subtitle" in sel:
            return mock_company_elem
        if "metadata-wrapper" in sel:
            return mock_location_elem
        return None

    mock_card.query_selector = card_query_selector
    mock_page.query_selector_all.return_value = [mock_card]
    mock_page.wait_for_selector = AsyncMock(return_value=mock_card)
    mock_page.query_selector = AsyncMock(return_value=mock_card)

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "current_query_index": 0,
        "current_page": 1,
        "search_queries": ["python developer"],
        "pages_per_query": 1,
        "seen_job_ids": set(),
        "job_listings": [],
        "logs": [],
        "errors": [],
    }

    with patch("jobapply.nodes.search.guard_page_account_safety") as mock_guard:

        async def guard_side_effect(page, stage="unknown", http_status=None, fail_closed=True):
            if stage == "search_pre_card_click":
                raise AccountSafetyBarrierError(
                    AccountSafetyDetection(
                        detected=True,
                        barrier_type=AccountSafetyBarrierType.CHECKPOINT,
                        reason="Challenge modal appeared before card click",
                        stage=stage,
                        url="https://www.linkedin.com/jobs/search",
                    )
                )

        mock_guard.side_effect = guard_side_effect

        update = await search_node(state)

        assert update["account_safety_paused"] is True
        assert update["account_safety_stage"] == "search_pre_card_click"
        # Verify card was NEVER clicked after detection
        mock_link_elem.click.assert_not_called()
        mock_page.close.assert_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Execution Node Pre-Action & Ordered Guard Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.managed_browser")
@patch("jobapply.nodes.execution.TelegramClient")
@patch("jobapply.nodes.execution.os.path.exists", return_value=True)
@patch("yaml.safe_load", return_value={})
async def test_execution_node_pauses_before_easy_apply_click(
    mock_yaml, mock_exists, mock_telegram_class, mock_managed_browser
):
    mock_telegram = AsyncMock()
    mock_telegram_class.return_value = mock_telegram

    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.url = "https://www.linkedin.com/jobs/view/123"
    mock_page.title.return_value = "Software Engineer | LinkedIn"
    mock_context.new_page.return_value = mock_page

    mock_btn_locator = MagicMock()
    mock_btn_locator.filter.return_value = mock_btn_locator
    mock_btn_locator.first = mock_btn_locator
    mock_btn_locator.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_btn_locator)
    mock_page.wait_for_selector = AsyncMock()

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "dry_run": False,
        "current_job": {
            "job_id": "123",
            "title": "Software Engineer",
            "company": "Tech Corp",
            "url": "https://www.linkedin.com/jobs/view/123",
        },
        "resume_path": "resume.pdf",
        "applications_count": 0,
        "daily_applications_count": 0,
        "daily_application_cap": 10,
        "application_outcomes": [],
        "logs": [],
        "errors": [],
    }

    with patch("jobapply.nodes.execution.guard_page_account_safety") as mock_guard:

        async def guard_side_effect(page, stage="unknown", http_status=None, fail_closed=True):
            if stage == "execution_pre_easy_apply_click":
                raise AccountSafetyBarrierError(
                    AccountSafetyDetection(
                        detected=True,
                        barrier_type=AccountSafetyBarrierType.TWO_FACTOR,
                        reason="2FA required before Easy Apply",
                        stage=stage,
                        url="https://www.linkedin.com/checkpoint/challenge/totp",
                    )
                )

        mock_guard.side_effect = guard_side_effect

        update = await execution_node(state)

        assert update["account_safety_paused"] is True
        assert update["account_safety_barrier_type"] == AccountSafetyBarrierType.TWO_FACTOR.value
        # Ensure Easy Apply button was never clicked
        mock_btn_locator.click.assert_not_called()
        mock_page.close.assert_awaited()


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.managed_browser")
@patch("jobapply.nodes.execution.TelegramClient")
@patch("jobapply.nodes.execution.os.path.exists", return_value=True)
@patch("yaml.safe_load", return_value={})
async def test_execution_node_pauses_top_of_form_step_and_never_fills(
    mock_yaml, mock_exists, mock_telegram_class, mock_managed_browser
):
    mock_telegram = AsyncMock()
    mock_telegram_class.return_value = mock_telegram

    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.url = "https://www.linkedin.com/jobs/view/123"
    mock_page.title.return_value = "Software Engineer | LinkedIn"
    mock_context.new_page.return_value = mock_page

    mock_btn_locator = MagicMock()
    mock_btn_locator.filter.return_value = mock_btn_locator
    mock_btn_locator.first = mock_btn_locator
    mock_btn_locator.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_btn_locator)

    mock_modal = AsyncMock()
    mock_modal.inner_text = AsyncMock(return_value="Easy Apply Form")
    mock_modal.query_selector_all = AsyncMock(return_value=[])
    mock_page.wait_for_selector = AsyncMock(return_value=mock_modal)
    mock_page.query_selector = AsyncMock(return_value=mock_modal)

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "dry_run": False,
        "current_job": {
            "job_id": "123",
            "title": "Software Engineer",
            "company": "Tech Corp",
            "url": "https://www.linkedin.com/jobs/view/123",
        },
        "resume_path": "resume.pdf",
        "applications_count": 0,
        "daily_applications_count": 0,
        "daily_application_cap": 10,
        "application_outcomes": [],
        "logs": [],
        "errors": [],
    }

    with patch("jobapply.nodes.execution.guard_page_account_safety") as mock_guard:

        async def guard_side_effect(page, stage="unknown", http_status=None, fail_closed=True):
            if stage == "execution_form_step_top":
                raise AccountSafetyBarrierError(
                    AccountSafetyDetection(
                        detected=True,
                        barrier_type=AccountSafetyBarrierType.RESTRICTION,
                        reason="Account restricted inside form",
                        stage=stage,
                        url="https://www.linkedin.com/identity/restricted",
                    )
                )

        mock_guard.side_effect = guard_side_effect

        update = await execution_node(state)

        assert update["account_safety_paused"] is True
        assert update["account_safety_barrier_type"] == AccountSafetyBarrierType.RESTRICTION.value
        mock_page.close.assert_awaited()


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.managed_browser")
@patch("jobapply.nodes.execution.TelegramClient")
@patch("jobapply.nodes.execution.os.path.exists", return_value=True)
@patch("yaml.safe_load", return_value={})
@patch("jobapply.nodes.execution.find_navigation_button")
async def test_execution_node_pre_submit_barrier_pauses_dry_run_without_success(
    mock_find_nav, mock_yaml, mock_exists, mock_telegram_class, mock_managed_browser
):
    """In dry-run mode, a pre-submit barrier must pause safely rather than reporting dry-run success."""
    mock_telegram = AsyncMock()
    mock_telegram_class.return_value = mock_telegram

    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.url = "https://www.linkedin.com/jobs/view/123"
    mock_page.title.return_value = "Software Engineer | LinkedIn"
    mock_context.new_page.return_value = mock_page

    mock_locator = MagicMock()
    mock_locator.filter.return_value = mock_locator
    mock_locator.first = mock_locator
    mock_locator.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)

    mock_modal = AsyncMock()
    mock_modal.inner_text = AsyncMock(return_value="Easy Apply Form")
    mock_modal.query_selector_all = AsyncMock(return_value=[])
    mock_page.wait_for_selector = AsyncMock(return_value=mock_modal)
    mock_page.query_selector = AsyncMock(return_value=mock_modal)
    mock_page.inner_text = AsyncMock(return_value="Easy Apply Form")

    mock_submit_btn = AsyncMock()
    mock_find_nav.return_value = (mock_submit_btn, "submit", "Submit application")

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "dry_run": True,  # DRY RUN
        "current_job": {
            "job_id": "123",
            "title": "Software Engineer",
            "company": "Tech Corp",
            "url": "https://www.linkedin.com/jobs/view/123",
        },
        "resume_path": "resume.pdf",
        "applications_count": 0,
        "daily_applications_count": 0,
        "daily_application_cap": 10,
        "application_outcomes": [],
        "logs": [],
        "errors": [],
    }

    with patch("jobapply.nodes.execution.guard_page_account_safety") as mock_guard:

        async def guard_side_effect(page, stage="unknown", http_status=None, fail_closed=True):
            if stage == "execution_pre_submit":
                raise AccountSafetyBarrierError(
                    AccountSafetyDetection(
                        detected=True,
                        barrier_type=AccountSafetyBarrierType.CAPTCHA,
                        reason="CAPTCHA challenge at pre-submit",
                        stage=stage,
                        url="https://www.linkedin.com/checkpoint/challenge/captcha",
                    )
                )

        mock_guard.side_effect = guard_side_effect

        update = await execution_node(state)

        # Must be paused on barrier, NOT reported as dry_run outcome
        assert update["account_safety_paused"] is True
        assert update["account_safety_barrier_type"] == AccountSafetyBarrierType.CAPTCHA.value
        assert update["application_status"] == ApplicationStatus.NEEDS_MANUAL_REVIEW.value
        assert update.get("applications_count") is None
        mock_submit_btn.click.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Delayed Post-Submit Monitoring & Ambiguity Handling Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_for_submission_or_safety_detects_delayed_checkpoint():
    mock_page = AsyncMock()
    mock_page.wait_for_function = AsyncMock(side_effect=TimeoutError("Not yet confirmed"))
    mock_page.evaluate = AsyncMock(return_value=False)
    mock_page.locator.return_value.inner_text = AsyncMock(return_value="processing...")

    poll_count = 0

    with patch("jobapply.nodes.execution.inspect_page_account_safety") as mock_inspect:

        async def inspect_poll(page, stage="unknown", fail_closed=True):
            nonlocal poll_count
            poll_count += 1
            if poll_count >= 2:
                return AccountSafetyDetection(
                    detected=True,
                    barrier_type=AccountSafetyBarrierType.CHECKPOINT,
                    reason="Delayed security checkpoint challenge",
                    stage=stage,
                )
            return AccountSafetyDetection(detected=False)

        mock_inspect.side_effect = inspect_poll

        confirmed, barrier = await wait_for_submission_or_safety(mock_page, timeout_ms=3000)

        assert confirmed is False
        assert barrier is not None
        assert barrier.detected is True
        assert barrier.barrier_type == AccountSafetyBarrierType.CHECKPOINT


@pytest.mark.asyncio
async def test_wait_for_submission_or_safety_confirms_success():
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=True)

    with patch("jobapply.nodes.execution.inspect_page_account_safety") as mock_inspect:
        mock_inspect.return_value = AccountSafetyDetection(detected=False)

        confirmed, barrier = await wait_for_submission_or_safety(mock_page, timeout_ms=3000)

        assert confirmed is True
        assert barrier is None


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.managed_browser")
@patch("jobapply.nodes.execution.TelegramClient")
@patch("jobapply.nodes.execution.os.path.exists", return_value=True)
@patch("yaml.safe_load", return_value={})
@patch("jobapply.nodes.execution.find_navigation_button")
async def test_execution_node_unconfirmed_timeout_halts_with_pause_and_notification_route(
    mock_find_nav, mock_yaml, mock_exists, mock_telegram_class, mock_managed_browser
):
    """If submission confirmation times out, the entire run must halt with a durable pause flag."""
    mock_telegram = AsyncMock()
    mock_telegram_class.return_value = mock_telegram

    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.url = "https://www.linkedin.com/jobs/view/123"
    mock_page.title.return_value = "Software Engineer | LinkedIn"
    mock_context.new_page.return_value = mock_page

    mock_locator = MagicMock()
    mock_locator.filter.return_value = mock_locator
    mock_locator.first = mock_locator
    mock_locator.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)

    mock_modal = AsyncMock()
    mock_modal.inner_text = AsyncMock(return_value="Easy Apply Form")
    mock_modal.query_selector_all = AsyncMock(return_value=[])
    mock_page.wait_for_selector = AsyncMock(return_value=mock_modal)
    mock_page.query_selector = AsyncMock(return_value=mock_modal)
    mock_page.inner_text = AsyncMock(return_value="Easy Apply Form")

    mock_submit_btn = AsyncMock()
    mock_submit_btn.is_enabled.return_value = True
    mock_submit_btn.click = AsyncMock()
    mock_find_nav.return_value = (mock_submit_btn, "submit", "Submit application")

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "dry_run": False,
        "current_job": {
            "job_id": "123",
            "title": "Software Engineer",
            "company": "Tech Corp",
            "url": "https://www.linkedin.com/jobs/view/123",
        },
        "resume_path": "resume.pdf",
        "applications_count": 0,
        "daily_applications_count": 0,
        "daily_application_cap": 10,
        "application_outcomes": [],
        "logs": [],
        "errors": [],
    }

    with patch("jobapply.nodes.execution.DeduplicationStore") as mock_dedup:
        mock_dedup_inst = AsyncMock()
        mock_dedup_inst.get_daily_count.return_value = 0
        mock_dedup.return_value = mock_dedup_inst

        with patch("jobapply.nodes.execution.inspect_page_account_safety") as mock_inspect:
            mock_inspect.return_value = AccountSafetyDetection(detected=False)

            with patch("jobapply.nodes.execution.wait_for_submission_or_safety") as mock_wait:
                # Timeout: unconfirmed outcome without specific barrier
                mock_wait.return_value = (False, None)

                with patch("jobapply.nodes.execution.take_error_screenshot") as mock_screenshot:
                    update = await execution_node(state)

                    assert update["account_safety_paused"] is True
                    assert (
                        update["account_safety_barrier_type"]
                        == AccountSafetyBarrierType.INSPECTION_UNAVAILABLE.value
                    )
                    assert (
                        update["application_status"] == ApplicationStatus.NEEDS_MANUAL_REVIEW.value
                    )
                    outcome = update["application_outcomes"][-1]
                    assert outcome["status"] == "needs_manual_review"
                    assert outcome.get("ambiguous_submission") is True
                    # Counters must NOT be incremented
                    assert update.get("applications_count") is None
                    assert update.get("daily_applications_count") is None
                    # Screenshots suppressed
                    mock_screenshot.assert_not_called()
                    # Graph router must halt to notification_node
                    assert route_post_processing(update) == "notification_node"


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.managed_browser")
@patch("jobapply.nodes.execution.TelegramClient")
@patch("jobapply.nodes.execution.os.path.exists", return_value=True)
@patch("yaml.safe_load", return_value={})
@patch("jobapply.nodes.execution.find_navigation_button")
async def test_execution_node_submit_click_exception_treated_as_ambiguous_review(
    mock_find_nav, mock_yaml, mock_exists, mock_telegram_class, mock_managed_browser
):
    """If an exception occurs during/after submit click, it must be marked as ambiguous review with durable pause."""
    mock_telegram = AsyncMock()
    mock_telegram_class.return_value = mock_telegram

    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.url = "https://www.linkedin.com/jobs/view/123"
    mock_page.title.return_value = "Software Engineer | LinkedIn"
    mock_context.new_page.return_value = mock_page

    mock_locator = MagicMock()
    mock_locator.filter.return_value = mock_locator
    mock_locator.first = mock_locator
    mock_locator.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)

    mock_modal = AsyncMock()
    mock_modal.inner_text = AsyncMock(return_value="Easy Apply Form")
    mock_modal.query_selector_all = AsyncMock(return_value=[])
    mock_page.wait_for_selector = AsyncMock(return_value=mock_modal)
    mock_page.query_selector = AsyncMock(return_value=mock_modal)
    mock_page.inner_text = AsyncMock(return_value="Easy Apply Form")

    mock_submit_btn = AsyncMock()
    mock_submit_btn.is_enabled.return_value = True
    mock_submit_btn.click.side_effect = RuntimeError("Browser connection severed during submit")
    mock_find_nav.return_value = (mock_submit_btn, "submit", "Submit application")

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "dry_run": False,
        "current_job": {
            "job_id": "123",
            "title": "Software Engineer",
            "company": "Tech Corp",
            "url": "https://www.linkedin.com/jobs/view/123",
        },
        "resume_path": "resume.pdf",
        "applications_count": 0,
        "daily_applications_count": 0,
        "daily_application_cap": 10,
        "application_outcomes": [],
        "logs": [],
        "errors": [],
    }

    with patch("jobapply.nodes.execution.DeduplicationStore") as mock_dedup:
        mock_dedup_inst = AsyncMock()
        mock_dedup_inst.get_daily_count.return_value = 0
        mock_dedup.return_value = mock_dedup_inst

        with patch("jobapply.nodes.execution.inspect_page_account_safety") as mock_inspect:
            mock_inspect.return_value = AccountSafetyDetection(detected=False)

            with patch("jobapply.nodes.execution.take_error_screenshot") as mock_screenshot:
                update = await execution_node(state)

                assert update["account_safety_paused"] is True
                assert (
                    update["account_safety_barrier_type"]
                    == AccountSafetyBarrierType.INSPECTION_UNAVAILABLE.value
                )
                assert update["application_status"] == ApplicationStatus.NEEDS_MANUAL_REVIEW.value
                outcome = update["application_outcomes"][-1]
                assert outcome["status"] == "needs_manual_review"
                assert outcome.get("ambiguous_submission") is True
                assert update.get("applications_count") is None
                # Screenshots must NOT be taken after ambiguous attempt
                mock_screenshot.assert_not_called()
                assert route_post_processing(update) == "notification_node"


@pytest.mark.asyncio
@patch("jobapply.nodes.execution.managed_browser")
@patch("jobapply.nodes.execution.TelegramClient")
@patch("jobapply.nodes.execution.os.path.exists", return_value=True)
@patch("yaml.safe_load", return_value={})
@patch("jobapply.nodes.execution.find_navigation_button")
async def test_execution_node_typed_barrier_error_through_broad_handler(
    mock_find_nav, mock_yaml, mock_exists, mock_telegram_class, mock_managed_browser
):
    """A typed AccountSafetyBarrierError must be handled cleanly as a paused update."""
    mock_telegram = AsyncMock()
    mock_telegram_class.return_value = mock_telegram

    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.is_closed = MagicMock(return_value=False)
    mock_page.url = "https://www.linkedin.com/jobs/view/123"
    mock_page.title.return_value = "Software Engineer | LinkedIn"
    mock_context.new_page.return_value = mock_page

    mock_locator = MagicMock()
    mock_locator.filter.return_value = mock_locator
    mock_locator.first = mock_locator
    mock_locator.click = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)

    mock_managed_browser.side_effect = lambda: mock_managed_browser_cm(mock_browser, mock_context)

    state = {
        "run_id": "test_run",
        "dry_run": False,
        "current_job": {
            "job_id": "123",
            "title": "Software Engineer",
            "company": "Tech Corp",
            "url": "https://www.linkedin.com/jobs/view/123",
        },
        "resume_path": "resume.pdf",
        "applications_count": 0,
        "daily_applications_count": 0,
        "daily_application_cap": 10,
        "application_outcomes": [],
        "logs": [],
        "errors": [],
    }

    with patch("jobapply.nodes.execution.guard_page_account_safety") as mock_guard:
        mock_guard.side_effect = AccountSafetyBarrierError(
            AccountSafetyDetection(
                detected=True,
                barrier_type=AccountSafetyBarrierType.TWO_FACTOR,
                reason="2FA challenge at navigation",
                stage="execution_navigation",
            )
        )

        update = await execution_node(state)

        assert update["account_safety_paused"] is True
        assert update["account_safety_barrier_type"] == AccountSafetyBarrierType.TWO_FACTOR.value
        assert update["application_status"] == ApplicationStatus.NEEDS_MANUAL_REVIEW.value


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph Routing & Notification Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_graph_routers_short_circuit_when_account_safety_paused():
    paused_state = {"account_safety_paused": True}
    assert route_start(paused_state) == "notification_node"
    assert route_select_next_job(paused_state) == "notification_node"
    assert route_after_qualification(paused_state) == "notification_node"
    assert route_after_generation(paused_state) == "notification_node"
    assert route_after_approval(paused_state) == "notification_node"
    assert route_post_processing(paused_state) == "notification_node"


@pytest.mark.asyncio
async def test_select_next_job_node_halts_on_paused_state():
    state = {
        "account_safety_paused": True,
        "job_listings": [{"job_id": "1"}, {"job_id": "2"}],
        "current_job_index": 0,
    }
    update = await select_next_job_node(state)
    assert update == {"current_job": None}


def test_format_session_summary_redacts_and_preserves_safety_instructions():
    state = {
        "run_id": "run_test_123",
        "account_safety_paused": True,
        "account_safety_barrier_type": "two_factor",
        "account_safety_stage": "execution_navigation",
        "account_safety_url": "https://user:pass@www.linkedin.com/checkpoint/challenge/totp?token=secret123",
        "account_safety_reason": "Enter your 6-digit code: 928374",
        "account_safety_resume_instructions": "Resolve in Microsoft Edge and start a NEW session (do not reuse the paused run ID).",
        "application_outcomes": [],
        "logs": [],
    }

    summary = format_session_summary(state)

    assert "🛑 ACCOUNT SAFETY PAUSED — HUMAN ACTION REQUIRED" in summary
    assert "TWO_FACTOR" in summary
    assert "https://www.linkedin.com/checkpoint/challenge/totp" in summary
    assert "secret123" not in summary
    assert "user:pass" not in summary
    assert ":pass@" not in summary
    assert "928374" not in summary
    assert "[REDACTED_CODE]" in summary
    assert "start a NEW session" in summary


@pytest.mark.asyncio
@patch("jobapply.nodes.notification.TelegramClient")
async def test_notification_node_handles_telegram_failure_without_unpausing(mock_telegram_class):
    mock_telegram = AsyncMock()
    mock_telegram.send_message.side_effect = RuntimeError("Network unreachable")
    mock_telegram_class.return_value = mock_telegram

    state = {
        "run_id": "test_run",
        "account_safety_paused": True,
        "account_safety_barrier_type": "captcha",
        "application_outcomes": [],
        "errors": [],
    }

    update = await notification_node(state)

    assert update["notification_sent"] is False
    assert any("Telegram summary notification failed" in err for err in update["errors"])
    # The pause in state must remain unmutated / intact
    assert state["account_safety_paused"] is True


def test_format_account_safety_notification():
    detection = AccountSafetyDetection(
        detected=True,
        barrier_type=AccountSafetyBarrierType.CHECKPOINT,
        reason="Security challenge required",
        stage="execution_navigation",
        url="https://www.linkedin.com/checkpoint/challenge/123",
    )
    alert = format_account_safety_notification("run_456", detection)
    assert "🛑 ACCOUNT SAFETY BARRIER DETECTED" in alert
    assert "CHECKPOINT" in alert
    assert "run_456" in alert
    assert "OPERATOR ACTION REQUIRED" in alert


def test_account_safety_barrier_error():
    detection = AccountSafetyDetection(
        detected=True,
        barrier_type=AccountSafetyBarrierType.TWO_FACTOR,
        reason="2FA required",
        stage="execution_pre_submit",
    )
    err = AccountSafetyBarrierError(detection)
    assert "two_factor" in str(err)
    assert err.detection == detection


@pytest.mark.asyncio
async def test_wait_for_submission_confirmation_wrapper():
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=True)

    with patch("jobapply.nodes.execution.inspect_page_account_safety") as mock_inspect:
        mock_inspect.return_value = AccountSafetyDetection(detected=False)
        confirmed = await wait_for_submission_confirmation(mock_page, timeout_ms=1000)
        assert confirmed is True
