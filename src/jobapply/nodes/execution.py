"""Execution node - Easy Apply automation with inline Telegram Q&A."""

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Optional

import yaml
from langsmith.run_helpers import trace

from jobapply.execution import (
    AUTO_SKIP_PATTERNS,
    CHOICE_PLACEHOLDERS,
    KNOWN_FIELD_PATTERNS,
    MODAL_CSS,
    MODAL_SELECTORS,
    STANDARD_TEXT_FIELD_SELECTOR,
    FormQaInfrastructureError,
    RequiredFieldValidationResult,
    UserSkippedJob,
    account_safety_execution_update,
    applied_update,
    choice_is_unanswered,
    classify_navigation_action,
    failed_update,
    find_already_applied_indicator,
    find_navigation_button,
    format_application_receipt,
    format_job_question_summary,
    get_auto_fill_value,
    get_choice_label,
    get_form_field_label,
    get_radio_option_label,
    is_auto_skip_field,
    is_known_field,
    is_required_field,
    is_skip_job_reply,
    manual_review_update,
    match_choice_index,
    select_live_role_radio_option,
    send_application_receipt,
    skipped_update,
    text_indicates_already_applied,
    validate_visible_required_controls,
    visible_button_labels,
)
from jobapply.execution.controls import (
    _fieldset_question_text,
)
from jobapply.execution.controls import (
    select_live_radio_option as _ext_select_live_radio_option,
)
from jobapply.execution.controls import (
    select_radio_option as _ext_select_radio_option,
)
from jobapply.execution.navigation import (
    wait_for_submission_confirmation as _ext_wait_for_submission_confirmation,
)
from jobapply.execution.navigation import (
    wait_for_submission_or_safety as _ext_wait_for_submission_or_safety,
)
from jobapply.execution.telegram_qa import (
    ask_user_for_question as _ext_ask_user_for_question,
)
from jobapply.execution.telegram_qa import (
    extract_answer_from_reply as _ext_extract_answer_from_reply,
)
from jobapply.execution.telegram_qa import (
    translate_question_for_telegram as _ext_translate_question_for_telegram,
)
from jobapply.models.application import AttemptStatus
from jobapply.nodes.outcomes import (
    append_application_outcome,
    build_application_outcome,
    resume_was_edited,
    state_list,
)
from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.account_safety import (
    SAFE_RESUME_INSTRUCTIONS,
    AccountSafetyBarrierError,
    AccountSafetyBarrierType,
    AccountSafetyDetection,
    guard_page_account_safety,
    sanitize_url_for_evidence,
)
from jobapply.utils.account_safety import (
    inspect_page_account_safety as _default_inspect_page_account_safety,
)
from jobapply.utils.attempts import AttemptRepository, QuotaRepository
from jobapply.utils.browser import get_randomized_delay, managed_browser, take_error_screenshot
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.limits import caps_reached
from jobapply.utils.llm import get_llm as _default_get_llm
from jobapply.utils.observability import bound_text, fingerprint, log_event
from jobapply.utils.telegram import TelegramClient
from jobapply.utils.tracing import get_execution_metadata, get_safe_job_metadata

# Module-level symbols maintained as patch targets
get_llm = _default_get_llm
select_radio_option = _ext_select_radio_option
inspect_page_account_safety = _default_inspect_page_account_safety

EASY_APPLY_SELECTOR = (
    "button:has-text('Easy Apply'), "
    "button.jobs-apply-button, "
    "a[aria-label*='LinkedIn Apply' i], "
    "a[href*='openSDUIApplyFlow=true']"
)
FIRST_FORM_MODAL_TIMEOUT_MS = 15_000


# Compatibility wrappers routing through module-level symbols for test mocking
async def select_live_radio_option(
    page: Any,
    question_text: str,
    option_text: str,
    attempts: int = 3,
) -> str:
    """Select a live radio using the module-level select_radio_option dependency."""
    return await _ext_select_live_radio_option(
        page,
        question_text,
        option_text,
        attempts=attempts,
        select_radio_option_fn=select_radio_option,
    )


async def translate_question_for_telegram(
    question_text: str,
    options: list[str] | None,
    target_language: str,
) -> tuple[str, list[str]]:
    """Translate a question using the module-level get_llm dependency."""
    return await _ext_translate_question_for_telegram(
        question_text,
        options,
        target_language,
        get_llm_fn=get_llm,
    )


async def extract_answer_from_reply(
    question_text: str,
    user_reply: str,
    options: list[str] | None = None,
    displayed_options: list[str] | None = None,
) -> str:
    """Extract an answer using the module-level get_llm dependency."""
    return await _ext_extract_answer_from_reply(
        question_text,
        user_reply,
        options=options,
        displayed_options=displayed_options,
        get_llm_fn=get_llm,
    )


async def ask_user_for_question(
    question_text: str,
    job: dict,
    telegram: TelegramClient,
    settings: Any,
    options: list[str] | None = None,
    run_id: str | None = None,
    ordinal: int = 0,
) -> tuple[str | None, bool]:
    """Ask a question using the module-level get_llm and translate/extract dependencies."""
    return await _ext_ask_user_for_question(
        question_text,
        job,
        telegram,
        settings,
        options=options,
        run_id=run_id,
        ordinal=ordinal,
        get_llm_fn=get_llm,
        translate_fn=translate_question_for_telegram,
        extract_fn=extract_answer_from_reply,
    )


async def wait_for_submission_or_safety(
    page: Any,
    timeout_ms: int = 12000,
) -> tuple[bool, Optional[AccountSafetyDetection]]:
    """Wait for submission or safety barriers using module-level inspect_page_account_safety."""
    return await _ext_wait_for_submission_or_safety(
        page,
        timeout_ms=timeout_ms,
        inspect_safety_fn=inspect_page_account_safety,
    )


async def wait_for_submission_confirmation(
    page: Any,
    timeout_ms: int = 12000,
) -> bool:
    """Wait for submission confirmation using module-level inspect_page_account_safety."""
    return await _ext_wait_for_submission_confirmation(
        page,
        timeout_ms=timeout_ms,
        inspect_safety_fn=inspect_page_account_safety,
    )


# Internal backward-compatibility aliases for outcome builders
_skipped_update = skipped_update
_manual_review_update = manual_review_update
_account_safety_execution_update = account_safety_execution_update
_failed_update = failed_update
_applied_update = applied_update

__all__ = [
    "AUTO_SKIP_PATTERNS",
    "CHOICE_PLACEHOLDERS",
    "KNOWN_FIELD_PATTERNS",
    "MODAL_CSS",
    "MODAL_SELECTORS",
    "STANDARD_TEXT_FIELD_SELECTOR",
    "FormQaInfrastructureError",
    "RequiredFieldValidationResult",
    "UserSkippedJob",
    "account_safety_execution_update",
    "append_application_outcome",
    "applied_update",
    "ask_user_for_question",
    "build_application_outcome",
    "choice_is_unanswered",
    "classify_navigation_action",
    "execution_node",
    "extract_answer_from_reply",
    "failed_update",
    "find_already_applied_indicator",
    "find_navigation_button",
    "format_application_receipt",
    "format_job_question_summary",
    "get_auto_fill_value",
    "get_choice_label",
    "get_form_field_label",
    "get_llm",
    "get_radio_option_label",
    "guard_page_account_safety",
    "inspect_page_account_safety",
    "is_auto_skip_field",
    "is_known_field",
    "is_required_field",
    "is_skip_job_reply",
    "managed_browser",
    "manual_review_update",
    "match_choice_index",
    "resume_was_edited",
    "select_live_radio_option",
    "select_live_role_radio_option",
    "select_radio_option",
    "send_application_receipt",
    "skipped_update",
    "state_list",
    "take_error_screenshot",
    "text_indicates_already_applied",
    "translate_question_for_telegram",
    "validate_visible_required_controls",
    "visible_button_labels",
    "wait_for_submission_confirmation",
    "wait_for_submission_or_safety",
]


async def execution_node(state: JobApplyState) -> dict:
    """Execute Easy Apply with form filling and inline Telegram Q&A.

    Args:
        state: Current graph state.

    Returns:
        State updates dict with application status and results.
    """
    settings = get_settings()
    current_job = state.get("current_job")
    run_id = str(state.get("run_id") or "default_run")
    form_qa_exchanges: list[dict] = []

    if caps_reached(state):
        return _skipped_update(
            state,
            "application_cap_reached",
            "Application cap reached before execution",
            form_qa_exchanges,
        )

    if not current_job:
        return _failed_update(state, "Execution failed: no current job selected", form_qa_exchanges)

    resume_path = state.get("resume_path")
    if not resume_path:
        return _failed_update(state, "Execution failed: resume_path is not set", form_qa_exchanges)
    if not os.path.exists(resume_path):
        return _failed_update(
            state, f"Execution failed: resume file not found: {resume_path}", form_qa_exchanges
        )

    telegram = TelegramClient()

    # Load profile for auto-fill
    profile_path = settings.resolve_data_path("profile.yaml")
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}
    except Exception as e:
        return _failed_update(
            state, f"Execution failed: profile load failed: {str(e)}", form_qa_exchanges
        )

    run_id = str(state.get("run_id") or "default_run")
    dry_run = bool(state.get("dry_run", True))
    job_id = str(current_job.get("job_id") or "")
    attempt_repo = AttemptRepository()
    quota_repo = QuotaRepository()

    # Read-only attempt preflight check before opening browser
    if not dry_run and job_id:
        try:
            preflight = await attempt_repo.preflight_check(job_id=job_id)
            if not preflight.can_proceed:
                if preflight.status == AttemptStatus.SUBMITTED:
                    return _skipped_update(
                        state,
                        "already_submitted",
                        f"Job {job_id} was already submitted in a prior run",
                        form_qa_exchanges,
                        extra={"job_id": job_id, "attempt_status": "submitted"},
                    )
                if preflight.status == AttemptStatus.SUBMISSION_UNKNOWN:
                    return await _manual_review_update(
                        state,
                        "submission_unknown_prior_attempt",
                        f"Job {job_id} has a prior submission_unknown attempt; manual review required before retrying",
                        form_qa_exchanges,
                        extra={
                            "job_id": job_id,
                            "attempt_status": "submission_unknown",
                            "ambiguous_submission": True,
                        },
                    )
                return _skipped_update(
                    state,
                    "concurrent_worker_active",
                    f"Job {job_id} has an active in-progress attempt ({preflight.reason})",
                    form_qa_exchanges,
                    extra={"job_id": job_id, "attempt_id": preflight.existing_attempt_id},
                )
        except Exception as exc:
            return _failed_update(
                state,
                f"Execution preflight check failed ({type(exc).__name__})",
                form_qa_exchanges,
            )

    application_submitted = False
    submission_attempted = False
    primary_submission_persisted = False
    already_submitted_terminal = False
    should_increment_counters = False
    attempt_id: Optional[str] = None
    quota_reserved = False
    secondary_errors: list[str] = []
    page = None

    async def _safe_close_page(p: Any) -> Optional[str]:
        """Safely close page and return a bounded class-only error if close fails."""
        if p is not None:
            try:
                if hasattr(p, "is_closed") and not p.is_closed():
                    await p.close()
                elif not hasattr(p, "is_closed"):
                    await p.close()
            except Exception as close_exc:
                return f"PageCloseError ({type(close_exc).__name__})"
        return None

    async def _cleanup_pre_click(reason: str, quota_was_reserved: bool) -> tuple[bool, list[str]]:
        cleanup_errs: list[str] = []
        quota_ok = True
        attempt_ok = True
        if quota_was_reserved and attempt_id:
            try:
                q_res = await quota_repo.release_quota(attempt_id, run_id)
                if not q_res.success and not q_res.already_released:
                    quota_ok = False
                    cleanup_errs.append(f"Pre-click quota release failed ({q_res.reason})")
            except Exception as q_exc:
                quota_ok = False
                cleanup_errs.append(f"Pre-click quota release exception ({type(q_exc).__name__})")
        if attempt_id:
            try:
                a_ok = await attempt_repo.mark_released(job_id, attempt_id, reason=reason)
                if not a_ok:
                    attempt_ok = False
                    cleanup_errs.append("Pre-click attempt release failed")
            except Exception as a_exc:
                attempt_ok = False
                cleanup_errs.append(f"Pre-click attempt release exception ({type(a_exc).__name__})")
        return (quota_ok and attempt_ok, cleanup_errs)

    try:
        async with trace(
            "easy_apply_execution",
            run_type="chain",
            metadata=get_safe_job_metadata(current_job, include_description=False),
        ) as run_tree:
            async with managed_browser() as (browser, context):
                page = await context.new_page()
                await telegram.send_message(
                    format_job_question_summary(
                        current_job,
                        state.get("qualification_result"),
                    )
                )

                log_event(
                    "info",
                    "execution.started",
                    f"\n🚀 [LIVE] Starting application process for '{current_job['title']}' at '{current_job['company']}'",
                    run_id=run_id,
                    job_id=job_id or None,
                    node="execution_node",
                )
                log_event(
                    "info",
                    "execution.job_url",
                    f"🔗 [LIVE] URL: {current_job['url']}",
                    run_id=run_id,
                    job_id=job_id or None,
                    node="execution_node",
                )

                # Navigate to job
                log_event(
                    "info",
                    "execution.navigating",
                    "[LIVE] Navigating to job page...",
                    run_id=run_id,
                    job_id=job_id or None,
                    node="execution_node",
                )
                response = await page.goto(current_job["url"], wait_until="domcontentloaded")
                http_status = response.status if response else None
                await asyncio.sleep(get_randomized_delay())

                # Guard job page navigation
                await guard_page_account_safety(
                    page, stage="execution_navigation", http_status=http_status
                )

                # Check for Easy Apply button
                try:
                    await page.wait_for_selector(EASY_APPLY_SELECTOR, timeout=5000)
                except Exception:
                    applied_evidence = await find_already_applied_indicator(page)
                    if applied_evidence:
                        log_event(
                            "info",
                            "execution.already_applied_indicator",
                            "[LIVE] LinkedIn shows this job was already applied to. Skipping safely.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                        )
                        await _safe_close_page(page)
                        update = _skipped_update(
                            state,
                            "already_applied",
                            "LinkedIn shows this job was already applied to",
                            form_qa_exchanges,
                            extra={"linkedin_evidence": applied_evidence},
                        )
                        errors = list(state.get("errors") or [])
                        try:
                            await DeduplicationStore().mark_seen(
                                current_job["job_id"],
                                {
                                    "status": "already_applied",
                                    "detected_applied_at": datetime.now(timezone.utc),
                                    "linkedin_evidence": applied_evidence,
                                },
                            )
                        except Exception as exc:
                            errors.append(f"Already-applied status persistence failed: {exc}")
                        receipt_error = await send_application_receipt(
                            telegram,
                            update["application_outcomes"][-1],
                            run_id=run_id,
                        )

                        if receipt_error:
                            errors.append(receipt_error)
                        if errors:
                            update["errors"] = errors
                        return update

                    log_event(
                        "warning",
                        "execution.no_easy_apply",
                        "❌ [LIVE] No Easy Apply button or applied status found. Skipping.",
                        run_id=run_id,
                        job_id=job_id or None,
                        node="execution_node",
                        outcome="skipped",
                    )
                    await _safe_close_page(page)
                    return _skipped_update(
                        state,
                        "no_easy_apply",
                        "No Easy Apply button or explicit applied status found",
                        form_qa_exchanges,
                    )

                # Guard immediately BEFORE clicking Easy Apply
                await guard_page_account_safety(page, stage="execution_pre_easy_apply_click")

                # Click Easy Apply
                log_event(
                    "info",
                    "execution.easy_apply_click",
                    "[LIVE] Clicking 'Easy Apply' button...",
                    run_id=run_id,
                    job_id=job_id or None,
                    node="execution_node",
                )
                await (
                    page.locator(EASY_APPLY_SELECTOR)
                    .filter(visible=True)
                    .first.click()
                )
                await asyncio.sleep(get_randomized_delay())

                # Guard opening Easy Apply
                await guard_page_account_safety(page, stage="execution_easy_apply_click")

                # Process form steps
                max_steps = 10  # safety limit
                for step in range(max_steps):
                    log_event(
                        "info",
                        "execution.form_step",
                        f"\n📝 [LIVE] Processing Page {step + 1}...",
                        run_id=run_id,
                        job_id=job_id or None,
                        node="execution_node",
                        details={"step": step + 1},
                    )

                    # Guard at top of each form step
                    await guard_page_account_safety(page, stage="execution_form_step_top")

                    # Wait for modal (longer timeout on first page)
                    modal_timeout = FIRST_FORM_MODAL_TIMEOUT_MS if step == 0 else 5000
                    modal_found = False
                    try:
                        await page.wait_for_selector(MODAL_CSS, timeout=modal_timeout)
                        modal_found = True
                    except Exception:
                        pass

                    if not modal_found:
                        # Debug: check if an "already applied" or success message appeared
                        try:
                            body_text = await page.evaluate(
                                "() => document.body.innerText.substring(0, 2000)"
                            )
                            if "already applied" in body_text.lower():
                                log_event(
                                    "info",
                                    "execution.already_applied_body",
                                    "[LIVE] Already applied to this job previously.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                )
                                await _safe_close_page(page)
                                return _skipped_update(
                                    state,
                                    "already_applied",
                                    "LinkedIn indicates this job was already applied to",
                                    form_qa_exchanges,
                                )
                            if (
                                "application submitted" in body_text.lower()
                                or "application sent" in body_text.lower()
                            ):
                                log_event(
                                    "info",
                                    "execution.submitted_body_detected",
                                    "✅ [LIVE] Application was submitted successfully!",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                )
                                application_submitted = True
                                break
                            visible_modals = await page.evaluate("""() => {
                                const modals = document.querySelectorAll('dialog, [role="dialog"], .artdeco-modal, [class*="modal"]');
                                return Array.from(modals).map(m => ({
                                    tag: m.tagName,
                                    classes: m.className.substring(0, 100),
                                    visible: m.offsetParent !== null,
                                    text: m.innerText.substring(0, 100)
                                }));
                            }""")
                            if visible_modals:
                                log_event(
                                    "debug",
                                    "execution.modal_inspection",
                                    f"[LIVE] Found {len(visible_modals)} dialog element(s) on page.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                    details={"dialog_count": len(visible_modals)},
                                )
                            else:
                                log_event(
                                    "debug",
                                    "execution.modal_inspection",
                                    "[LIVE] No modal/dialog elements found on page.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                )
                        except Exception as dbg_err:
                            log_event(
                                "debug",
                                "execution.modal_inspection_failed",
                                "[LIVE] Could not inspect page for diagnostics.",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                exc=dbg_err,
                            )

                        log_event(
                            "info",
                            "execution.modal_timeout",
                            f"[LIVE] Form modal not found after {modal_timeout}ms. Ending form loop.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                        )
                        break

                    # Get a reference to the modal element to scope queries
                    modal = await page.query_selector(MODAL_CSS)
                    if not modal:
                        modal = page  # Fallback to page if modal ref fails

                    # Check for external redirect or assessment
                    modal_text = await modal.inner_text()
                    if "external" in modal_text.lower() or "assessment" in modal_text.lower():
                        log_event(
                            "warning",
                            "execution.external_assessment",
                            "⚠️ [LIVE] Form requires external redirect or assessment. Needs manual review.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            outcome="needs_manual_review",
                        )
                        await _safe_close_page(page)
                        return await _manual_review_update(
                            state,
                            "external_or_assessment",
                            "External redirect or assessment required",
                            form_qa_exchanges,
                        )

                    # LinkedIn commonly places text/number questions before radio
                    # groups. Answer these first so a later choice cannot prevent
                    # earlier fields from reaching Telegram.
                    early_text_fields = []
                    for field in await modal.query_selector_all(STANDARD_TEXT_FIELD_SELECTOR):
                        if not await field.is_visible():
                            continue
                        try:
                            current_value = await field.input_value()
                        except Exception:
                            current_value = (await field.inner_text()).strip()
                        if not current_value:
                            early_text_fields.append(field)
                    if early_text_fields:
                        log_event(
                            "debug",
                            "execution.early_text_fields",
                            f"[LIVE] Found {len(early_text_fields)} unanswered "
                            "text/number field(s) before choice questions.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={"field_count": len(early_text_fields)},
                        )
                    for input_elem in early_text_fields:
                        label = await get_form_field_label(input_elem, page)
                        if is_auto_skip_field(label):
                            continue

                        auto_value = (
                            get_auto_fill_value(label, profile) if is_known_field(label) else None
                        )
                        if not auto_value:
                            log_event(
                                "info",
                                "execution.question_prompted",
                                "[LIVE] Sending form field question to operator.",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                details={
                                    "kind": "input",
                                    "ordinal": len(form_qa_exchanges),
                                    "question_fingerprint": fingerprint(label),
                                },
                            )
                            answer, timed_out = await ask_user_for_question(
                                label,
                                current_job,
                                telegram,
                                settings,
                                run_id=run_id,
                                ordinal=len(form_qa_exchanges),
                            )

                            form_qa_exchanges.append(
                                {
                                    "question": label,
                                    "answer": answer,
                                    "timed_out": timed_out,
                                }
                            )
                            if timed_out:
                                await _safe_close_page(page)
                                return _skipped_update(
                                    state,
                                    "form_qa_timeout",
                                    f"Q&A timeout on field: {label}",
                                    form_qa_exchanges,
                                    {"timeout_field": label},
                                )
                            auto_value = answer

                        if auto_value is not None:
                            log_event(
                                "debug",
                                "execution.field_fill",
                                "[LIVE] Filling form field.",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                details={"question_fingerprint": fingerprint(label)},
                            )
                            await input_elem.fill(str(auto_value))
                            await asyncio.sleep(0.3)

                    # 1. Handle Fieldsets (Radio Button Groups)
                    fieldsets = await modal.query_selector_all("fieldset")
                    if fieldsets:
                        log_event(
                            "debug",
                            "execution.fieldset_groups",
                            f"[LIVE] Found {len(fieldsets)} question group(s).",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={"group_count": len(fieldsets)},
                        )
                    fieldset_questions = []
                    for fieldset in fieldsets:
                        legend_text = await _fieldset_question_text(fieldset, page)

                        # Skip if already answered
                        checked_radio = await fieldset.query_selector("input[type='radio']:checked")
                        if checked_radio:
                            continue

                        radios = await fieldset.query_selector_all("input[type='radio']")
                        if not radios:
                            continue

                        # Extract option labels
                        option_labels = [
                            await get_radio_option_label(radio, fieldset) for radio in radios
                        ]
                        option_labels = [l_opt for l_opt in option_labels if l_opt]
                        if option_labels:
                            fieldset_questions.append((legend_text, option_labels))

                    for legend_text, option_labels in fieldset_questions:
                        # Match option
                        auto_answer = get_auto_fill_value(legend_text, profile)
                        selected_label = None
                        if auto_answer:
                            matched = match_choice_index(
                                str(auto_answer),
                                option_labels,
                            )
                            if matched is not None:
                                selected_label = option_labels[matched]

                        if not selected_label:
                            # Ask user via Telegram
                            prompt_question = legend_text
                            log_event(
                                "info",
                                "execution.question_prompted",
                                "[LIVE] Sending form choice question to operator.",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                details={
                                    "kind": "fieldset",
                                    "ordinal": len(form_qa_exchanges),
                                    "question_fingerprint": fingerprint(prompt_question),
                                    "option_count": len(option_labels),
                                },
                            )
                            answer, timed_out = await ask_user_for_question(
                                prompt_question,
                                current_job,
                                telegram,
                                settings,
                                options=option_labels,
                                run_id=run_id,
                                ordinal=len(form_qa_exchanges),
                            )

                            form_qa_exchanges.append(
                                {
                                    "question": prompt_question,
                                    "answer": answer,
                                    "timed_out": timed_out,
                                }
                            )

                            if timed_out:
                                log_event(
                                    "warning",
                                    "execution.qa_timeout",
                                    "❌ [LIVE] Q&A timed out. Skipping job.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                )
                                await _safe_close_page(page)
                                return _skipped_update(
                                    state,
                                    "form_qa_timeout",
                                    f"Q&A timeout on fieldset: {legend_text}",
                                    form_qa_exchanges,
                                    {"timeout_field": legend_text},
                                )

                            if answer is not None:
                                matched = match_choice_index(
                                    answer,
                                    option_labels,
                                )
                            if matched is not None:
                                selected_label = option_labels[matched]

                        if not selected_label:
                            await _safe_close_page(page)
                            return await _manual_review_update(
                                state,
                                "choice_answer_unmatched",
                                f"Could not match an answer for choice question: {legend_text}",
                                form_qa_exchanges,
                                {"choice_field": legend_text},
                            )

                        log_event(
                            "debug",
                            "execution.radio_selecting",
                            "[LIVE] Selecting fieldset radio option.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={
                                "question_fingerprint": fingerprint(legend_text),
                                "option_fingerprint": fingerprint(selected_label),
                            },
                        )
                        selection_method = await select_live_radio_option(
                            page,
                            legend_text,
                            selected_label,
                        )
                        log_event(
                            "debug",
                            "execution.radio_selected",
                            f"[LIVE] Radio selected through {selection_method}.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={"method": bound_text(selection_method, 60)},
                        )
                        await asyncio.sleep(0.3)

                    # LinkedIn may render choices with ARIA roles instead of native radios.
                    modal = await page.query_selector(MODAL_CSS) or page
                    role_groups = await page.query_selector_all("[role='radiogroup']")
                    role_group_questions = []
                    for group in role_groups:
                        if not await group.is_visible():
                            continue
                        if await group.query_selector("input[type='radio']"):
                            continue
                        role_options = await group.query_selector_all("[role='radio']")
                        if not role_options:
                            continue
                        selected_role_option = False
                        for option in role_options:
                            if (await option.get_attribute("aria-checked") or "").lower() == "true":
                                selected_role_option = True
                                break
                        if selected_role_option:
                            continue
                        labels = []
                        for option in role_options:
                            label = await option.get_attribute("aria-label")
                            if not label:
                                label = (await option.inner_text()).strip()
                            labels.append(label or "")
                        question = await get_choice_label(group, page)
                        role_group_questions.append((question, labels))

                    for question, labels in role_group_questions:
                        log_event(
                            "info",
                            "execution.question_prompted",
                            "[LIVE] Sending form choice question to operator.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={
                                "kind": "choice",
                                "ordinal": len(form_qa_exchanges),
                                "question_fingerprint": fingerprint(question),
                                "option_count": len(labels),
                            },
                        )
                        answer, timed_out = await ask_user_for_question(
                            question,
                            current_job,
                            telegram,
                            settings,
                            options=labels,
                            run_id=run_id,
                            ordinal=len(form_qa_exchanges),
                        )

                        form_qa_exchanges.append(
                            {
                                "question": question,
                                "answer": answer,
                                "timed_out": timed_out,
                            }
                        )
                        if timed_out:
                            await _safe_close_page(page)
                            return _skipped_update(
                                state,
                                "form_qa_timeout",
                                f"Q&A timeout on choice: {question}",
                                form_qa_exchanges,
                                {"timeout_field": question},
                            )
                        matched = match_choice_index(answer, labels) if answer is not None else None
                        if matched is None:
                            await _safe_close_page(page)
                            return await _manual_review_update(
                                state,
                                "choice_answer_unmatched",
                                f"Could not match an answer for choice question: {question}",
                                form_qa_exchanges,
                                {"choice_field": question},
                            )
                        log_event(
                            "debug",
                            "execution.choice_selecting",
                            "[LIVE] Selecting choice option.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={
                                "question_fingerprint": fingerprint(question),
                                "option_fingerprint": fingerprint(labels[matched]),
                            },
                        )
                        selection_method = await select_live_role_radio_option(
                            page,
                            question,
                            labels[matched],
                        )
                        log_event(
                            "debug",
                            "execution.choice_selected",
                            f"[LIVE] Choice selected through {selection_method}.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={"method": bound_text(selection_method, 60)},
                        )
                        await asyncio.sleep(0.3)

                    modal = await page.query_selector(MODAL_CSS) or page

                    # Handle custom dropdowns built from ARIA combobox/listbox widgets.
                    custom_combos = await modal.query_selector_all(
                        "[role='combobox']:not(select), [aria-haspopup='listbox']:not(select)"
                    )
                    for combo in custom_combos:
                        tag_name = await combo.evaluate("el => el.tagName.toLowerCase()")
                        current_value = (
                            await combo.input_value()
                            if tag_name in ("input", "textarea")
                            else (await combo.inner_text()).strip()
                        )
                        placeholder = await combo.get_attribute("placeholder")
                        if not choice_is_unanswered(current_value, placeholder):
                            continue

                        question = await get_choice_label(
                            combo,
                            page,
                            fallback=placeholder or "Choice question",
                        )
                        await combo.click()
                        await asyncio.sleep(0.3)
                        visible_options = []
                        for option in await page.query_selector_all("[role='option']"):
                            if await option.is_visible():
                                visible_options.append(option)
                        option_pairs = []
                        for option in visible_options:
                            label = (await option.inner_text()).strip()
                            if label:
                                option_pairs.append((option, label))
                        labels = [l_opt for _, l_opt in option_pairs]
                        if not labels:
                            await _safe_close_page(page)
                            return await _manual_review_update(
                                state,
                                "choice_options_not_found",
                                f"Could not read options for choice question: {question}",
                                form_qa_exchanges,
                                {"choice_field": question},
                            )

                        log_event(
                            "info",
                            "execution.question_prompted",
                            "[LIVE] Sending form dropdown question to operator.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={
                                "kind": "dropdown",
                                "ordinal": len(form_qa_exchanges),
                                "question_fingerprint": fingerprint(question),
                                "option_count": len(labels),
                            },
                        )
                        answer, timed_out = await ask_user_for_question(
                            question,
                            current_job,
                            telegram,
                            settings,
                            options=labels,
                            run_id=run_id,
                            ordinal=len(form_qa_exchanges),
                        )

                        form_qa_exchanges.append(
                            {
                                "question": question,
                                "answer": answer,
                                "timed_out": timed_out,
                            }
                        )
                        if timed_out:
                            await _safe_close_page(page)
                            return _skipped_update(
                                state,
                                "form_qa_timeout",
                                f"Q&A timeout on dropdown: {question}",
                                form_qa_exchanges,
                                {"timeout_field": question},
                            )
                        matched = match_choice_index(answer, labels) if answer is not None else None
                        if matched is None:
                            await _safe_close_page(page)
                            return await _manual_review_update(
                                state,
                                "choice_answer_unmatched",
                                f"Could not match an answer for dropdown: {label}",
                                form_qa_exchanges,
                                {"choice_field": label},
                            )
                        log_event(
                            "debug",
                            "execution.dropdown_selecting",
                            "[LIVE] Selecting custom dropdown option.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={
                                "question_fingerprint": fingerprint(question),
                                "option_fingerprint": fingerprint(labels[matched]),
                            },
                        )
                        await option_pairs[matched][0].click()
                        await asyncio.sleep(0.3)

                    # 2. Handle Text, Textarea, and Select elements
                    inputs = await modal.query_selector_all(
                        "input[type='text'], input[type='tel'], input[type='email'], input[type='number'], textarea, select"
                    )
                    empty_count = 0
                    for inp in inputs:
                        val = await inp.input_value()
                        if not val:
                            empty_count += 1
                    log_event(
                        "debug",
                        "execution.standard_fields",
                        f"[LIVE] Found {len(inputs)} standard fields on this page ({empty_count} empty).",
                        run_id=run_id,
                        job_id=job_id or None,
                        node="execution_node",
                        details={"field_count": len(inputs), "empty_count": empty_count},
                    )
                    for input_elem in inputs:
                        value = await input_elem.input_value()
                        tag_name = await input_elem.evaluate("el => el.tagName.toLowerCase()")
                        if tag_name == "select":
                            selected_text = await input_elem.evaluate(
                                "el => el.options[el.selectedIndex]?.textContent?.trim() || ''"
                            )
                            if not choice_is_unanswered(value, selected_text):
                                continue
                        elif value:
                            continue

                        label = await get_form_field_label(input_elem, page)

                        if is_auto_skip_field(label):
                            continue

                        log_event(
                            "debug",
                            "execution.field_inspected",
                            "[LIVE] Inspecting standard form field.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={
                                "field_type": bound_text(tag_name, 20),
                                "question_fingerprint": fingerprint(label),
                            },
                        )

                        auto_value = None
                        if is_known_field(label):
                            auto_value = get_auto_fill_value(label, profile)

                        # Process dropdown (select) vs inputs
                        if tag_name == "select":
                            options = await input_elem.query_selector_all("option")
                            option_values = []
                            for opt in options:
                                val = await opt.get_attribute("value")
                                txt = await opt.inner_text()
                                option_values.append((val, txt.strip()))

                            selectable_options = [
                                (val, txt)
                                for val, txt in option_values
                                if txt and not choice_is_unanswered(val, txt)
                            ]

                            val_to_select = None
                            if auto_value:
                                matched = match_choice_index(
                                    str(auto_value),
                                    [txt for _, txt in selectable_options],
                                )
                                if matched is not None:
                                    val, txt = selectable_options[matched]
                                    val_to_select = val or txt

                            if not val_to_select:
                                option_labels = [txt for _, txt in selectable_options]
                                prompt_question = label
                                log_event(
                                    "info",
                                    "execution.question_prompted",
                                    "[LIVE] Sending form dropdown question to operator.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                    details={
                                        "kind": "select_dropdown",
                                        "ordinal": len(form_qa_exchanges),
                                        "question_fingerprint": fingerprint(prompt_question),
                                        "option_count": len(option_labels),
                                    },
                                )
                                answer, timed_out = await ask_user_for_question(
                                    prompt_question,
                                    current_job,
                                    telegram,
                                    settings,
                                    options=option_labels,
                                    run_id=run_id,
                                    ordinal=len(form_qa_exchanges),
                                )

                                form_qa_exchanges.append(
                                    {
                                        "question": prompt_question,
                                        "answer": answer,
                                        "timed_out": timed_out,
                                    }
                                )
                                if timed_out:
                                    log_event(
                                        "warning",
                                        "execution.qa_timeout",
                                        "❌ [LIVE] Q&A timed out. Skipping job.",
                                        run_id=run_id,
                                        job_id=job_id or None,
                                        node="execution_node",
                                    )
                                    await _safe_close_page(page)
                                    return _skipped_update(
                                        state,
                                        "form_qa_timeout",
                                        f"Q&A timeout on dropdown: {label}",
                                        form_qa_exchanges,
                                        {"timeout_field": label},
                                    )
                                matched = (
                                    match_choice_index(answer, option_labels)
                                    if answer is not None
                                    else None
                                )
                                if matched is not None:
                                    val, txt = selectable_options[matched]
                                    val_to_select = val or txt

                            if not val_to_select:
                                await _safe_close_page(page)
                                return await _manual_review_update(
                                    state,
                                    "choice_answer_unmatched",
                                    f"Could not match an answer for dropdown: {label}",
                                    form_qa_exchanges,
                                    {"choice_field": label},
                                )

                            log_event(
                                "debug",
                                "execution.dropdown_selecting",
                                "[LIVE] Selecting dropdown option.",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                details={
                                    "question_fingerprint": fingerprint(label),
                                    "option_fingerprint": fingerprint(val_to_select),
                                },
                            )
                            await input_elem.select_option(value=val_to_select)
                            await asyncio.sleep(0.3)
                        else:
                            # Text input/textarea
                            _required = is_required_field(
                                await input_elem.get_attribute("required"),
                                await input_elem.get_attribute("aria-required"),
                                label,
                            )
                            if not auto_value:
                                log_event(
                                    "info",
                                    "execution.question_prompted",
                                    "[LIVE] Sending form field question to operator.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                    details={
                                        "kind": "input",
                                        "ordinal": len(form_qa_exchanges),
                                        "question_fingerprint": fingerprint(label),
                                    },
                                )
                                answer, timed_out = await ask_user_for_question(
                                    label,
                                    current_job,
                                    telegram,
                                    settings,
                                    run_id=run_id,
                                    ordinal=len(form_qa_exchanges),
                                )

                                form_qa_exchanges.append(
                                    {
                                        "question": label,
                                        "answer": answer,
                                        "timed_out": timed_out,
                                    }
                                )
                                if timed_out:
                                    log_event(
                                        "warning",
                                        "execution.qa_timeout",
                                        "❌ [LIVE] Q&A timed out. Skipping job.",
                                        run_id=run_id,
                                        job_id=job_id or None,
                                        node="execution_node",
                                    )
                                    await _safe_close_page(page)
                                    return _skipped_update(
                                        state,
                                        "form_qa_timeout",
                                        f"Q&A timeout on field: {label}",
                                        form_qa_exchanges,
                                        {"timeout_field": label},
                                    )
                                auto_value = answer

                            if auto_value:
                                source = (
                                    "Auto-filling"
                                    if is_known_field(label) and get_auto_fill_value(label, profile)
                                    else "User answer"
                                )
                                log_event(
                                    "debug",
                                    "execution.field_filled",
                                    f"[LIVE]   -> {source} form field.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                    details={
                                        "source": bound_text(source, 20),
                                        "question_fingerprint": fingerprint(label),
                                    },
                                )
                                await input_elem.fill(auto_value)
                                await asyncio.sleep(0.3)

                    # 3. Handle Checkboxes
                    checkboxes = await modal.query_selector_all("input[type='checkbox']")
                    for cb in checkboxes:
                        is_checked = await cb.is_checked()
                        if is_checked:
                            continue

                        cb_id = await cb.get_attribute("id")
                        cb_label = ""
                        if cb_id:
                            label_elem = await page.query_selector(f"label[for='{cb_id}']")
                            if label_elem:
                                cb_label = (await label_elem.inner_text()).strip()
                        if not cb_label:
                            try:
                                cb_label = (
                                    await cb.evaluate("el => el.parentElement.innerText")
                                ).strip()
                            except Exception:
                                pass

                        if any(
                            k in cb_label.lower() for k in ["agree", "terms", "privacy", "consent"]
                        ):
                            if settings.auto_accept_application_terms:
                                consent_granted = True
                            else:
                                answer, timed_out = await ask_user_for_question(
                                    f"Accept this application agreement? {cb_label}",
                                    current_job,
                                    telegram,
                                    settings,
                                    options=["Yes", "No"],
                                    run_id=run_id,
                                    ordinal=len(form_qa_exchanges),
                                )

                                form_qa_exchanges.append(
                                    {
                                        "question": cb_label,
                                        "answer": answer,
                                        "timed_out": timed_out,
                                    }
                                )
                                if timed_out:
                                    await _safe_close_page(page)
                                    return _skipped_update(
                                        state,
                                        "form_qa_timeout",
                                        f"Consent confirmation timed out: {cb_label}",
                                        form_qa_exchanges,
                                    )
                                consent_granted = bool(
                                    answer
                                    and answer.strip().lower()
                                    in {
                                        "yes",
                                        "y",
                                        "agree",
                                        "accept",
                                        "approved",
                                    }
                                )
                            if not consent_granted:
                                await _safe_close_page(page)
                                return _skipped_update(
                                    state,
                                    "consent_declined",
                                    f"Application agreement was not accepted: {cb_label}",
                                    form_qa_exchanges,
                                )
                            log_event(
                                "debug",
                                "execution.consent_checked",
                                "[LIVE] Checking approved agreement checkbox.",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                details={"question_fingerprint": fingerprint(cb_label)},
                            )
                            await cb.check()
                            await asyncio.sleep(0.3)
                        else:
                            required_choice = is_required_field(
                                await cb.get_attribute("required"),
                                await cb.get_attribute("aria-required"),
                                cb_label,
                            )
                            if cb_label or required_choice:
                                answer, timed_out = await ask_user_for_question(
                                    cb_label or "Select this required option?",
                                    current_job,
                                    telegram,
                                    settings,
                                    options=["Yes", "No"],
                                    run_id=run_id,
                                    ordinal=len(form_qa_exchanges),
                                )

                                form_qa_exchanges.append(
                                    {
                                        "question": cb_label,
                                        "answer": answer,
                                        "timed_out": timed_out,
                                    }
                                )
                                if timed_out:
                                    await _safe_close_page(page)
                                    return _skipped_update(
                                        state,
                                        "form_qa_timeout",
                                        f"Q&A timeout on checkbox: {cb_label}",
                                        form_qa_exchanges,
                                        {"timeout_field": cb_label},
                                    )
                                if answer and answer.strip().lower() in {
                                    "yes",
                                    "y",
                                    "true",
                                    "check",
                                    "selected",
                                }:
                                    await cb.check()
                                    await asyncio.sleep(0.3)
                                elif required_choice:
                                    await _safe_close_page(page)
                                    return await _manual_review_update(
                                        state,
                                        "required_choice_declined",
                                        f"Required checkbox was not selected: {cb_label}",
                                        form_qa_exchanges,
                                        {"choice_field": cb_label},
                                    )

                    # 4. Handle Resume/File upload
                    file_inputs = await modal.query_selector_all("input[type='file']")
                    resume_path = state.get("resume_path")
                    for file_input in file_inputs:
                        if resume_path and os.path.exists(resume_path):
                            log_event(
                                "debug",
                                "execution.resume_upload",
                                "[LIVE] Uploading resume file.",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                            )
                            await file_input.set_input_files(resume_path)
                            await asyncio.sleep(1)

                    # 5. Advance to the next step or submit.
                    next_btn, navigation_action, btn_text = await find_navigation_button(modal)

                    if next_btn:
                        # Validate all visible required controls before any consequential advance/submit click
                        val_res = await validate_visible_required_controls(modal, page)
                        if not val_res.is_valid:
                            reason_msg = val_res.reason or "Unresolved visible required field(s)"
                            log_event(
                                "warning",
                                "execution.required_fields_unresolved",
                                f"⚠️ [LIVE] {reason_msg}",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                details={"unresolved_count": len(val_res.unresolved_fields)},
                            )
                            await _safe_close_page(page)
                            return await _manual_review_update(
                                state,
                                "unresolved_required_fields",
                                reason_msg,
                                form_qa_exchanges,
                                extra={"unresolved_fields": val_res.unresolved_fields},
                            )

                        if navigation_action == "submit":
                            # Guard immediately BEFORE submit (and BEFORE dry-run branch / enabled checks)
                            await guard_page_account_safety(page, stage="execution_pre_submit")

                            # Dry run mode
                            if state["dry_run"]:
                                log_event(
                                    "info",
                                    "execution.dry_run_reached_submit",
                                    "[LIVE] Dry run mode - reached review/submit step. Closing modal without submitting.",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    node="execution_node",
                                    outcome="dry_run",
                                )
                                await _safe_close_page(page)
                                update = _applied_update(state, "dry_run", form_qa_exchanges)
                                try:
                                    await DeduplicationStore().mark_seen(
                                        current_job["job_id"],
                                        {
                                            "status": "dry_run",
                                            "dry_run_at": datetime.now(timezone.utc),
                                            "qa_count": len(form_qa_exchanges),
                                        },
                                    )
                                except Exception as exc:
                                    update["errors"] = list(state.get("errors") or []) + [
                                        f"Dry-run result persistence failed: {exc}"
                                    ]
                                receipt_error = await send_application_receipt(
                                    telegram,
                                    update["application_outcomes"][-1],
                                    run_id=run_id,
                                )

                                if receipt_error:
                                    update["errors"] = list(
                                        update.get("errors") or state.get("errors") or []
                                    ) + [receipt_error]
                                return update

                            if not await next_btn.is_enabled():
                                await _safe_close_page(page)
                                return await _manual_review_update(
                                    state,
                                    "submit_button_disabled",
                                    "Submit button is disabled; required fields may be incomplete",
                                    form_qa_exchanges,
                                )

                            # 1. Claim attempt ownership at the final submit boundary
                            claim = await attempt_repo.begin_attempt(
                                job_id=job_id,
                                run_id=run_id,
                                metadata={
                                    "title": current_job.get("title"),
                                    "company": current_job.get("company"),
                                    "url": current_job.get("url"),
                                },
                            )
                            if not claim.claimed:
                                await _safe_close_page(page)
                                if claim.status == AttemptStatus.SUBMITTED:
                                    return _skipped_update(
                                        state,
                                        "already_submitted",
                                        f"Job {job_id} was already submitted in a prior run",
                                        form_qa_exchanges,
                                        extra={"job_id": job_id, "attempt_status": "submitted"},
                                    )
                                if claim.status == AttemptStatus.SUBMISSION_UNKNOWN:
                                    return await _manual_review_update(
                                        state,
                                        "submission_unknown_prior_attempt",
                                        f"Job {job_id} has a prior submission_unknown attempt; manual review required before retrying",
                                        form_qa_exchanges,
                                        extra={
                                            "job_id": job_id,
                                            "attempt_status": "submission_unknown",
                                            "ambiguous_submission": True,
                                        },
                                    )
                                return _skipped_update(
                                    state,
                                    "concurrent_worker_active",
                                    f"Job {job_id} is currently claimed by run {claim.existing_run_id}",
                                    form_qa_exchanges,
                                    extra={
                                        "job_id": job_id,
                                        "existing_run_id": claim.existing_run_id,
                                    },
                                )

                            attempt_id = claim.attempt_id

                            # 2. Atomic conditional quota reservation
                            try:
                                reservation = await quota_repo.reserve_quota(
                                    attempt_id=attempt_id,
                                    job_id=job_id,
                                    run_id=run_id,
                                    daily_cap=int(
                                        state.get(
                                            "daily_application_cap", settings.daily_application_cap
                                        )
                                    ),
                                    session_cap=int(
                                        state.get(
                                            "max_applications",
                                            settings.max_applications_per_session,
                                        )
                                    ),
                                )
                            except Exception as q_err:
                                cleaned, c_errs = await _cleanup_pre_click(
                                    reason=f"reserve_quota_exception:{type(q_err).__name__}",
                                    quota_was_reserved=False,
                                )
                                close_err = await _safe_close_page(page)
                                if close_err:
                                    c_errs.append(close_err)
                                msg = f"Quota reservation failed ({type(q_err).__name__})"
                                if not cleaned:
                                    msg = f"{msg} (Cleanup required: {'; '.join(c_errs)})"
                                return await _manual_review_update(
                                    state,
                                    "quota_reservation_failed",
                                    msg,
                                    form_qa_exchanges,
                                    extra={
                                        "cleanup_errors": c_errs,
                                        "job_id": job_id,
                                        "attempt_id": attempt_id,
                                    },
                                )

                            if not reservation.success:
                                cleaned, c_errs = await _cleanup_pre_click(
                                    reason=reservation.reason or "cap_reached",
                                    quota_was_reserved=False,
                                )
                                close_err = await _safe_close_page(page)
                                if close_err:
                                    c_errs.append(close_err)
                                if not cleaned:
                                    return await _manual_review_update(
                                        state,
                                        "cleanup_required",
                                        f"Application cap reached but cleanup required: {'; '.join(c_errs)}",
                                        form_qa_exchanges,
                                        extra={
                                            "cleanup_errors": c_errs,
                                            "job_id": job_id,
                                            "attempt_id": attempt_id,
                                        },
                                    )
                                return _skipped_update(
                                    state,
                                    reservation.reason or "daily_cap_reached",
                                    "Application cap reached immediately before submission",
                                    form_qa_exchanges,
                                )

                            quota_reserved = True

                            # 3. Transition attempt to QUOTA_RESERVED
                            try:
                                state_ok = await attempt_repo.reserve_quota_state(
                                    job_id, attempt_id
                                )
                            except Exception:
                                state_ok = False

                            if not state_ok:
                                cleaned, c_errs = await _cleanup_pre_click(
                                    reason="reserve_quota_state_failed",
                                    quota_was_reserved=True,
                                )
                                close_err = await _safe_close_page(page)
                                if close_err:
                                    c_errs.append(close_err)
                                msg = "Failed to transition attempt to quota_reserved before submit"
                                if not cleaned:
                                    msg = f"{msg} (Cleanup required: {'; '.join(c_errs)})"
                                return await _manual_review_update(
                                    state,
                                    "reserve_quota_state_failed",
                                    msg,
                                    form_qa_exchanges,
                                    extra={
                                        "cleanup_errors": c_errs,
                                        "job_id": job_id,
                                        "attempt_id": attempt_id,
                                    },
                                )

                            # 4. Persist SUBMISSION_UNKNOWN durably BEFORE clicking submit
                            try:
                                persisted = await attempt_repo.mark_unknown(
                                    job_id=job_id,
                                    attempt_id=attempt_id,
                                    metadata={
                                        "title": current_job.get("title"),
                                        "company": current_job.get("company"),
                                        "url": current_job.get("url"),
                                        "qa_count": len(form_qa_exchanges),
                                    },
                                )
                                if not persisted:
                                    raise RuntimeError("Attempt mark_unknown returned False")
                            except Exception as exc:
                                cleaned, c_errs = await _cleanup_pre_click(
                                    reason=f"persist_unknown_failed:{type(exc).__name__}",
                                    quota_was_reserved=True,
                                )
                                close_err = await _safe_close_page(page)
                                if close_err:
                                    c_errs.append(close_err)
                                msg = f"Failed to persist submission_unknown record before click ({type(exc).__name__})"
                                if not cleaned:
                                    msg = f"{msg} (Cleanup required: {'; '.join(c_errs)})"
                                return await _manual_review_update(
                                    state,
                                    "persist_unknown_failed",
                                    msg,
                                    form_qa_exchanges,
                                    extra={
                                        "cleanup_errors": c_errs,
                                        "job_id": job_id,
                                        "attempt_id": attempt_id,
                                    },
                                )

                            submission_attempted = True
                            log_event(
                                "info",
                                "execution.submit_clicked",
                                "🚀 [LIVE] CLICKING SUBMIT - SUBMITTING APPLICATION!",
                                run_id=run_id,
                                job_id=job_id or None,
                                attempt_id=attempt_id,
                                node="execution_node",
                            )
                            await next_btn.click()

                            # 5. Monitor delayed post-submit safety barriers and confirmation
                            confirmed, post_barrier = await wait_for_submission_or_safety(page)
                            if post_barrier and post_barrier.detected:
                                log_event(
                                    "warning",
                                    "execution.post_submit_barrier",
                                    f"🛑 [LIVE] Account safety barrier detected after submit: {post_barrier.reason}",
                                    run_id=run_id,
                                    job_id=job_id or None,
                                    attempt_id=attempt_id,
                                    node="execution_node",
                                )
                                await _safe_close_page(page)
                                return await _account_safety_execution_update(
                                    state,
                                    post_barrier,
                                    current_job,
                                    form_qa_exchanges,
                                    pre_submit=False,
                                )

                            if not confirmed:
                                await _safe_close_page(page)
                                unconfirmed_detection = AccountSafetyDetection(
                                    detected=True,
                                    barrier_type=AccountSafetyBarrierType.INSPECTION_UNAVAILABLE,
                                    reason="Submit was clicked but LinkedIn confirmation was unconfirmed within timeout",
                                    stage="execution_post_submit_timeout",
                                    url=sanitize_url_for_evidence(getattr(page, "url", "")),
                                    detected_at=datetime.now(timezone.utc).isoformat(),
                                    resume_instructions=SAFE_RESUME_INSTRUCTIONS,
                                )
                                return await _account_safety_execution_update(
                                    state,
                                    unconfirmed_detection,
                                    current_job,
                                    form_qa_exchanges,
                                    pre_submit=False,
                                )

                            # 6. Explicit confirmation received -> primary durable submitted transition
                            try:
                                sub_result = await attempt_repo.mark_submitted(
                                    job_id=job_id,
                                    attempt_id=attempt_id,
                                    metadata={
                                        "title": current_job.get("title"),
                                        "company": current_job.get("company"),
                                        "url": current_job.get("url"),
                                        "qa_count": len(form_qa_exchanges),
                                        "resume_edited": resume_was_edited(state),
                                    },
                                )
                                if not sub_result.success:
                                    raise RuntimeError(
                                        f"mark_submitted failed ({sub_result.reason})"
                                    )
                            except Exception as db_err:
                                await _safe_close_page(page)
                                return await _manual_review_update(
                                    state,
                                    "submission_confirmed_persistence_failed",
                                    f"Application confirmed on LinkedIn but primary database transition failed ({type(db_err).__name__})",
                                    form_qa_exchanges,
                                    extra={
                                        "ambiguous_submission": True,
                                        "job_id": job_id,
                                        "attempt_id": attempt_id,
                                    },
                                )

                            if not sub_result.changed:
                                already_submitted_terminal = True
                                close_err = await _safe_close_page(page)
                                if close_err:
                                    secondary_errors.append(close_err)
                                update = _skipped_update(
                                    state,
                                    "already_submitted",
                                    f"Job {job_id} was already marked submitted",
                                    form_qa_exchanges,
                                    extra={"job_id": job_id, "attempt_status": "submitted"},
                                )
                                if secondary_errors:
                                    update["errors"] = (
                                        list(state.get("errors") or []) + secondary_errors
                                    )
                                return update

                            primary_submission_persisted = True
                            application_submitted = True
                            should_increment_counters = True

                            # 7. Secondary reconciliation writes
                            try:
                                commit_res = await quota_repo.commit_quota(attempt_id, run_id)
                                if not commit_res.success and not commit_res.already_committed:
                                    secondary_errors.append(
                                        f"Secondary quota commit failed: {commit_res.reason}"
                                    )
                            except Exception as exc:
                                secondary_errors.append(
                                    f"Secondary quota commit error: {type(exc).__name__}"
                                )

                            try:
                                await DeduplicationStore().mark_seen(
                                    job_id,
                                    {
                                        "status": "submitted",
                                        "applied_at": datetime.now(timezone.utc),
                                        "qa_count": len(form_qa_exchanges),
                                        "resume_edited": resume_was_edited(state),
                                    },
                                )
                            except Exception as exc:
                                secondary_errors.append(
                                    f"Secondary dedup mark_seen error: {type(exc).__name__}"
                                )

                            break
                        else:
                            # Next/Review button
                            # Guard immediately BEFORE clicking Next/Review
                            await guard_page_account_safety(page, stage="execution_pre_step_click")

                            log_event(
                                "debug",
                                "execution.nav_click",
                                f"[LIVE] Clicking: '{btn_text}'",
                                run_id=run_id,
                                job_id=job_id or None,
                                node="execution_node",
                                details={"button": bound_text(btn_text, 40)},
                            )
                            await next_btn.click()
                            await asyncio.sleep(get_randomized_delay())

                            # Guard form step navigation
                            await guard_page_account_safety(page, stage="execution_form_step")
                    else:
                        labels = await visible_button_labels(modal)
                        label_text = ", ".join(labels) if labels else "none"
                        log_event(
                            "warning",
                            "execution.no_navigation_control",
                            "[LIVE] No recognized forward/submit control found.",
                            run_id=run_id,
                            job_id=job_id or None,
                            node="execution_node",
                            details={"visible_buttons": bound_text(label_text, 120)},
                        )
                        await _safe_close_page(page)
                        return await _manual_review_update(
                            state,
                            "no_next_or_submit_button",
                            "No recognized forward/submit control on Easy Apply form. "
                            f"Visible buttons: {label_text}",
                            form_qa_exchanges,
                        )

                # Clean close outside form steps loop
                log_event(
                    "debug",
                    "execution.page_close",
                    "[LIVE] Closing job details page.",
                    run_id=run_id,
                    job_id=job_id or None,
                    node="execution_node",
                )
                close_err = await _safe_close_page(page)
                if close_err:
                    secondary_errors.append(close_err)

                # Update trace with execution results
                if run_tree:
                    run_tree.metadata.update(
                        get_execution_metadata(
                            dry_run=state["dry_run"],
                            status="submitted" if application_submitted else "completed",
                            qa_count=len(form_qa_exchanges),
                            form_steps=0,
                            timed_out=False,
                        )
                    )

        if application_submitted:
            log_event(
                "info",
                "execution.completed_submitted",
                f"✅ [LIVE] Successfully submitted application for {current_job['title']} at {current_job['company']}!",
                run_id=run_id,
                job_id=job_id or None,
                attempt_id=attempt_id,
                node="execution_node",
                outcome="submitted",
            )
            update = _applied_update(
                state,
                "submitted",
                form_qa_exchanges,
                increment_counters=should_increment_counters,
            )
            if secondary_errors:
                update["errors"] = list(state.get("errors") or []) + secondary_errors
            receipt_error = await send_application_receipt(
                telegram,
                update["application_outcomes"][-1],
                run_id=run_id,
            )

            if receipt_error:
                update["errors"] = list(update.get("errors") or state.get("errors") or []) + [
                    receipt_error
                ]
            return update
        else:
            log_event(
                "error",
                "execution.incomplete",
                "❌ [LIVE] Application incomplete or failed.",
                run_id=run_id,
                job_id=job_id or None,
                node="execution_node",
                outcome="failed",
            )
            return _failed_update(state, "Could not complete application flow", form_qa_exchanges)

    except AccountSafetyBarrierError as safety_err:
        if already_submitted_terminal:
            update = _skipped_update(
                state,
                "already_submitted",
                f"Job {job_id} was already marked submitted",
                form_qa_exchanges,
                extra={"job_id": job_id, "attempt_status": "submitted"},
            )
            update["errors"] = (
                list(state.get("errors") or [])
                + secondary_errors
                + [f"Post-already-submitted barrier ({safety_err.detection.reason})"]
            )
            return update

        if primary_submission_persisted:
            update = _applied_update(
                state,
                "submitted",
                form_qa_exchanges,
                increment_counters=should_increment_counters,
            )
            update["errors"] = (
                list(state.get("errors") or [])
                + secondary_errors
                + [f"Post-submit account safety barrier detected: {safety_err.detection.reason}"]
            )
            return update
        await _safe_close_page(page)
        return await _account_safety_execution_update(
            state,
            safety_err.detection,
            current_job,
            form_qa_exchanges,
            pre_submit=not submission_attempted,
        )
    except UserSkippedJob as exc:
        log_event(
            "info",
            "execution.user_skipped",
            "[LIVE] User skipped this job from Telegram while answering.",
            run_id=run_id,
            job_id=job_id or None,
            node="execution_node",
            details={"question_fingerprint": fingerprint(exc.question)},
        )
        form_qa_exchanges.append(
            {
                "question": exc.question,
                "answer": "/skip",
                "timed_out": False,
                "skipped": True,
            }
        )
        await _safe_close_page(page)
        try:
            await telegram.send_message(
                f"Job skipped: {current_job['title']} at {current_job['company']}."
            )
        except Exception as telegram_exc:
            log_event(
                "warning",
                "execution.skip_ack_failed",
                "[LIVE] Skip acknowledgement could not be sent.",
                run_id=run_id,
                job_id=job_id or None,
                node="execution_node",
                exc=telegram_exc,
            )
        return _skipped_update(
            state,
            "user_skipped",
            f"User skipped the job while answering: {exc.question}",
            form_qa_exchanges,
            {"skip_question": exc.question},
        )
    except FormQaInfrastructureError as qa_err:
        await _safe_close_page(page)
        return await _manual_review_update(
            state,
            "form_qa_infrastructure_failed",
            f"Telegram Q&A storage/delivery error ({type(qa_err).__name__})",
            form_qa_exchanges,
        )
    except Exception as e:
        if already_submitted_terminal:
            await _safe_close_page(page)
            update = _skipped_update(
                state,
                "already_submitted",
                f"Job {job_id} was already marked submitted",
                form_qa_exchanges,
                extra={"job_id": job_id, "attempt_status": "submitted"},
            )
            update["errors"] = (
                list(state.get("errors") or [])
                + secondary_errors
                + [f"Post-already-submitted error ({type(e).__name__})"]
            )
            return update

        if primary_submission_persisted:
            await _safe_close_page(page)
            update = _applied_update(
                state,
                "submitted",
                form_qa_exchanges,
                increment_counters=should_increment_counters,
            )
            update["errors"] = (
                list(state.get("errors") or [])
                + secondary_errors
                + [f"Post-submit error ({type(e).__name__})"]
            )
            return update

        if attempt_id and not submission_attempted:
            cleaned, c_errs = await _cleanup_pre_click(
                reason=f"unhandled_pre_click_exception:{type(e).__name__}",
                quota_was_reserved=quota_reserved,
            )
            close_err = await _safe_close_page(page)
            if close_err:
                c_errs.append(close_err)
            msg = f"Execution pre-click error ({type(e).__name__})"
            if not cleaned:
                msg = f"{msg} (Cleanup required: {'; '.join(c_errs)})"
            return await _manual_review_update(
                state,
                "unhandled_pre_click_exception",
                msg,
                form_qa_exchanges,
                extra={"cleanup_errors": c_errs, "job_id": job_id, "attempt_id": attempt_id},
            )

        if submission_attempted:
            try:
                if page is not None and hasattr(page, "is_closed") and not page.is_closed():
                    safety_check = await inspect_page_account_safety(
                        page, stage="execution_post_submit_error", fail_closed=True
                    )
                    await _safe_close_page(page)
                    if safety_check.detected:
                        return await _account_safety_execution_update(
                            state,
                            safety_check,
                            current_job,
                            form_qa_exchanges,
                            pre_submit=False,
                        )
            except Exception:
                pass
            ambiguous_detection = AccountSafetyDetection(
                detected=True,
                barrier_type=AccountSafetyBarrierType.INSPECTION_UNAVAILABLE,
                reason=f"Ambiguous submission attempt error: {type(e).__name__}",
                stage="execution_post_submit_exception",
                url=sanitize_url_for_evidence(getattr(page, "url", "") if page else ""),
                detected_at=datetime.now(timezone.utc).isoformat(),
                resume_instructions=SAFE_RESUME_INSTRUCTIONS,
            )
            return await _account_safety_execution_update(
                state,
                ambiguous_detection,
                current_job,
                form_qa_exchanges,
                pre_submit=False,
            )

        error_msg = f"Execution error for {current_job['title']}: {str(e)}"
        log_event(
            "error",
            "execution.error",
            f"❌ [LIVE] Execution Error: {type(e).__name__}",
            run_id=run_id,
            job_id=job_id or None,
            node="execution_node",
            outcome="failed",
            exc=e,
        )
        try:
            if page is not None and hasattr(page, "is_closed") and not page.is_closed():
                safety_check = await inspect_page_account_safety(page, stage="execution_error")
                if safety_check.detected:
                    await _safe_close_page(page)
                    return await _account_safety_execution_update(
                        state, safety_check, current_job, form_qa_exchanges, pre_submit=True
                    )
                await take_error_screenshot(
                    page, state.get("run_id", ""), f"exec_error_{current_job.get('job_id', '')}"
                )
        except Exception as screenshot_err:
            log_event(
                "warning",
                "execution.error_screenshot_failed",
                "Failed to take error screenshot.",
                run_id=run_id,
                job_id=job_id or None,
                node="execution_node",
                exc=screenshot_err,
            )
        await _safe_close_page(page)
        return _failed_update(state, error_msg, form_qa_exchanges)
