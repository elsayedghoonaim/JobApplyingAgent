"""Navigation and submission evidence helpers for Easy Apply forms."""

import asyncio
import time
from typing import Any, Optional

from jobapply.utils.account_safety import (
    AccountSafetyDetection,
    inspect_page_account_safety,
)

MODAL_SELECTORS: list[str] = [
    "dialog",
    "[role='dialog']",
    ".jobs-easy-apply-modal",
    ".jobs-easy-apply-content",
    ".jobs-easy-apply-form",
    "form.jobs-easy-apply-form",
    "[data-test-easy-apply-modal]",
    "[data-test-easy-apply-content]",
    "[data-test-easy-apply-form]",
    "div[data-test-modal]",
    ".artdeco-modal",
    "div.artdeco-modal__content",
]
MODAL_CSS: str = ", ".join(MODAL_SELECTORS)


async def resolve_easy_apply_container(page: Any, timeout_ms: int) -> Any | None:
    """Resolve modal, inline, or framed Easy Apply form containers.

    LinkedIn can render Easy Apply as an Artdeco modal, a native dialog, an
    inline SDUI form, or (occasionally) inside a child frame.  Waiting only for
    the historical modal class incorrectly treats valid inline forms as failed.
    """
    try:
        await page.wait_for_selector(MODAL_CSS, timeout=timeout_ms)
    except Exception:
        pass

    surfaces: list[Any] = [page]
    try:
        for frame in page.frames:
            if frame not in surfaces:
                surfaces.append(frame)
    except (AttributeError, TypeError):
        pass

    for surface in surfaces:
        for selector in MODAL_SELECTORS:
            try:
                candidates = await surface.query_selector_all(selector)
            except Exception:
                continue
            for candidate in candidates:
                try:
                    if await candidate.is_visible():
                        return candidate
                except Exception:
                    continue

        # Semantic fallback for new LinkedIn markup: accept a visible form only
        # when it has both input controls and a recognized forward/submit action.
        try:
            forms = await surface.query_selector_all("form")
        except Exception:
            forms = []
        for form in forms:
            try:
                if not await form.is_visible():
                    continue
                controls = await form.query_selector_all(
                    "input, select, textarea, [contenteditable='true']"
                )
                navigation, action, _ = await find_navigation_button(form)
                if controls and navigation is not None and action is not None:
                    return form
            except Exception:
                continue
    return None


def classify_navigation_action(
    text: str | None,
    aria_label: str | None,
) -> str | None:
    """Classify a visible Easy Apply control as advance, submit, or unrelated."""
    label = " ".join(f"{text or ''} {aria_label or ''}".casefold().split())
    if not label:
        return None
    if any(word in label for word in ("back", "close", "cancel", "dismiss", "discard")):
        return None
    if any(phrase in label for phrase in ("submit application", "send application")):
        return "submit"
    if "submit" in label:
        return "submit"
    if any(word in label for word in ("next", "continue", "review", "proceed")):
        return "advance"
    return None


async def find_navigation_button(modal: Any) -> tuple[Any, str | None, str | None]:
    """Return the first visible enabled forward/submit control and its label."""
    candidates = await modal.query_selector_all("button, [role='button']")
    for candidate in candidates:
        if not await candidate.is_visible() or not await candidate.is_enabled():
            continue
        text = (await candidate.inner_text()).strip()
        aria_label = await candidate.get_attribute("aria-label")
        action = classify_navigation_action(text, aria_label)
        if action:
            return candidate, action, text or aria_label or action
    return None, None, None


async def visible_button_labels(modal: Any) -> list[str]:
    """Return visible button labels for actionable diagnostics."""
    labels: list[str] = []
    for candidate in await modal.query_selector_all("button, [role='button']"):
        if not await candidate.is_visible():
            continue
        text = (await candidate.inner_text()).strip()
        aria_label = (await candidate.get_attribute("aria-label") or "").strip()
        label = text or aria_label
        if label:
            labels.append(label)
    return labels


def text_indicates_already_applied(text: str | None) -> bool:
    """Return whether text is an explicit LinkedIn application-status marker."""
    normalized = " ".join((text or "").casefold().split())
    return normalized == "applied" or any(
        phrase in normalized
        for phrase in (
            "already applied",
            "application submitted",
            "application sent",
            "you applied",
        )
    )


async def find_already_applied_indicator(page: Any) -> str | None:
    """Find a visible applied marker in LinkedIn's primary job action area."""
    selectors = (
        ".jobs-s-apply",
        ".jobs-details-top-card__actions-container",
        "button.jobs-apply-button",
        "button[aria-label*='applied' i]",
    )
    try:
        for selector in selectors:
            for candidate in await page.query_selector_all(selector):
                if not await candidate.is_visible():
                    continue
                text = (await candidate.inner_text()).strip()
                aria_label = (await candidate.get_attribute("aria-label") or "").strip()
                evidence = text or aria_label
                if text_indicates_already_applied(text) or text_indicates_already_applied(
                    aria_label
                ):
                    return evidence
    except Exception:
        return None
    return None


async def wait_for_submission_or_safety(
    page: Any,
    timeout_ms: int = 12000,
    *,
    inspect_safety_fn: Any = inspect_page_account_safety,
) -> tuple[bool, Optional[AccountSafetyDetection]]:
    """Poll for both explicit LinkedIn submission confirmation and account safety barriers.

    Returns:
        (True, None) if explicit submission success is confirmed.
        (False, detection) if an account safety barrier is detected during or after submission.
        (False, None) if timeout occurs without confirmation or barrier (unconfirmed).
    """
    success_phrases = (
        "application submitted",
        "application sent",
        "your application was sent",
    )
    start_time = time.monotonic()
    deadline = start_time + (timeout_ms / 1000.0)

    while time.monotonic() < deadline:
        # 1. Check account safety barrier first
        safety = await inspect_safety_fn(page, stage="execution_post_submit", fail_closed=True)
        if safety.detected:
            return False, safety

        # 2. Check for explicit success confirmation
        try:
            if hasattr(page, "wait_for_function"):
                await page.wait_for_function(
                    """phrases => {
                        const text = (document.body?.innerText || '').toLowerCase();
                        return phrases.some(phrase => text.includes(phrase));
                    }""",
                    list(success_phrases),
                    timeout=500,
                )
                return True, None
        except Exception:
            pass

        try:
            confirmed = await page.evaluate(
                """(phrases) => {
                    const text = (document.body?.innerText || '').toLowerCase();
                    return phrases.some(phrase => text.includes(phrase));
                }""",
                list(success_phrases),
            )
            if confirmed:
                return True, None
        except Exception:
            try:
                body_text = (await page.locator("body").inner_text()).lower()
                if any(phrase in body_text for phrase in success_phrases):
                    return True, None
            except Exception:
                pass

        await asyncio.sleep(0.5)

    # Final safety inspection at deadline before declaring unconfirmed
    final_safety = await inspect_safety_fn(page, stage="execution_post_submit", fail_closed=True)
    if final_safety.detected:
        return False, final_safety

    return False, None


async def wait_for_submission_confirmation(
    page: Any,
    timeout_ms: int = 12000,
    *,
    inspect_safety_fn: Any = inspect_page_account_safety,
) -> bool:
    """Return True only when LinkedIn exposes an explicit success state."""
    confirmed, _ = await wait_for_submission_or_safety(
        page, timeout_ms=timeout_ms, inspect_safety_fn=inspect_safety_fn
    )
    return confirmed
