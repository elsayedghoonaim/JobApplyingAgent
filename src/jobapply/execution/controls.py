"""Playwright-compatible DOM and form control helpers."""

import asyncio
from typing import Any

from jobapply.execution.planning import (
    STANDARD_TEXT_FIELD_SELECTOR,
    RequiredFieldValidationResult,
    _normalized_choice_text,
    choice_is_unanswered,
    is_required_field,
    match_choice_index,
)
from jobapply.utils.account_safety import sanitize_evidence_string


async def get_choice_label(control: Any, page: Any, fallback: str = "Choice question") -> str:
    """Resolve a stable accessible question label for a choice control."""
    label = await control.get_attribute("aria-label")
    if label:
        return label.strip()
    try:
        label = await control.evaluate(
            r"""el => {
                const labelled = (el.getAttribute('aria-labelledby') || '')
                    .split(/\s+/).filter(Boolean)
                    .map(id => document.getElementById(id)?.innerText || '')
                    .join(' ').trim();
                if (labelled) return labelled;
                if (el.id) {
                    const explicit = [...document.querySelectorAll('label')]
                        .find(item => item.htmlFor === el.id);
                    if (explicit?.innerText) return explicit.innerText.trim();
                }
                const previous = el.previousElementSibling;
                if (previous?.innerText?.trim()) return previous.innerText.trim();
                const group = el.closest('fieldset, [role="radiogroup"]');
                const heading = group?.querySelector('legend, label, h1, h2, h3, h4, [data-test-form-element-label]');
                return (heading?.innerText || '').trim();
            }"""
        )
        if label:
            return label.strip()
    except Exception:
        pass
    return fallback


async def get_form_field_label(control: Any, page: Any, fallback: str = "Unknown field") -> str:
    """Resolve a standard field label from accessible and native markup."""
    for attribute in ("aria-label", "placeholder"):
        label = (await control.get_attribute(attribute) or "").strip()
        if label:
            return label
    field_id = await control.get_attribute("id")
    if field_id:
        try:
            label_elem = await page.query_selector(f"label[for='{field_id}']")
            if label_elem:
                label = (await label_elem.inner_text()).strip()
                if label:
                    return label
        except Exception:
            pass
    return fallback


async def get_radio_option_label(radio: Any, fieldset: Any) -> str:
    """Resolve option text from native or LinkedIn role-based radio markup."""
    aria_label = (await radio.get_attribute("aria-label") or "").strip()
    if aria_label:
        return aria_label

    radio_id = await radio.get_attribute("id")
    if radio_id:
        try:
            label_elem = await fieldset.query_selector(f"label[for='{radio_id}']")
            if label_elem:
                label = (await label_elem.inner_text()).strip()
                if label:
                    return label
        except Exception:
            pass

    try:
        label = await radio.evaluate(
            r"""el => {
                const labelled = (el.getAttribute('aria-labelledby') || '')
                    .split(/\s+/).filter(Boolean)
                    .map(id => document.getElementById(id)?.innerText || '')
                    .join(' ').trim();
                if (labelled) return labelled;
                const roleOption = el.closest('[role="radio"]');
                if (roleOption?.innerText?.trim()) return roleOption.innerText.trim();
                const wrappingLabel = el.closest('label');
                if (wrappingLabel?.innerText?.trim()) return wrappingLabel.innerText.trim();
                return (el.parentElement?.innerText || '').trim();
            }"""
        )
        if label:
            return label.strip()
    except Exception:
        pass

    return (await radio.get_attribute("value") or "").strip()


async def _fieldset_question_text(fieldset: Any, page: Any) -> str:
    """Extract legend or accessible label for a radio fieldset."""
    legend = await fieldset.query_selector("legend")
    if legend:
        text = (await legend.inner_text()).strip()
        if text:
            return text
    return await get_choice_label(fieldset, page)


async def select_radio_option(radio: Any, fieldset: Any) -> str:
    """Select a radio through its visible LinkedIn control, with a native fallback."""
    role_handle = None
    try:
        role_handle = await radio.evaluate_handle("el => el.closest('[role=radio]')")
        role_option = role_handle.as_element()
        if role_option and await role_option.is_visible():
            await role_option.click(timeout=5_000)
            await asyncio.sleep(0.1)
            if await radio.is_checked():
                return "visible role=radio control"
    except Exception:
        pass
    finally:
        if role_handle is not None:
            try:
                await role_handle.dispose()
            except Exception:
                pass

    radio_id = await radio.get_attribute("id")
    if radio_id:
        try:
            label = await fieldset.query_selector(f"label[for='{radio_id}']")
            if label and await label.is_visible():
                await label.click(timeout=5_000)
                await asyncio.sleep(0.1)
                if await radio.is_checked():
                    return "visible label"
        except Exception:
            pass

    await radio.check(force=True, timeout=5_000)
    if not await radio.is_checked():
        raise RuntimeError("LinkedIn radio did not become checked")
    return "forced native radio fallback"


async def select_live_radio_option(
    page: Any,
    question_text: str,
    option_text: str,
    attempts: int = 3,
    *,
    select_radio_option_fn: Any = select_radio_option,
) -> str:
    """Re-find and select a native radio, retrying across LinkedIn re-renders."""
    last_error: Exception | None = None
    normalized_question = _normalized_choice_text(question_text)
    for attempt in range(attempts):
        try:
            for fieldset in await page.query_selector_all("fieldset"):
                if not await fieldset.is_visible():
                    continue
                current_question = await _fieldset_question_text(fieldset, page)
                if _normalized_choice_text(current_question) != normalized_question:
                    continue
                radios = await fieldset.query_selector_all("input[type='radio']")
                labels = [await get_radio_option_label(radio, fieldset) for radio in radios]
                matched = match_choice_index(option_text, labels)
                if matched is None:
                    raise RuntimeError(
                        f"Option '{option_text}' is no longer present for: {question_text}"
                    )
                radio = radios[matched]
                if await radio.is_checked():
                    return "live radio already selected after re-render"
                return await select_radio_option_fn(radio, fieldset)
            raise RuntimeError(f"Radio question is no longer present: {question_text}")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(0.25)
    raise RuntimeError(
        f"Could not select refreshed radio option '{option_text}' for "
        f"'{question_text}': {last_error}"
    ) from last_error


async def select_live_role_radio_option(
    page: Any,
    question_text: str,
    option_text: str,
    attempts: int = 3,
) -> str:
    """Re-find and click an ARIA radio option across LinkedIn re-renders."""
    last_error: Exception | None = None
    normalized_question = _normalized_choice_text(question_text)
    for attempt in range(attempts):
        try:
            for group in await page.query_selector_all("[role='radiogroup']"):
                if not await group.is_visible():
                    continue
                if await group.query_selector("input[type='radio']"):
                    continue
                current_question = await get_choice_label(group, page)
                if _normalized_choice_text(current_question) != normalized_question:
                    continue
                role_options = await group.query_selector_all("[role='radio']")
                labels = []
                for option in role_options:
                    label = await option.get_attribute("aria-label")
                    labels.append(label or (await option.inner_text()).strip())
                matched = match_choice_index(option_text, labels)
                if matched is None:
                    raise RuntimeError(
                        f"Option '{option_text}' is no longer present for: {question_text}"
                    )
                option = role_options[matched]
                if (await option.get_attribute("aria-checked") or "").lower() == "true":
                    return "live role=radio already selected after re-render"
                await option.click(timeout=5_000)
                await asyncio.sleep(0.1)
                if (await option.get_attribute("aria-checked") or "").lower() == "true":
                    return "refreshed role=radio control"
                raise RuntimeError("LinkedIn role=radio did not become checked")
            raise RuntimeError(f"ARIA radio question is no longer present: {question_text}")
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(0.25)
    raise RuntimeError(
        f"Could not select refreshed ARIA radio option '{option_text}' for "
        f"'{question_text}': {last_error}"
    ) from last_error


async def validate_visible_required_controls(
    modal: Any,
    page: Any,
) -> RequiredFieldValidationResult:
    """Inspect visible form controls in the active modal and verify all required ones are answered.

    Fails closed on any inspection/DOM exceptions with bounded, redacted diagnostics.
    """
    unresolved_raw: list[str] = []
    inspection_failures: list[str] = []

    # 1. Text / Number / Area / Contenteditable inputs
    try:
        text_fields = await modal.query_selector_all(STANDARD_TEXT_FIELD_SELECTOR)
        for field in text_fields:
            if not await field.is_visible():
                continue
            label = await get_form_field_label(field, page)
            req = is_required_field(
                await field.get_attribute("required"),
                await field.get_attribute("aria-required"),
                label,
            )
            if not req:
                continue
            try:
                val = await field.input_value()
            except Exception:
                try:
                    val = (await field.inner_text()).strip()
                except Exception:
                    val = ""
            if not val or not str(val).strip():
                unresolved_raw.append(label)
    except Exception as exc:
        inspection_failures.append(
            f"Required-field inspection unavailable: text controls ({type(exc).__name__})"
        )

    # 2. Select dropdowns
    try:
        selects = await modal.query_selector_all("select")
        for sel in selects:
            if not await sel.is_visible():
                continue
            label = await get_form_field_label(sel, page)
            req = is_required_field(
                await sel.get_attribute("required"),
                await sel.get_attribute("aria-required"),
                label,
            )
            if not req:
                continue
            try:
                val = await sel.input_value()
            except Exception:
                val = ""
            selected_text = ""
            try:
                selected_text = await sel.evaluate(
                    "el => el.options[el.selectedIndex]?.textContent?.trim() || ''"
                )
            except Exception:
                pass
            if choice_is_unanswered(val, selected_text):
                unresolved_raw.append(label)
    except Exception as exc:
        inspection_failures.append(
            f"Required-field inspection unavailable: select controls ({type(exc).__name__})"
        )

    # 3. Radio fieldsets
    try:
        fieldsets = await modal.query_selector_all("fieldset")
        for fs in fieldsets:
            if not await fs.is_visible():
                continue
            radios = await fs.query_selector_all("input[type='radio']")
            if not radios:
                continue
            q_text = await _fieldset_question_text(fs, page)
            fs_req = (
                await fs.get_attribute("required") is not None
                or (await fs.get_attribute("aria-required") or "").lower() == "true"
                or is_required_field(None, None, q_text)
            )
            if not fs_req:
                continue
            checked = await fs.query_selector("input[type='radio']:checked")
            if not checked:
                unresolved_raw.append(q_text)
    except Exception as exc:
        inspection_failures.append(
            f"Required-field inspection unavailable: fieldset radio controls ({type(exc).__name__})"
        )

    # 4. ARIA radiogroups
    try:
        role_groups = await modal.query_selector_all("[role='radiogroup']")
        for rg in role_groups:
            if not await rg.is_visible():
                continue
            if await rg.query_selector("input[type='radio']"):
                continue
            q_text = await get_choice_label(rg, page)
            rg_req = (
                await rg.get_attribute("aria-required") or ""
            ).lower() == "true" or is_required_field(None, None, q_text)
            if not rg_req:
                continue
            role_options = await rg.query_selector_all("[role='radio']")
            any_checked = False
            for opt in role_options:
                if (await opt.get_attribute("aria-checked") or "").lower() == "true":
                    any_checked = True
                    break
            if not any_checked:
                unresolved_raw.append(q_text)
    except Exception as exc:
        inspection_failures.append(
            f"Required-field inspection unavailable: ARIA radiogroup controls ({type(exc).__name__})"
        )

    # 5. Checkboxes
    try:
        checkboxes = await modal.query_selector_all("input[type='checkbox']")
        for cb in checkboxes:
            if not await cb.is_visible():
                continue
            cb_id = await cb.get_attribute("id")
            cb_label = ""
            if cb_id:
                try:
                    label_elem = await page.query_selector(f"label[for='{cb_id}']")
                    if label_elem:
                        cb_label = (await label_elem.inner_text()).strip()
                except Exception:
                    pass
            if not cb_label:
                try:
                    cb_label = (await cb.evaluate("el => el.parentElement.innerText")).strip()
                except Exception:
                    pass
            cb_label = cb_label or "Required checkbox"
            req = is_required_field(
                await cb.get_attribute("required"),
                await cb.get_attribute("aria-required"),
                cb_label,
            )
            if not req:
                continue
            if not await cb.is_checked():
                unresolved_raw.append(cb_label)
    except Exception as exc:
        inspection_failures.append(
            f"Required-field inspection unavailable: checkbox controls ({type(exc).__name__})"
        )

    # 6. Custom comboboxes
    try:
        custom_combos = await modal.query_selector_all(
            "[role='combobox']:not(select), [aria-haspopup='listbox']:not(select)"
        )
        for combo in custom_combos:
            if not await combo.is_visible():
                continue
            placeholder = await combo.get_attribute("placeholder")
            q_text = await get_choice_label(combo, page, fallback=placeholder or "Choice question")
            req = (
                await combo.get_attribute("aria-required") or ""
            ).lower() == "true" or is_required_field(None, None, q_text)
            if not req:
                continue
            tag_name = await combo.evaluate("el => el.tagName.toLowerCase()")
            current_value = (
                await combo.input_value()
                if tag_name in ("input", "textarea")
                else (await combo.inner_text()).strip()
            )
            if choice_is_unanswered(current_value, placeholder):
                unresolved_raw.append(q_text)
    except Exception as exc:
        inspection_failures.append(
            f"Required-field inspection unavailable: custom combobox controls ({type(exc).__name__})"
        )

    # Collect and sanitize all unresolved fields and inspection failure diagnostic items
    sanitized_items: list[str] = []
    for failure in inspection_failures:
        sanitized_items.append(sanitize_evidence_string(failure, max_length=140))

    for raw_label in unresolved_raw:
        sanitized = sanitize_evidence_string(raw_label, max_length=140)
        if sanitized:
            sanitized_items.append(sanitized)

    if sanitized_items:
        unique_items = list(dict.fromkeys(sanitized_items))
        max_reported_fields = 10
        if len(unique_items) > max_reported_fields:
            kept = max_reported_fields - 1
            omitted = len(unique_items) - kept
            bounded_items = unique_items[:kept] + [f"... ({omitted} more fields omitted)"]
        else:
            bounded_items = unique_items

        fields_summary = ", ".join(bounded_items)
        reason = sanitize_evidence_string(
            f"Unresolved visible required field(s): {fields_summary}",
            max_length=300,
        )
        return RequiredFieldValidationResult(
            is_valid=False,
            unresolved_fields=bounded_items,
            reason=reason,
        )

    return RequiredFieldValidationResult(is_valid=True, unresolved_fields=[])
