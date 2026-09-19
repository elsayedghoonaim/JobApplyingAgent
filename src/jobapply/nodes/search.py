"""Search node - LinkedIn job search via Playwright."""

import asyncio
import re

from langsmith.run_helpers import trace

from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.account_safety import (
    AccountSafetyBarrierError,
    guard_page_account_safety,
    normalize_safety_log_payload,
    sanitize_evidence_string,
)
from jobapply.utils.browser import get_linkedin_page, get_randomized_delay, managed_browser
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.job_filters import (
    get_location_exclusion_reason,
    get_title_exclusion_reason,
)
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.llm import get_llm
from jobapply.utils.observability import bound_text, log_event
from jobapply.utils.prompts import get_job_parser_prompt, truncate_head_tail
from jobapply.utils.tracing import get_search_metadata


def _account_safety_search_update(state: JobApplyState, detection) -> dict:
    """Produce explicit, non-infrastructure-failure pause state update."""
    norm = normalize_safety_log_payload(
        run_id=str(state.get("run_id") or "unknown"),
        barrier_type=detection.barrier_type.value if detection.barrier_type else None,
        stage=detection.stage,
        reason=detection.reason,
        url=detection.url,
        resume_instructions=detection.resume_instructions,
    )
    btype_val = detection.barrier_type.value if detection.barrier_type else "unknown"
    safe_evidence = (
        sanitize_evidence_string(detection.evidence, max_length=120) if detection.evidence else None
    )
    return {
        "job_listings": [],
        "current_job_index": 0,
        "current_job": None,
        "search_failed": False,
        "seen_job_ids": set(state.get("seen_job_ids") or set()),
        "account_safety_paused": True,
        "account_safety_barrier_type": btype_val,
        "account_safety_reason": norm["reason"],
        "account_safety_stage": norm["stage"],
        "account_safety_url": norm["url"],
        "account_safety_detected_at": detection.detected_at,
        "account_safety_evidence": safe_evidence,
        "account_safety_resume_instructions": norm["resume_instructions"],
        "logs": list(state.get("logs") or [])
        + [f"🛑 Account-safety pause ({btype_val}): {norm['reason']}"],
    }


def build_search_url(
    base_url: str,
    query: str,
    page_num: int,
    *,
    location: str | None = None,
    recency_days: int | None = None,
) -> str:
    """Build a correctly encoded LinkedIn Easy Apply search URL.

    Location defaults to Worldwide; recency_days (bounded 1-30) maps
    deterministically onto LinkedIn's f_TPR parameter and None disables it.
    """
    from jobapply.utils.search_config import build_linkedin_search_url

    return build_linkedin_search_url(
        base_url,
        query,
        page_num,
        location=location,
        recency_days=recency_days,
    )


def text_indicates_applied_card_status(text: str | None) -> bool:
    """Return whether short search-card text is an explicit applied marker."""
    normalized = " ".join((text or "").casefold().split())
    if normalized in {"applied", "already applied"}:
        return True
    if normalized.startswith(("you applied", "application submitted", "application sent")):
        return True
    return (
        re.fullmatch(
            r"applied\s+(?:\d+\s+)?(?:minute|hour|day|week|month|year)s?\s+ago",
            normalized,
        )
        is not None
    )


async def find_applied_status_on_card(card) -> str | None:
    """Read an explicit applied marker before extracting a LinkedIn job card."""
    try:
        candidates = await card.evaluate(
            r"""card => [...card.querySelectorAll('[aria-label], span, p, div')]
                .filter(node => node.getClientRects().length > 0)
                .filter(node => !node.closest('a.job-card-container__link'))
                .flatMap(node => [
                    (node.getAttribute('aria-label') || '').trim(),
                    (node.innerText || '').trim(),
                ])
                .filter(Boolean)"""
        )
    except Exception:
        return None

    for candidate in candidates or []:
        if text_indicates_applied_card_status(candidate):
            return candidate
    return None


REPOST_STATUS_PATTERN = re.compile(
    r"^reposted(?:(?: by .{0,40})|(?: \d{1,3} (?:second|minute|hour|day|week|month|year)s? ago))?$",
    re.IGNORECASE,
)
MAX_REPOST_EVIDENCE_LENGTH = 80


def text_indicates_repost(text: str | None) -> bool:
    """Recognize explicit LinkedIn repost status phrases only.

    Accepts status-like metadata such as "Reposted", "Reposted 2 weeks ago",
    or "Reposted by <name>". Arbitrary title/company text merely containing
    the word "repost" (e.g. "Repost Coordinator") is never evidence.
    """
    if not text:
        return False
    normalized = " ".join(str(text).split())
    return REPOST_STATUS_PATTERN.match(normalized) is not None


async def find_repost_indicator_on_card(card) -> str | None:
    """Detect visible repost indicators from already-loaded card metadata.

    Reads only bounded footer/metadata/status elements of the current card DOM
    while it is attached — no extra navigation, never title/company/link
    content. Candidate count and text length are capped inside the browser.
    Returns bounded, redacted evidence text or None.
    """
    from jobapply.utils.redaction import redact_string

    try:
        candidates = await card.evaluate(
            r"""card => {
                const nodes = [...card.querySelectorAll(
                    '.job-card-container__footer-item, '
                  + '.job-card-container__footer, '
                  + '.job-card-container__metadata-wrapper li, '
                  + '.job-card-container__metadata-wrapper span'
                )];
                const excluded = node => node.closest(
                    'a.job-card-container__link, '
                  + '.artdeco-entity-lockup__title, '
                  + '.artdeco-entity-lockup__subtitle'
                );
                return nodes
                    .filter(node => !excluded(node))
                    .filter(node => node.getClientRects().length > 0)
                    .flatMap(node => [
                        (node.getAttribute('aria-label') || '').trim(),
                        (node.innerText || '').trim(),
                    ])
                    .filter(Boolean)
                    .slice(0, 20)
                    .map(text => String(text).slice(0, 120));
            }"""
        )
    except Exception:
        return None

    bounded_candidates = list(candidates or [])[:40]
    for candidate in bounded_candidates:
        candidate_text = str(candidate)[:160]
        if text_indicates_repost(candidate_text):
            evidence = " ".join(candidate_text.split())
            return redact_string(evidence)[:MAX_REPOST_EVIDENCE_LENGTH]
    return None


async def parse_job_description(raw_description: str) -> dict:
    """Parse job description to extract structured fields.

    Args:
        raw_description: Raw job description text

    Returns:
        Dict with parsed fields
    """
    try:
        llm = get_llm(
            temperature=0.1,
            max_output_tokens=1536,
            response_mime_type="application/json",
            response_json_schema={
                "type": "object",
                "properties": {
                    "parsed_location": {"type": ["string", "null"]},
                    "duration": {"type": ["string", "null"]},
                    "work_type": {"type": ["string", "null"]},
                    "responsibilities": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "requirements": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "required_languages": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "clean_description": {"type": "string"},
                },
                "required": [
                    "parsed_location",
                    "duration",
                    "work_type",
                    "responsibilities",
                    "requirements",
                    "required_languages",
                    "clean_description",
                ],
            },
        )
        prompt = get_job_parser_prompt(raw_description)

        raw_response = await llm.ainvoke(prompt)
        response_text = (
            raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        )

        parsed = extract_json_object(response_text)

        return parsed
    except Exception as e:
        log_event(
            "debug",
            "search.description_parse_failed",
            "Job description parsing failed; continuing with raw text",
            exc=e,
        )
        # Return empty structure if parsing fails
        return {
            "parsed_location": None,
            "duration": None,
            "work_type": None,
            "responsibilities": [],
            "requirements": [],
            "required_languages": [],
            "clean_description": raw_description,
        }


async def get_loaded_card_ids(page) -> list[str]:
    """Retrieve all rendered job card IDs currently in the DOM without swallowing exceptions."""
    cards = await page.query_selector_all("li[data-occludable-job-id]")
    ids: list[str] = []
    for c in cards:
        jid = await c.get_attribute("data-occludable-job-id")
        if jid:
            ids.append(jid)
    return ids


async def load_search_cards_adaptively(
    page,
    max_scroll_rounds: int = 5,
    stability_rounds: int = 2,
    scroll_delay_seconds: float = 0.3,
) -> None:
    """Scroll adaptively until the rendered job card set stabilizes or maximum rounds are reached."""
    seen_ids = await get_loaded_card_ids(page)
    stable_count = 0

    for _ in range(max_scroll_rounds):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(scroll_delay_seconds)
        current_ids = await get_loaded_card_ids(page)

        if len(current_ids) > len(seen_ids) or set(current_ids) != set(seen_ids):
            seen_ids = current_ids
            stable_count = 0
        else:
            stable_count += 1
            if stable_count >= stability_rounds:
                break


async def search_node(state: JobApplyState) -> dict:
    """Fetch job listings from LinkedIn for current query + page without making LLM calls.

    Does NOT iterate or pick jobs - just fetches and returns listings.

    Args:
        state: Current graph state.

    Returns:
        State updates dict with job_listings, updated seen_job_ids, and reset current_job_index.
    """
    settings = get_settings()

    query_index = state["current_query_index"]
    page_num = state["current_page"]
    query = state["search_queries"][query_index]

    # Effective location/recency live in checkpoint state so resumed runs stay
    # stable; legacy checkpoints fall back to configured settings.
    from jobapply.utils.search_config import resolve_state_search_params

    effective_location, effective_recency = resolve_state_search_params(state, settings)

    search_url = build_search_url(
        settings.linkedin_base_url,
        query,
        page_num,
        location=effective_location,
        recency_days=effective_recency,
    )

    job_listings = []

    log_event(
        "info",
        "search.page_started",
        f"🔍 Searching: '{query}' (page {page_num}/{state['pages_per_query']})",
        run_id=str(state.get("run_id") or "") or None,
        node="search_node",
        details={"query_index": query_index, "page": page_num},
    )

    page = None
    try:
        async with managed_browser() as (browser, context):
            async with trace(
                "linkedin_search",
                run_type="tool",
                metadata={"search_url": search_url, "query": query, "page": page_num},
            ) as run_tree:
                page = await get_linkedin_page(context)

                # Navigate to search results
                response = await page.goto(
                    search_url,
                    wait_until="domcontentloaded",
                    timeout=settings.linkedin_navigation_timeout_ms,
                )
                http_status = response.status if response else None

                # Guard search navigation
                await guard_page_account_safety(
                    page, stage="search_navigation", http_status=http_status
                )

                # Wait for job cards to load
                try:
                    await page.wait_for_selector(
                        ".scaffold-layout__list", timeout=settings.job_load_timeout_ms
                    )
                    await page.wait_for_selector(
                        "li[data-occludable-job-id]", timeout=settings.job_load_timeout_ms
                    )

                    # Adaptively scroll to load lazy-loaded cards
                    await load_search_cards_adaptively(
                        page,
                        max_scroll_rounds=settings.search_max_scroll_rounds,
                        stability_rounds=settings.search_card_stability_rounds,
                        scroll_delay_seconds=settings.search_scroll_delay_seconds,
                    )
                except Exception:
                    # No results found
                    log_event(
                        "info",
                        "search.no_results",
                        f"❌ No results found for '{query}' page {page_num}",
                        run_id=str(state.get("run_id") or "") or None,
                        node="search_node",
                        details={"page": page_num},
                    )
                    if run_tree:
                        run_tree.metadata.update(get_search_metadata(query, page_num, 0))
                    return {
                        "job_listings": [],
                        "current_job_index": 0,
                        "current_job": None,
                        "search_failed": False,
                        "query_exhausted": True,
                        "seen_job_ids": set(state.get("seen_job_ids") or set()),
                        "logs": list(state.get("logs") or [])
                        + [f"No results for query '{query}' page {page_num}"],
                    }

                # Extract all job cards on the page without arbitrary truncation
                job_cards = await page.query_selector_all("li[data-occludable-job-id]")
                log_event(
                    "debug",
                    "search.cards_found",
                    f"Found {len(job_cards)} job card elements",
                    run_id=str(state.get("run_id") or "") or None,
                    node="search_node",
                    details={"card_count": len(job_cards), "page": page_num},
                )

                # Collect all card job IDs on the page
                page_job_ids: list[str] = []
                for card in job_cards:
                    try:
                        jid = await card.get_attribute("data-occludable-job-id")
                        if jid:
                            page_job_ids.append(jid)
                    except Exception:
                        pass

                # Batch-query seen IDs from MongoDB for the cards present on this page
                dedup = DeduplicationStore()
                persisted_seen = await dedup.find_seen_ids(
                    page_job_ids,
                    for_live_run=not state.get("dry_run", True),
                )

                # Combine in-memory and persisted seen IDs immutably
                incoming_seen = set(state.get("seen_job_ids") or set())
                already_seen = incoming_seen | persisted_seen
                updated_seen_ids = set(already_seen)

                # Track job IDs to detect duplicates during extraction on this page
                extracted_job_ids = set()
                exclusions_to_persist: list[dict] = []

                # Inspect every card returned for the page
                for card in job_cards:
                    try:
                        # Extract job ID from data attribute
                        job_id = await card.get_attribute("data-occludable-job-id")

                        if not job_id:
                            log_event(
                                "debug",
                                "search.card_missing_id",
                                "Skipping card with no job_id",
                                run_id=str(state.get("run_id") or "") or None,
                                node="search_node",
                            )
                            continue

                        # Check if already seen in previous runs/sessions or state
                        if job_id in already_seen:
                            log_event(
                                "debug",
                                "search.card_already_seen",
                                f"Skipping already-seen job {job_id}",
                                run_id=str(state.get("run_id") or "") or None,
                                job_id=job_id,
                                node="search_node",
                            )
                            continue

                        # Check for duplicate in current extraction
                        if job_id in extracted_job_ids:
                            log_event(
                                "debug",
                                "search.card_duplicate_in_page",
                                f"Duplicate job_id {job_id} found in same page, skipping",
                                run_id=str(state.get("run_id") or "") or None,
                                job_id=job_id,
                                node="search_node",
                            )
                            continue

                        applied_status = await find_applied_status_on_card(card)
                        if applied_status:
                            log_event(
                                "debug",
                                "search.card_already_applied",
                                f"Skipping already-applied LinkedIn job {job_id}",
                                run_id=str(state.get("run_id") or "") or None,
                                job_id=job_id,
                                node="search_node",
                                details={"applied_status": bound_text(applied_status, 60)},
                            )
                            extracted_job_ids.add(job_id)
                            updated_seen_ids.add(job_id)
                            already_seen.add(job_id)
                            exclusions_to_persist.append(
                                {
                                    "job_id": job_id,
                                    "status": "skipped",
                                    "reason": "already_applied",
                                    "applied_status": applied_status,
                                }
                            )
                            continue

                        fresh_card = await page.query_selector(
                            f"li[data-occludable-job-id='{job_id}']"
                        )
                        if not fresh_card:
                            log_event(
                                "debug",
                                "search.card_detached",
                                f"Job {job_id} card no longer in DOM",
                                run_id=str(state.get("run_id") or "") or None,
                                job_id=job_id,
                                node="search_node",
                            )
                            continue

                        # Scroll card into view to trigger loading
                        try:
                            await fresh_card.scroll_into_view_if_needed()
                            await asyncio.sleep(0.5)
                        except Exception:
                            pass  # Continue even if scroll fails

                        # Wait for card elements to load (retry up to 3 times)
                        link_elem = None
                        company_elem = None
                        for attempt in range(3):
                            fresh_card = await page.query_selector(
                                f"li[data-occludable-job-id='{job_id}']"
                            )
                            if not fresh_card:
                                break

                            link_elem = await fresh_card.query_selector(
                                "a.job-card-container__link"
                            )
                            company_elem = await fresh_card.query_selector(
                                ".artdeco-entity-lockup__subtitle"
                            )

                            if link_elem and company_elem:
                                break

                            await asyncio.sleep(1)

                        # Extract basic info from card
                        if link_elem is None or company_elem is None:
                            log_event(
                                "debug",
                                "search.card_elements_unloaded",
                                f"Job {job_id} elements still not loaded after retries, skipping",
                                run_id=str(state.get("run_id") or "") or None,
                                job_id=job_id,
                                node="search_node",
                            )
                            continue

                        # Location is in the metadata wrapper
                        location_elem = await fresh_card.query_selector(
                            ".job-card-container__metadata-wrapper span"
                        )

                        # Title is in the aria-label of the link
                        title = await link_elem.get_attribute("aria-label")
                        if not title:
                            # Fallback: try to get from inner text
                            title = (await link_elem.inner_text()).strip()

                        title_exclusion_reason = get_title_exclusion_reason(
                            title,
                            settings.target_title_keywords_list,
                            exclude_senior_titles=settings.exclude_senior_titles,
                        )
                        if title_exclusion_reason:
                            log_event(
                                "debug",
                                "search.title_excluded",
                                f"Skipping excluded title {job_id}",
                                run_id=str(state.get("run_id") or "") or None,
                                job_id=job_id,
                                node="search_node",
                                details={"title": bound_text(title, 120)},
                            )
                            extracted_job_ids.add(job_id)
                            updated_seen_ids.add(job_id)
                            already_seen.add(job_id)
                            exclusions_to_persist.append(
                                {
                                    "job_id": job_id,
                                    "title": title,
                                    "company": (await company_elem.inner_text()).strip()
                                    if company_elem
                                    else "",
                                    "location": (await location_elem.inner_text()).strip()
                                    if location_elem
                                    else "Unknown",
                                    "status": "not_qualified",
                                    "reason": title_exclusion_reason,
                                }
                            )
                            continue

                        company = (await company_elem.inner_text()).strip()
                        location = (
                            (await location_elem.inner_text()).strip()
                            if location_elem
                            else "Unknown"
                        )

                        location_exclusion_reason = get_location_exclusion_reason(
                            {"location": location},
                            settings.excluded_locations_list,
                        )
                        if location_exclusion_reason:
                            log_event(
                                "info",
                                "search.location_excluded",
                                f"Skipping job in excluded location {job_id}",
                                run_id=str(state.get("run_id") or "") or None,
                                job_id=job_id,
                                node="search_node",
                                details={"location": bound_text(location, 120)},
                            )
                            extracted_job_ids.add(job_id)
                            updated_seen_ids.add(job_id)
                            already_seen.add(job_id)
                            exclusions_to_persist.append(
                                {
                                    "job_id": job_id,
                                    "title": title,
                                    "company": company,
                                    "location": location,
                                    "status": "not_qualified",
                                    "reason": location_exclusion_reason,
                                }
                            )
                            continue

                        # Non-destructive repost detection BEFORE any click or
                        # navigation, while the card is guaranteed attached:
                        # reads only footer/metadata/status elements and never
                        # excludes or deprioritizes a job.
                        repost_evidence = await find_repost_indicator_on_card(fresh_card)

                        log_event(
                            "debug",
                            "search.card_extracting",
                            f"Extracting job {job_id}",
                            run_id=str(state.get("run_id") or "") or None,
                            job_id=job_id,
                            node="search_node",
                            details={
                                "title": bound_text(title, 120),
                                "company": bound_text(company, 120),
                            },
                        )

                        # Guard immediately BEFORE card click
                        await guard_page_account_safety(page, stage="search_pre_card_click")

                        # Click to load full description
                        await link_elem.click()

                        # Guard card click navigation
                        await guard_page_account_safety(page, stage="search_card_click")

                        # Extract full description (wait for it to load)
                        try:
                            desc_elem = await page.wait_for_selector(
                                ".jobs-description-content__text, .jobs-description__content",
                                timeout=settings.job_load_timeout_ms,
                            )
                            raw_description = (await desc_elem.inner_text()).strip()
                        except Exception:
                            raw_description = "Description not available"

                        bounded_description = truncate_head_tail(
                            raw_description, max_chars=settings.max_job_description_chars
                        )

                        extracted_job_ids.add(job_id)
                        job_listings.append(
                            {
                                "job_id": job_id,
                                "title": title,
                                "company": company,
                                "location": location,
                                "url": f"https://www.linkedin.com/jobs/view/{job_id}",
                                "description": bounded_description,
                                "is_repost": bool(repost_evidence),
                                "repost_evidence": repost_evidence,
                            }
                        )
                        log_event(
                            "info",
                            "search.job_extracted",
                            f"✅ Extracted job {job_id}: {title} at {company}",
                            run_id=str(state.get("run_id") or "") or None,
                            job_id=job_id,
                            node="search_node",
                            details={
                                "is_repost": bool(repost_evidence),
                            },
                        )

                        await asyncio.sleep(get_randomized_delay() / 2)  # Small delay between cards

                    except Exception as e:
                        if isinstance(e, AccountSafetyBarrierError):
                            raise e
                        failed_job_id = job_id if "job_id" in locals() else "unknown"
                        log_event(
                            "warning",
                            "search.card_extraction_failed",
                            f"Error extracting job {failed_job_id}",
                            run_id=str(state.get("run_id") or "") or None,
                            job_id=str(failed_job_id) if failed_job_id != "unknown" else None,
                            node="search_node",
                            exc=e,
                        )
                        continue

                # Bulk persist all deterministic exclusions encountered on this page
                if exclusions_to_persist:
                    try:
                        await dedup.mark_seen_many(exclusions_to_persist)
                    except Exception as persist_err:
                        log_event(
                            "warning",
                            "search.exclusion_persist_failed",
                            "Failed to bulk-persist deterministic search exclusions",
                            run_id=str(state.get("run_id") or "") or None,
                            node="search_node",
                            exc=persist_err,
                        )

                await page.close()

                log_event(
                    "info",
                    "search.page_completed",
                    f"\n✅ Found {len(job_listings)} jobs for '{query}' page {page_num}",
                    run_id=str(state.get("run_id") or "") or None,
                    node="search_node",
                    details={"listing_count": len(job_listings), "page": page_num},
                )

                # Update trace with results
                if run_tree:
                    run_tree.metadata.update(
                        get_search_metadata(query, page_num, len(job_listings))
                    )

        return {
            "job_listings": job_listings,
            "current_job_index": 0,
            "current_job": None,
            "search_failed": False,
            "query_exhausted": False,
            "seen_job_ids": updated_seen_ids,
            "logs": list(state.get("logs") or [])
            + [f"Fetched {len(job_listings)} jobs for '{query}' page {page_num}"],
        }

    except AccountSafetyBarrierError as safety_err:
        try:
            if page is not None and not (
                page.is_closed() if callable(getattr(page, "is_closed", None)) else False
            ):
                await page.close()
        except Exception:
            pass
        return _account_safety_search_update(state, safety_err.detection)
    except Exception as e:
        error_msg = f"Search failed for '{query}' page {page_num}: {str(e)}"
        log_event(
            "error",
            "search.failed",
            error_msg,
            run_id=str(state.get("run_id") or "") or None,
            node="search_node",
            details={"query_index": query_index, "page": page_num},
            exc=e,
        )
        return {
            "job_listings": [],
            "current_job_index": 0,
            "current_job": None,
            "search_failed": True,
            "seen_job_ids": set(state.get("seen_job_ids") or set()),
            "errors": list(state.get("errors") or []) + [error_msg],
            "logs": list(state.get("logs") or []) + [error_msg],
        }
