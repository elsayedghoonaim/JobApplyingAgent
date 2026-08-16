"""Search node - LinkedIn job search via Playwright."""

import asyncio
import re
from urllib.parse import urlencode

from langsmith.run_helpers import trace

from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.account_safety import (
    AccountSafetyBarrierError,
    guard_page_account_safety,
    normalize_safety_log_payload,
    sanitize_evidence_string,
)
from jobapply.utils.browser import get_randomized_delay, managed_browser
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.job_filters import (
    find_disallowed_required_languages,
    is_senior_position_title,
)
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.llm import get_llm
from jobapply.utils.prompts import get_job_parser_prompt
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


def build_search_url(base_url: str, query: str, page_num: int) -> str:
    """Build a correctly encoded LinkedIn Easy Apply search URL."""
    params = {
        "keywords": query,
        "f_AL": "true",
        "location": "Worldwide",
        "f_TPR": "r86400",
        "start": (page_num - 1) * 25,
    }
    return f"{base_url.rstrip('/')}/jobs/search/?{urlencode(params)}"


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
        print(f"[DEBUG] Job parsing failed: {e}")
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


async def search_node(state: JobApplyState) -> dict:
    """Fetch job listings from LinkedIn for current query + page.

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

    search_url = build_search_url(settings.linkedin_base_url, query, page_num)

    job_listings = []

    print(f"🔍 Searching: '{query}' (page {page_num}/{state['pages_per_query']})")

    page = None
    try:
        async with managed_browser() as (browser, context):
            async with trace(
                "linkedin_search",
                run_type="tool",
                metadata={"search_url": search_url, "query": query, "page": page_num},
            ) as run_tree:
                page = await context.new_page()

                # Navigate to search results
                response = await page.goto(search_url, wait_until="domcontentloaded")
                http_status = response.status if response else None

                # Guard search navigation
                await guard_page_account_safety(
                    page, stage="search_navigation", http_status=http_status
                )

                # Wait for job cards to load (updated selector for new LinkedIn structure)
                try:
                    await page.wait_for_selector(".scaffold-layout__list", timeout=10000)
                    # Additional wait for job cards to populate
                    await page.wait_for_selector("li[data-occludable-job-id]", timeout=10000)

                    # Scroll to load all job cards (LinkedIn lazy loads)
                    print("[DEBUG] Scrolling to load all jobs...")
                    for scroll_attempt in range(3):  # Scroll 3 times to load more cards
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await asyncio.sleep(1.5)

                    # Wait a bit more for lazy-loaded cards
                    await asyncio.sleep(2)
                except Exception:
                    # No results found
                    print(f"❌ No results found for '{query}' page {page_num}")
                    if run_tree:
                        run_tree.metadata.update(get_search_metadata(query, page_num, 0))
                    return {
                        "job_listings": [],
                        "current_job_index": 0,
                        "current_job": None,
                        "search_failed": False,
                        "seen_job_ids": set(state.get("seen_job_ids") or set()),
                        "logs": list(state.get("logs") or [])
                        + [f"No results for query '{query}' page {page_num}"],
                    }

                # Extract all job cards on the page without arbitrary truncation
                job_cards = await page.query_selector_all("li[data-occludable-job-id]")
                print(f"[DEBUG] Found {len(job_cards)} job card elements")

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
                            print("[DEBUG] Skipping card - no job_id")
                            continue

                        # Check if already seen in previous runs/sessions or state
                        if job_id in already_seen:
                            print(f"[DEBUG] Skipping already-seen job {job_id}")
                            continue

                        # Check for duplicate in current extraction
                        if job_id in extracted_job_ids:
                            print(f"[DEBUG] Duplicate job_id {job_id} found in same page, skipping")
                            continue

                        applied_status = await find_applied_status_on_card(card)
                        if applied_status:
                            print(
                                f"[DEBUG] Skipping already-applied LinkedIn job {job_id} "
                                f"(status: {applied_status})"
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

                        # Re-query elements fresh each time to avoid detached DOM issues
                        fresh_card = await page.query_selector(
                            f"li[data-occludable-job-id='{job_id}']"
                        )
                        if not fresh_card:
                            print(f"[DEBUG] Job {job_id} - card no longer in DOM")
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
                            print(
                                f"[DEBUG] Job {job_id} - elements still not loaded after retries, skipping"
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

                        if is_senior_position_title(title):
                            print(f"[DEBUG] Skipping senior-level position {job_id}: {title}")
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
                                    "reason": f"Senior-level position excluded: {title}",
                                }
                            )
                            continue

                        company = (await company_elem.inner_text()).strip()
                        location = (
                            (await location_elem.inner_text()).strip()
                            if location_elem
                            else "Unknown"
                        )

                        print(f"[DEBUG] Extracting: {title} at {company}")

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

                        # Parse job description to extract structured fields
                        parsed = await parse_job_description(raw_description)

                        disallowed_languages = find_disallowed_required_languages(
                            {
                                "description": raw_description,
                                "required_languages": parsed.get("required_languages", []),
                            }
                        )
                        if disallowed_languages:
                            lang_str = ", ".join(disallowed_languages)
                            print(
                                f"[DEBUG] Skipping job {job_id}; required language(s) "
                                f"not allowed: {lang_str}"
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
                                    "reason": f"Disallowed required language: {lang_str}",
                                    "disallowed_languages": disallowed_languages,
                                }
                            )
                            continue

                        extracted_job_ids.add(job_id)
                        job_listings.append(
                            {
                                "job_id": job_id,
                                "title": title,
                                "company": company,
                                "location": location,
                                "url": f"https://www.linkedin.com/jobs/view/{job_id}",
                                "description": parsed.get("clean_description", raw_description),
                                "parsed_location": parsed.get("parsed_location"),
                                "duration": parsed.get("duration"),
                                "work_type": parsed.get("work_type"),
                                "responsibilities": parsed.get("responsibilities", []),
                                "requirements": parsed.get("requirements", []),
                                "required_languages": parsed.get("required_languages", []),
                            }
                        )
                        print(f"✅ Extracted job {job_id}: {title} at {company}")

                        await asyncio.sleep(get_randomized_delay() / 2)  # Small delay between cards

                    except Exception as e:
                        if isinstance(e, AccountSafetyBarrierError):
                            raise e
                        print(
                            f"[DEBUG] Error extracting job {job_id if 'job_id' in locals() else 'unknown'}: {e}"
                        )
                        import traceback

                        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                        continue

                # Bulk persist all deterministic exclusions encountered on this page
                if exclusions_to_persist:
                    try:
                        await dedup.mark_seen_many(exclusions_to_persist)
                    except Exception as persist_err:
                        print(f"[DEBUG] Failed to bulk-persist exclusions: {persist_err}")

                await page.close()

                print(f"\n✅ Found {len(job_listings)} jobs for '{query}' page {page_num}")

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
        print(f"[ERROR] {error_msg}")
        import traceback

        print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
        return {
            "job_listings": [],
            "current_job_index": 0,
            "current_job": None,
            "search_failed": True,
            "seen_job_ids": set(state.get("seen_job_ids") or set()),
            "errors": list(state.get("errors") or []) + [error_msg],
            "logs": list(state.get("logs") or []) + [error_msg],
        }
