"""Search node - LinkedIn job search via Playwright."""

import asyncio
import re
from urllib.parse import urlencode

from langsmith.run_helpers import trace

from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.browser import get_randomized_delay, managed_browser
from jobapply.utils.job_filters import (
    find_disallowed_required_languages,
    is_senior_position_title,
)
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.llm import get_llm
from jobapply.utils.prompts import get_job_parser_prompt
from jobapply.utils.tracing import get_search_metadata


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
        State updates dict with job_listings and reset current_job_index.
    """
    settings = get_settings()

    query_index = state["current_query_index"]
    page_num = state["current_page"]
    query = state["search_queries"][query_index]

    search_url = build_search_url(settings.linkedin_base_url, query, page_num)

    job_listings = []

    print(f"🔍 Searching: '{query}' (page {page_num}/{state['pages_per_query']})")

    try:
        async with managed_browser() as (browser, context):
            async with trace(
                "linkedin_search",
                run_type="tool",
                metadata={"search_url": search_url, "query": query, "page": page_num},
            ) as run_tree:
                page = await context.new_page()

                # Navigate to search results
                await page.goto(search_url, wait_until="domcontentloaded")

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
                        "logs": state["logs"] + [f"No results for query '{query}' page {page_num}"],
                    }

                # Extract job cards (updated selector)
                job_cards = await page.query_selector_all("li[data-occludable-job-id]")

                print(f"[DEBUG] Found {len(job_cards)} job card elements")

                # Determine how many unseen jobs we need to extract in this batch
                unseen_limit = 25
                max_jobs = state.get("max_jobs_to_evaluate")
                if max_jobs is not None:
                    remaining_evals = max_jobs - state.get("jobs_evaluated_count", 0)
                    unseen_limit = min(unseen_limit, remaining_evals)

                max_apps = state.get("max_applications")
                if max_apps is not None:
                    remaining_apps = max_apps - state.get("applications_count", 0)
                    # Fetch at least 2 unseen jobs to allow a safety buffer if one is not qualified
                    unseen_limit = min(unseen_limit, max(remaining_apps * 2, 2))

                print(f"[DEBUG] Target unseen jobs for this batch: {unseen_limit}")

                # Track job IDs to detect duplicates during extraction
                extracted_job_ids = set()

                for card in job_cards[:25]:  # Limit to 25 per page
                    try:
                        # Extract job ID from data attribute (faster than clicking)
                        job_id = await card.get_attribute("data-occludable-job-id")

                        if not job_id:
                            print("[DEBUG] Skipping card - no job_id")
                            continue

                        # Check if already seen in previous runs/sessions
                        if job_id in state["seen_job_ids"]:
                            print(f"[DEBUG] Skipping already-seen job {job_id}")
                            continue

                        applied_status = await find_applied_status_on_card(card)
                        if applied_status:
                            print(
                                f"[DEBUG] Skipping already-applied LinkedIn job {job_id} "
                                f"(status: {applied_status})"
                            )
                            state["seen_job_ids"].add(job_id)
                            continue

                        # Check if we already have enough unseen jobs in this batch
                        if len(job_listings) >= unseen_limit:
                            print(
                                f"[DEBUG] Already have enough unseen jobs in this batch ({len(job_listings)} >= {unseen_limit}), stopping card extraction."
                            )
                            break

                        # Check for duplicate in current extraction
                        if job_id in extracted_job_ids:
                            print(f"[DEBUG] Duplicate job_id {job_id} found in same page, skipping")
                            continue
                        extracted_job_ids.add(job_id)

                        # Re-query elements fresh each time to avoid detached DOM issues
                        # Use job_id to find the card again
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
                            # Re-query the card each attempt
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

                            # Wait a bit and try again
                            await asyncio.sleep(1)

                        # Extract basic info from card (updated selectors)
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
                            state["seen_job_ids"].add(job_id)
                            continue

                        company = (await company_elem.inner_text()).strip()
                        location = (
                            (await location_elem.inner_text()).strip()
                            if location_elem
                            else "Unknown"
                        )
                        job_url = await link_elem.get_attribute("href")

                        print(f"[DEBUG] Extracting: {title} at {company}")

                        # Click to load full description
                        await link_elem.click()

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
                            print(
                                f"[DEBUG] Skipping job {job_id}; required language(s) "
                                f"not allowed: {', '.join(disallowed_languages)}"
                            )
                            state["seen_job_ids"].add(job_id)
                            continue

                        job_listings.append(
                            {
                                "job_id": job_id,
                                "title": title,
                                "company": company,
                                "location": location,
                                "url": f"https://www.linkedin.com/jobs/view/{job_id}",
                                "description": parsed.get("clean_description", raw_description),
                                # Only include parsed fields that add new information
                                "parsed_location": parsed.get("parsed_location"),
                                "duration": parsed.get("duration"),
                                "work_type": parsed.get("work_type"),
                                "responsibilities": parsed.get("responsibilities", []),
                                "requirements": parsed.get("requirements", []),
                                "required_languages": parsed.get("required_languages", []),
                            }
                        )

                        # Check if we have extracted enough new unseen jobs for this batch
                        if len(job_listings) >= unseen_limit:
                            print(
                                f"[DEBUG] Reached target unseen jobs limit ({unseen_limit}), stopping extraction."
                            )
                            break

                        await asyncio.sleep(get_randomized_delay() / 2)  # Small delay between cards

                    except Exception as e:
                        # Skip problematic cards
                        job_id_str = locals().get("job_id", "unknown")
                        print(f"[DEBUG] Error extracting job {job_id_str}: {str(e)}")
                        import traceback

                        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                        continue

                await page.close()

                print(f"✅ Found {len(job_listings)} jobs for '{query}' page {page_num}")

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
            "logs": state["logs"]
            + [f"Fetched {len(job_listings)} jobs for '{query}' page {page_num}"],
        }

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
            "errors": state["errors"] + [error_msg],
            "logs": state["logs"] + [error_msg],
        }
