"""Qualification node - LLM job-fit evaluation with combined structured description extraction."""

from langsmith.run_helpers import trace

from jobapply.models.job import CombinedJobAnalysis
from jobapply.nodes.outcomes import job_repost_persistence_fields
from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.job_filters import (
    find_disallowed_required_languages,
    get_title_exclusion_reason,
)
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.llm import get_llm
from jobapply.utils.observability import log_event
from jobapply.utils.prompts import get_qualification_prompt, truncate_head_tail
from jobapply.utils.source_cache import get_cached_profile
from jobapply.utils.tracing import get_qualification_metadata, get_safe_job_metadata


async def qualification_node(state: JobApplyState) -> dict:
    """Evaluate job fit via single combined LLM call with structured output.

    Args:
        state: Current graph state.

    Returns:
        State updates dict with qualification result and updated current_job.
    """
    settings = get_settings()
    current_job = state.get("current_job")

    if not current_job:
        return {
            "qualification_result": {
                "qualified": False,
                "score": 0.0,
                "reasoning": "No job to evaluate",
                "key_matches": [],
                "gaps": [],
            }
        }

    exclusion_reason = get_title_exclusion_reason(
        current_job.get("title"),
        settings.target_title_keywords_list,
        exclude_senior_titles=settings.exclude_senior_titles,
    )
    if exclusion_reason:
        log_event(
            "info",
            "qualification.title_excluded",
            f"⛔ EXCLUDED - {exclusion_reason} | {current_job.get('title')}",
            run_id=str(state.get("run_id") or "") or None,
            job_id=current_job.get("job_id"),
            node="qualification_node",
            outcome="not_qualified",
        )
        seen_job_ids = set(state.get("seen_job_ids") or set())
        errors = list(state.get("errors") or [])
        if current_job.get("job_id"):
            seen_job_ids.add(current_job["job_id"])
            try:
                dedup = DeduplicationStore()
                await dedup.mark_seen(
                    current_job["job_id"],
                    {
                        "job_id": current_job["job_id"],
                        "title": current_job.get("title"),
                        "company": current_job.get("company"),
                        "location": current_job.get("location"),
                        "url": current_job.get("url"),
                        "status": "not_qualified",
                        "reason": exclusion_reason,
                    },
                )
            except Exception as e:
                err_msg = (
                    f"Failed to persist exclusion for {current_job['job_id']}: ({type(e).__name__})"
                )
                log_event(
                    "debug",
                    "qualification.exclusion_persist_failed",
                    err_msg,
                    run_id=str(state.get("run_id") or "") or None,
                    job_id=current_job["job_id"],
                    node="qualification_node",
                    exc=e,
                )
                errors.append(err_msg)

        return {
            "qualification_result": {
                "qualified": False,
                "score": 0.0,
                "reasoning": exclusion_reason,
                "key_matches": [],
                "gaps": [exclusion_reason],
                "job_summary": "Excluded before qualification.",
            },
            "seen_job_ids": seen_job_ids,
            "jobs_evaluated_count": state.get("jobs_evaluated_count", 0) + 1,
            "qualified_jobs_count": state.get("qualified_jobs_count", 0),
            "not_qualified_jobs_count": state.get("not_qualified_jobs_count", 0) + 1,
            "errors": errors,
        }

    # Load user profile from process-local cache
    errors = list(state.get("errors") or [])
    try:
        _, profile_text = get_cached_profile()
    except Exception as e:
        profile_text = ""
        errors.append(f"Profile load failed for qualification: {type(e).__name__}")

    # Build prompt
    prompt = get_qualification_prompt(profile_text, current_job)

    # Call LLM with combined structured output and 4096 token output budget
    llm = get_llm(
        temperature=0.2,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_json_schema=CombinedJobAnalysis.model_json_schema(),
    )

    try:
        async with trace(
            "llm_qualification_scoring",
            run_type="chain",
            metadata=get_safe_job_metadata(current_job, include_description=False),
        ) as run_tree:
            # Get raw response first
            raw_response = await llm.ainvoke(prompt)
            response_text = (
                raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            )

            # Clean markdown formatting if present
            result_dict = extract_json_object(response_text)
            result = CombinedJobAnalysis(**result_dict)

            # Bounded raw description from input
            raw_job_description = current_job.get("description", "")
            bounded_raw_description = truncate_head_tail(
                raw_job_description, max_chars=settings.max_job_description_chars
            )

            # Apply deterministic required-language exclusion after combined analysis using BOTH:
            # 1. original bounded raw job description (not model-produced clean_description)
            # 2. CombinedJobAnalysis.required_languages
            disallowed_languages = find_disallowed_required_languages(
                {
                    "description": bounded_raw_description,
                    "required_languages": result.required_languages,
                },
                settings.allowed_languages_list,
            )

            if disallowed_languages:
                lang_str = ", ".join(disallowed_languages)
                exclusion_reason = f"Disallowed required language: {lang_str}"
                qualified = False
                effective_score = 0.0
                effective_reasoning = (
                    f"{result.reasoning} | {exclusion_reason}"
                    if result.reasoning
                    else exclusion_reason
                )
                effective_gaps = list(result.gaps) + [exclusion_reason]
                status = "not_qualified"
                log_event(
                    "info",
                    "qualification.language_excluded",
                    f"⛔ Disallowed language(s) required ({lang_str}) | {current_job.get('title')}",
                    run_id=str(state.get("run_id") or "") or None,
                    job_id=current_job.get("job_id"),
                    node="qualification_node",
                    outcome="not_qualified",
                )
            else:
                qualified = result.score >= settings.qualification_threshold
                effective_score = result.score
                effective_reasoning = result.reasoning
                effective_gaps = result.gaps
                status = "qualified" if qualified else "not_qualified"
                exclusion_reason = None

                if qualified:
                    log_event(
                        "info",
                        "qualification.completed",
                        f"✅ QUALIFIED - Score: {effective_score:.2f} | {current_job.get('title')} at {current_job.get('company')}",
                        run_id=str(state.get("run_id") or "") or None,
                        job_id=current_job.get("job_id"),
                        node="qualification_node",
                        outcome="qualified",
                        details={"score": float(effective_score)},
                    )
                else:
                    log_event(
                        "info",
                        "qualification.completed",
                        f"❌ Not qualified - Score: {effective_score:.2f} (threshold: {settings.qualification_threshold}) | {current_job.get('title')}",
                        run_id=str(state.get("run_id") or "") or None,
                        job_id=current_job.get("job_id"),
                        node="qualification_node",
                        outcome="not_qualified",
                        details={
                            "score": float(effective_score),
                            "threshold": float(settings.qualification_threshold),
                        },
                    )

            # Update trace with qualification result
            if run_tree:
                run_tree.metadata.update(
                    get_qualification_metadata(
                        qualified=qualified,
                        score=effective_score,
                        threshold=settings.qualification_threshold,
                        key_matches_count=len(result.key_matches),
                        gaps_count=len(effective_gaps),
                    )
                )

        # Bound clean_description before storing in current_job or MongoDB
        if result.clean_description:
            bounded_clean_description = truncate_head_tail(
                result.clean_description, max_chars=settings.max_clean_description_chars
            )
        else:
            bounded_clean_description = bounded_raw_description

        # Build immutable updated current_job with extracted structured fields
        updated_job = {
            **current_job,
            "description": bounded_clean_description,
            "parsed_location": result.parsed_location or current_job.get("parsed_location"),
            "duration": result.duration or current_job.get("duration"),
            "work_type": result.work_type or current_job.get("work_type"),
            "responsibilities": list(
                result.responsibilities or current_job.get("responsibilities") or []
            ),
            "requirements": list(result.requirements or current_job.get("requirements") or []),
            "required_languages": list(
                result.required_languages or current_job.get("required_languages") or []
            ),
        }

        # Mark job as seen in dedup store with full details
        try:
            async with trace(
                "mongodb_store_job", run_type="tool", metadata={"job_id": current_job["job_id"]}
            ):
                dedup = DeduplicationStore()

                job_data = {
                    "job_id": current_job["job_id"],
                    "title": current_job.get("title"),
                    "company": current_job.get("company"),
                    "location": current_job.get("location"),
                    "url": current_job.get("url"),
                    "description": updated_job["description"],
                    "job_summary": result.job_summary,
                    "qualification_score": effective_score,
                    "qualification_reasoning": effective_reasoning,
                    "key_matches": result.key_matches,
                    "gaps": effective_gaps,
                    "status": status,
                    **job_repost_persistence_fields(current_job),
                }

                if exclusion_reason:
                    job_data["reason"] = exclusion_reason
                if disallowed_languages:
                    job_data["disallowed_languages"] = disallowed_languages

                if result.parsed_location and result.parsed_location != current_job.get("location"):
                    job_data["parsed_location"] = result.parsed_location

                if result.duration:
                    job_data["duration"] = result.duration
                if result.work_type:
                    job_data["work_type"] = result.work_type
                if result.responsibilities:
                    job_data["responsibilities"] = result.responsibilities
                if result.requirements:
                    job_data["requirements"] = result.requirements
                if result.required_languages:
                    job_data["required_languages"] = result.required_languages

                await dedup.mark_seen(current_job["job_id"], job_data)
        except Exception as store_err:
            err_msg = f"Failed to persist qualification for {current_job.get('job_id')}: ({type(store_err).__name__})"
            log_event(
                "debug",
                "qualification.persist_failed",
                err_msg,
                run_id=str(state.get("run_id") or "") or None,
                job_id=current_job.get("job_id"),
                node="qualification_node",
                exc=store_err,
            )
            errors.append(err_msg)

        # Add to in-memory seen set (copying state immutably)
        seen_job_ids = set(state.get("seen_job_ids") or set())
        if current_job.get("job_id"):
            seen_job_ids.add(current_job["job_id"])

        # Increment jobs evaluated counter
        jobs_evaluated = state.get("jobs_evaluated_count", 0) + 1

        # Increment qualified/not_qualified counters
        qualified_jobs = state.get("qualified_jobs_count", 0)
        not_qualified_jobs = state.get("not_qualified_jobs_count", 0)
        if qualified:
            qualified_jobs += 1
        else:
            not_qualified_jobs += 1

        return {
            "qualification_result": {
                "qualified": qualified,
                "score": effective_score,
                "reasoning": effective_reasoning,
                "key_matches": result.key_matches,
                "gaps": effective_gaps,
                "job_summary": result.job_summary,
            },
            "current_job": updated_job,
            "seen_job_ids": seen_job_ids,
            "jobs_evaluated_count": jobs_evaluated,
            "qualified_jobs_count": qualified_jobs,
            "not_qualified_jobs_count": not_qualified_jobs,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"Qualification error for {current_job.get('title', 'unknown')}: {str(e)}"
        jobs_evaluated = state.get("jobs_evaluated_count", 0) + 1
        log_event(
            "error",
            "qualification.failed",
            f"❌ ERROR during qualification: {current_job.get('title', 'unknown')}",
            run_id=str(state.get("run_id") or "") or None,
            job_id=current_job.get("job_id"),
            node="qualification_node",
            outcome="failed",
            exc=e,
        )
        return {
            "qualification_result": {
                "qualified": False,
                "score": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "key_matches": [],
                "gaps": [],
            },
            "seen_job_ids": set(state.get("seen_job_ids") or set()),
            "jobs_evaluated_count": jobs_evaluated,
            "errors": list(state.get("errors") or []) + [error_msg],
        }
