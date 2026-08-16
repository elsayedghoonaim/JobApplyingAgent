"""Qualification node - LLM job-fit evaluation."""

import yaml
from langsmith.run_helpers import trace

from jobapply.models.job import QualificationResult
from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.job_filters import get_job_exclusion_reason
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.llm import get_llm
from jobapply.utils.prompts import get_qualification_prompt
from jobapply.utils.tracing import get_qualification_metadata, get_safe_job_metadata


async def qualification_node(state: JobApplyState) -> dict:
    """Evaluate job fit via LLM with structured output.

    Args:
        state: Current graph state.

    Returns:
        State updates dict with qualification result.
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

    exclusion_reason = get_job_exclusion_reason(current_job)
    if exclusion_reason:
        print(f"⛔ EXCLUDED - {exclusion_reason} | {current_job.get('title')}")
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
                print(f"[DEBUG] {err_msg}")
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

    # Load user profile
    profile_path = get_settings().resolve_data_path("profile.yaml")
    with open(profile_path, "r", encoding="utf-8") as f:
        profile_data = yaml.safe_load(f)
    profile_text = yaml.dump(profile_data)

    # Build prompt
    prompt = get_qualification_prompt(profile_text, current_job)

    # Call LLM with structured output
    llm = get_llm(
        temperature=0.2,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_json_schema=QualificationResult.model_json_schema(),
    )

    errors = list(state.get("errors") or [])
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
            result = QualificationResult(**result_dict)

            # Check threshold
            qualified = result.score >= settings.qualification_threshold

            if qualified:
                print(
                    f"✅ QUALIFIED - Score: {result.score:.2f} | {current_job.get('title')} at {current_job.get('company')}"
                )
            else:
                print(
                    f"❌ Not qualified - Score: {result.score:.2f} (threshold: {settings.qualification_threshold}) | {current_job.get('title')}"
                )

            # Update trace with qualification result
            if run_tree:
                run_tree.metadata.update(
                    get_qualification_metadata(
                        qualified=qualified,
                        score=result.score,
                        threshold=settings.qualification_threshold,
                        key_matches_count=len(result.key_matches),
                        gaps_count=len(result.gaps),
                    )
                )

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
                    "description": current_job.get("description"),
                    "job_summary": result.job_summary,
                    "qualification_score": result.score,
                    "qualification_reasoning": result.reasoning,
                    "key_matches": result.key_matches,
                    "gaps": result.gaps,
                    "status": "qualified" if qualified else "not_qualified",
                }

                parsed_location = current_job.get("parsed_location")
                if parsed_location and parsed_location != current_job.get("location"):
                    job_data["parsed_location"] = parsed_location

                if current_job.get("duration"):
                    job_data["duration"] = current_job["duration"]
                if current_job.get("work_type"):
                    job_data["work_type"] = current_job["work_type"]
                if current_job.get("responsibilities"):
                    job_data["responsibilities"] = current_job["responsibilities"]
                if current_job.get("requirements"):
                    job_data["requirements"] = current_job["requirements"]

                await dedup.mark_seen(current_job["job_id"], job_data)
        except Exception as store_err:
            err_msg = f"Failed to persist qualification for {current_job.get('job_id')}: ({type(store_err).__name__})"
            print(f"[DEBUG] {err_msg}")
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
                "score": result.score,
                "reasoning": result.reasoning,
                "key_matches": result.key_matches,
                "gaps": result.gaps,
                "job_summary": result.job_summary,
            },
            "seen_job_ids": seen_job_ids,
            "jobs_evaluated_count": jobs_evaluated,
            "qualified_jobs_count": qualified_jobs,
            "not_qualified_jobs_count": not_qualified_jobs,
            "errors": errors,
        }

    except Exception as e:
        error_msg = f"Qualification error for {current_job.get('title', 'unknown')}: {str(e)}"
        jobs_evaluated = state.get("jobs_evaluated_count", 0) + 1
        print(f"❌ ERROR during qualification: {current_job.get('title', 'unknown')} - {str(e)}")
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
