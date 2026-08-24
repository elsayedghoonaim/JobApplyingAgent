"""Generation node - cover letter + urgency check."""

from langsmith.run_helpers import trace

from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.llm import get_llm
from jobapply.utils.paths import get_cover_letter_path
from jobapply.utils.prompts import get_cover_letter_prompt, get_urgency_check_prompt
from jobapply.utils.redaction import redact_string
from jobapply.utils.source_cache import (
    get_cached_profile,
    get_cached_resume_markdown,
)
from jobapply.utils.tracing import get_safe_job_metadata


def _data_path(*parts: str) -> str:
    """Build an absolute path under configured data_dir."""
    return get_settings().resolve_data_path(*parts)


async def generation_node(state: JobApplyState) -> dict:
    """Prepare per-job documents and decide whether resume edits need approval."""
    settings = get_settings()
    current_job = state.get("current_job")
    qual_result = state.get("qualification_result") or {}
    errors = list(state.get("errors") or [])

    base_resume_pdf = _data_path("resume.pdf")

    updates = {
        "cover_letter_text": "",
        "cover_letter_path": None,
        "resume_path": base_resume_pdf,
        "edits_urgent": False,
        "proposed_edits": None,
        "edit_reasoning": None,
        "approval_status": None,
        "approval_nonce": None,
        "application_status": None,
        "application_error": None,
        "form_qa_exchanges": None,
    }

    if not current_job:
        error_msg = "Generation skipped: no current job selected"
        return {
            **updates,
            "application_status": "failed",
            "application_error": error_msg,
            "errors": errors + [error_msg],
        }

    profile_text = ""
    try:
        _, profile_text = get_cached_profile()
    except Exception as e:
        errors.append(f"Profile load failed for generation: {redact_string(str(e))}")

    resume_text = ""
    try:
        resume_text = get_cached_resume_markdown()
    except Exception as e:
        errors.append(f"Resume markdown load failed for generation: {redact_string(str(e))}")

    try:
        llm = get_llm(temperature=0.4, max_output_tokens=768)
        cover_letter_prompt = get_cover_letter_prompt(
            profile_text,
            current_job,
            qual_result.get("key_matches", []),
        )

        async with trace(
            "llm_cover_letter_generation",
            run_type="chain",
            metadata=get_safe_job_metadata(current_job, include_description=False),
        ) as run_tree:
            cover_letter_result = await llm.ainvoke(cover_letter_prompt)
            cover_letter_text = getattr(cover_letter_result, "content", str(cover_letter_result))
            cover_letter_path = str(
                get_cover_letter_path(
                    run_id=state.get("run_id", "default_run"),
                    job_id=str(current_job.get("job_id", "unknown")),
                )
            )

            with open(cover_letter_path, "w", encoding="utf-8") as f:
                f.write(cover_letter_text)

            updates["cover_letter_text"] = cover_letter_text
            updates["cover_letter_path"] = cover_letter_path

            if run_tree:
                run_tree.metadata["cover_letter_path"] = cover_letter_path
                run_tree.metadata["cover_letter_length"] = len(cover_letter_text)
    except Exception as e:
        error_msg = f"Cover letter generation failed for {current_job.get('title', 'unknown job')}: {redact_string(str(e))}"
        errors.append(error_msg)
        updates["application_status"] = "failed"
        updates["application_error"] = error_msg
        return {**updates, "errors": errors}

    score = qual_result.get("score", 0.0)
    if score >= settings.urgent_edit_threshold and resume_text:
        try:
            llm = get_llm(
                temperature=0.2,
                max_output_tokens=768,
                response_mime_type="application/json",
                response_json_schema={
                    "type": "object",
                    "properties": {
                        "edits_urgent": {"type": "boolean"},
                        "proposed_edits": {"type": "string"},
                        "edit_reasoning": {"type": "string"},
                    },
                    "required": ["edits_urgent", "proposed_edits", "edit_reasoning"],
                },
            )
            urgency_prompt = get_urgency_check_prompt(
                resume_text,
                current_job.get("description", ""),
                score,
            )

            async with trace(
                "llm_urgency_check",
                run_type="chain",
                metadata={
                    **get_safe_job_metadata(current_job, include_description=False),
                    "score": score,
                    "threshold": settings.urgent_edit_threshold,
                },
            ) as run_tree:
                urgency_result = await llm.ainvoke(urgency_prompt)
                urgency_text = getattr(urgency_result, "content", str(urgency_result))
                urgency_data = extract_json_object(urgency_text)

                proposed_edits = urgency_data.get("proposed_edits") or ""
                edits_urgent = bool(urgency_data.get("edits_urgent")) and bool(
                    proposed_edits.strip()
                )

                updates["edits_urgent"] = edits_urgent
                updates["proposed_edits"] = proposed_edits if edits_urgent else None
                updates["edit_reasoning"] = (
                    urgency_data.get("edit_reasoning") if edits_urgent else None
                )

                if run_tree:
                    run_tree.metadata["edits_urgent"] = edits_urgent
                    run_tree.metadata["has_proposed_edits"] = bool(proposed_edits)
        except Exception as e:
            errors.append(
                f"Urgency check failed for {current_job.get('title', 'unknown job')}: {redact_string(str(e))}"
            )

    return {**updates, "errors": errors}
