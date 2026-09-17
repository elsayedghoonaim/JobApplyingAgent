"""Approval node - Telegram human-in-the-loop for urgent edits."""

import os
from html import escape
from typing import Any

from jobapply.models.telegram import CorrelationStatus, CorrelationWaitResult
from jobapply.settings import get_settings
from jobapply.state import JobApplyState
from jobapply.utils.dedup import canonicalize_job_id
from jobapply.utils.llm import get_llm
from jobapply.utils.paths import get_edited_resume_path
from jobapply.utils.pdf import markdown_to_pdf
from jobapply.utils.prompts import get_resume_edit_prompt
from jobapply.utils.redaction import redact_string
from jobapply.utils.telegram import TelegramClient


def _data_path(*parts: str) -> str:
    """Build an absolute path under configured data_dir."""
    return get_settings().resolve_data_path(*parts)


def format_approval_question(state: JobApplyState) -> str:
    """Build a structured Telegram question for the legacy edit-approval path."""
    job = state.get("current_job") or {}
    qualification = state.get("qualification_result") or {}
    reasoning = " ".join(str(state.get("edit_reasoning") or "No reasoning provided").split())
    proposed = str(state.get("proposed_edits") or "No changes proposed").strip()
    if len(reasoning) > 700:
        reasoning = reasoning[:697].rstrip() + "..."
    if len(proposed) > 1400:
        proposed = proposed[:1397].rstrip() + "..."

    lines = [
        "<b>📝 RESUME DECISION REQUIRED</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        "",
        "<b>JOB DETAILS</b>",
        f"• Role: {escape(str(job.get('title') or 'Unknown title'))}",
        f"• Company: {escape(str(job.get('company') or 'Unknown company'))}",
        f"• Fit score: {float(qualification.get('score') or 0):.0%}",
    ]
    if job.get("url"):
        lines.extend(["", "<b>JOB LINK</b>", escape(str(job["url"]))])
    lines.extend(
        [
            "",
            "<b>WHY AN EDIT WAS SUGGESTED</b>",
            escape(reasoning),
            "",
            "<b>PROPOSED CHANGES</b>",
            escape(proposed),
            "",
            "<b>ACTION REQUIRED</b>",
            "• Approve — use the edited resume.",
            "• Use Base — continue with the original resume.",
            "• Skip — skip this job.",
        ]
    )
    return "\n".join(lines)


async def approval_node(state: JobApplyState) -> dict:
    """Ask user via Telegram for approval of urgent resume edits."""
    settings = get_settings()
    telegram = TelegramClient()

    current_job = state.get("current_job") or {}
    qual_result = state.get("qualification_result") or {}
    run_id = str(state.get("run_id") or "default_run")
    canonical_job_id = canonicalize_job_id(current_job.get("job_id", "unknown")) or "unknown"
    corr_key = f"approval:{run_id}:{canonical_job_id}"

    raw_message = format_approval_question(state)

    errors = list(state.get("errors") or [])
    try:
        wait_res = await telegram.send_and_wait_for_reply(
            correlation_key=corr_key,
            purpose="approval",
            run_id=run_id,
            prompt_text=raw_message,
            timeout=settings.approval_timeout_seconds,
            job_id=canonical_job_id,
            inline_actions=["✅ Approve", "📄 Use Base", "❌ Skip"],
        )
    except Exception as exc:
        wait_res = CorrelationWaitResult(
            status=CorrelationStatus.STORAGE_ERROR,
            error_reason=f"Approval execution exception ({type(exc).__name__})",
        )

    if wait_res.status == CorrelationStatus.REPLIED and wait_res.reply_text:
        reply = wait_res.reply_text
    elif wait_res.status == CorrelationStatus.TIMED_OUT:
        reply = None
    else:
        reply = None
        errors.append(
            f"Telegram approval correlation failed ({wait_res.status.value}): {wait_res.error_reason or 'storage_or_delivery_error'}"
        )
    active_nonce = wait_res.nonce or ""

    # Callback presses map by persisted index to exact decisions; the rendered
    # button label is never used for identity.
    if wait_res.reply_kind == "callback" and wait_res.reply_option_index is not None:
        callback_actions = ["✅ Approve", "📄 Use Base", "❌ Skip"]
        index = wait_res.reply_option_index
        if 0 <= index < len(callback_actions):
            approval_status = {
                0: "approved",
                1: "use_base",
                2: "skip",
            }[index]
        else:
            approval_status = "use_base"
    # Parse reply
    elif reply is None:
        approval_status = "use_base"
    elif "approve" in reply.lower() or "✅" in reply:
        approval_status = "approved"
    elif "base" in reply.lower() or "📄" in reply:
        approval_status = "use_base"
    elif "skip" in reply.lower() or "❌" in reply:
        approval_status = "skip"
    else:
        approval_status = "use_base"

    updates: dict[str, Any] = {
        "approval_status": approval_status,
        "approval_nonce": active_nonce,
    }
    if errors:
        updates["errors"] = errors

    if approval_status == "skip":
        skipped_jobs = list(state.get("skipped_jobs") or [])
        outcomes = list(state.get("application_outcomes") or [])

        outcome = {
            "job_id": current_job.get("job_id", "unknown"),
            "title": current_job.get("title", "Unknown"),
            "company": current_job.get("company", "Unknown"),
            "url": current_job.get("url"),
            "score": qual_result.get("score", 0.0),
            "status": "skipped",
            "dry_run": bool(state.get("dry_run")),
            "reason": "user_skipped",
            "resume_path": state.get("resume_path"),
            "cover_letter_path": state.get("cover_letter_path"),
            "resume_edited": False,
            "qa_count": 0,
        }
        skipped_jobs.append(outcome)
        outcomes.append(outcome)

        updates["skipped_jobs"] = skipped_jobs
        updates["application_outcomes"] = outcomes
        updates["application_status"] = "skipped"

    elif approval_status == "approved":
        resume_markdown = ""
        try:
            with open(_data_path("resume.md"), "r", encoding="utf-8") as f:
                resume_markdown = f.read()
        except Exception as e:
            updates["errors"] = list(updates.get("errors") or state.get("errors") or []) + [
                f"Failed to load resume.md for editing: {redact_string(str(e))}"
            ]

        proposed_edits = state.get("proposed_edits")
        if resume_markdown and proposed_edits:
            job_id_str = str(current_job.get("job_id", "unknown"))
            edited_pdf_path = str(get_edited_resume_path(run_id=run_id, job_id=job_id_str))

            try:
                llm = get_llm(temperature=0.2, max_output_tokens=2048)
                prompt = get_resume_edit_prompt(resume_markdown, proposed_edits)

                llm_result = await llm.ainvoke(prompt)
                edited_markdown = getattr(llm_result, "content", str(llm_result))

                await markdown_to_pdf(edited_markdown, edited_pdf_path)

                if os.path.exists(edited_pdf_path):
                    updates["resume_path"] = edited_pdf_path
                    updates["resume_edited"] = True
                else:
                    raise RuntimeError("PDF file was not created on disk")
            except Exception as e:
                updates["errors"] = list(updates.get("errors") or state.get("errors") or []) + [
                    f"Resume editing/PDF generation failed for {current_job.get('title', 'unknown job')}: {redact_string(str(e))}"
                ]
                updates["resume_edited"] = False
        else:
            updates["resume_edited"] = False
    else:
        updates["resume_edited"] = False

    return updates
