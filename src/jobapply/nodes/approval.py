"""Approval node - Telegram human-in-the-loop for urgent edits."""

import os
from uuid import uuid4
from jobapply.state import JobApplyState
from jobapply.utils.telegram import TelegramClient
from jobapply.settings import get_settings
from jobapply.utils.llm import get_llm
from jobapply.utils.prompts import get_resume_edit_prompt
from jobapply.utils.pdf import markdown_to_pdf


def _data_path(*parts: str) -> str:
    """Build an absolute path under src/jobapply/data."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", *parts)


async def approval_node(state: JobApplyState) -> dict:
    """Ask user via Telegram for approval of urgent resume edits.

    This is the rare path, only triggered when edits_urgent=True.

    Args:
        state: Current graph state.

    Returns:
        State updates dict.
    """
    settings = get_settings()
    telegram = TelegramClient()
    
    nonce = str(uuid4())[:8]
    
    current_job = state["current_job"]
    qual_result = state["qualification_result"] or {}
    
    message = f"""🔴 **Urgent Edit Recommended** `[{nonce}]`
 
**Job:** {current_job['title']} at {current_job['company']}
**Score:** {qual_result.get('score', 0.0):.2f}

**Why:** {state.get('edit_reasoning', 'No reasoning provided')}

**Proposed changes:**
{state.get('proposed_edits', 'No changes proposed')}

Reply with `{nonce}` followed by:
✅ **approve** — use edited resume
📄 **base** — use original resume as-is
❌ **skip** — skip this job
"""
    
    prompt_message_id = await telegram.send_message(message)
    
    reply = await telegram.wait_for_correlated_reply(
        nonce=nonce,
        timeout=settings.approval_timeout_seconds,
        reply_to_message_id=prompt_message_id,
    )
    
    # Parse reply
    if reply is None:
        # Timeout - default to safe option
        approval_status = "use_base"
    elif "approve" in reply.lower() or "✅" in reply:
        approval_status = "approved"
    elif "base" in reply.lower() or "📄" in reply:
        approval_status = "use_base"
    elif "skip" in reply.lower() or "❌" in reply:
        approval_status = "skip"
    else:
        # Unrecognized - default to safe
        approval_status = "use_base"
    
    updates = {
        "approval_status": approval_status,
        "approval_nonce": nonce,
    }

    if approval_status == "skip":
        skipped_jobs = list(state.get("skipped_jobs") or [])
        outcomes = list(state.get("application_outcomes") or [])
        
        outcome = {
            "job_id": current_job["job_id"],
            "title": current_job["title"],
            "company": current_job["company"],
            "url": current_job.get("url"),
            "score": qual_result.get("score", 0.0),
            "status": "skipped",
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
        # Generate the edited resume
        resume_markdown = ""
        try:
            with open(_data_path("resume.md"), "r", encoding="utf-8") as f:
                resume_markdown = f.read()
        except Exception as e:
            updates["errors"] = list(state.get("errors") or []) + [
                f"Failed to load resume.md for editing: {str(e)}"
            ]
        
        proposed_edits = state.get("proposed_edits")
        if resume_markdown and proposed_edits:
            output_dir = os.path.join("outputs", state["run_id"])
            os.makedirs(output_dir, exist_ok=True)
            edited_pdf_path = os.path.join(output_dir, f"edited_resume_{current_job['job_id']}.pdf")
            
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
                updates["errors"] = list(state.get("errors") or []) + [
                    f"Resume editing/PDF generation failed for {current_job['title']}: {str(e)}"
                ]
                updates["resume_edited"] = False
        else:
            updates["resume_edited"] = False
    else:
        # use_base
        updates["resume_edited"] = False

    return updates
