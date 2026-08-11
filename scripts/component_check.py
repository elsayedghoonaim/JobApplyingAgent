"""Run bounded live checks for the non-browser workflow components."""

import argparse
import asyncio
from pathlib import Path

import yaml

from jobapply.models.job import QualificationResult
from jobapply.nodes.generation import generation_node
from jobapply.nodes.execution import extract_answer_from_reply
from jobapply.nodes.search import parse_job_description
from jobapply.settings import get_settings
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.json_output import extract_json_object
from jobapply.utils.llm import close_llm_client, get_llm
from jobapply.utils.pdf import markdown_to_pdf
from jobapply.utils.prompts import get_qualification_prompt
from jobapply.utils.telegram import TelegramClient


async def run(send_telegram: bool) -> None:
    settings = get_settings()
    assert settings.llm_model == "gemma-4-31b-it"

    store = DeduplicationStore()
    await store.client.admin.command("ping")
    print("MongoDB: OK")

    parsed = await parse_job_description(
        "Remote Python engineer. Responsibilities: build production ML APIs. "
        "Requirements: Python and API engineering."
    )
    assert parsed.get("requirements")
    print("Gemma job parser: OK")

    profile = yaml.safe_load(
        Path("src/jobapply/data/profile.yaml").read_text(encoding="utf-8")
    )
    sample_job = {
        "title": "Machine Learning Engineer",
        "company": "Workflow Test",
        "location": "Remote",
        "description": "Build Python machine-learning services and APIs.",
    }
    llm = get_llm(
        temperature=0.1,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_json_schema=QualificationResult.model_json_schema(),
    )
    qualification_reply = await llm.ainvoke(
        get_qualification_prompt(yaml.safe_dump(profile), sample_job)
    )
    qualification = QualificationResult(
        **extract_json_object(qualification_reply.content)
    )
    print(f"Gemma qualification JSON: OK (score={qualification.score:.2f})")

    extracted_answer = await extract_answer_from_reply(
        "How many years of Generative AI experience do you have?",
        "I have about three years of hands-on experience.",
    )
    assert extracted_answer == "3"
    print("Gemma Telegram answer extraction: OK")

    state = {
        "run_id": "component-check",
        "current_job": {
            "job_id": "component-1",
            "url": "https://example.test/job",
            **sample_job,
        },
        "qualification_result": {
            "qualified": True,
            "score": 0.7,
            "key_matches": ["Python"],
            "gaps": [],
        },
        "errors": [],
    }
    generated = await generation_node(state)
    assert generated.get("cover_letter_text")
    assert generated.get("cover_letter_path")
    print("Cover-letter generation node: OK")

    pdf_path = Path("C:/tmp/jobapply_component_check.pdf")
    await markdown_to_pdf("# Test Resume\n\n## Skills\n\n- Python", str(pdf_path))
    assert pdf_path.exists() and pdf_path.stat().st_size > 1000
    print("PDF generation: OK")

    if send_telegram:
        await TelegramClient().send_message(
            "JobApply component check passed: Gemma, MongoDB, and PDF workflow are working."
        )
        print("Telegram delivery: OK")


async def main(send_telegram: bool) -> None:
    try:
        await run(send_telegram)
    finally:
        await close_llm_client()
        await DeduplicationStore.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="also send one labeled Telegram test message",
    )
    asyncio.run(main(parser.parse_args().telegram))
