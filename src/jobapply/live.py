"""One-command live runner: preflight, Edge sign-in, then full workflow."""

import argparse
import asyncio
import os
from pathlib import Path
from urllib.parse import urlparse

from jobapply.main import run
from jobapply.settings import get_settings
from jobapply.utils.account_safety import (
    classify_url,
    inspect_page_account_safety,
)
from jobapply.utils.browser import get_linkedin_page, managed_browser
from jobapply.utils.dedup import DeduplicationStore
from jobapply.utils.llm import close_llm_client
from jobapply.utils.power import keep_system_awake
from jobapply.utils.telegram import TelegramClient

SIGNIN_PATH_MARKERS = (
    "/login",
    "/uas/",
    "/authwall",
    "/checkpoint",
    "/signup",
)


def linkedin_url_is_signed_in(url: str) -> bool:
    """Return True for an authenticated LinkedIn page URL."""
    if classify_url(url) is not None:
        return False
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    path = (parsed.path or "/").lower()
    return (
        (hostname == "linkedin.com" or hostname.endswith(".linkedin.com"))
        and not any(marker in path for marker in SIGNIN_PATH_MARKERS)
        and path not in ("/", "")
    )


def validate_local_configuration() -> None:
    """Fail early for missing model credentials or required application files."""
    settings = get_settings()
    key_name = "OPENROUTER_API_KEY" if settings.llm_provider == "openrouter" else "GOOGLE_API_KEY"
    if not os.getenv(key_name):
        raise RuntimeError(f"{key_name} is missing from .env")
    required_files = (
        Path(settings.resolve_data_path("profile.yaml")),
        Path(settings.resolve_data_path("resume.md")),
        Path(settings.resolve_data_path("resume.pdf")),
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required application files are missing: {', '.join(missing)}")


async def wait_for_linkedin_signin() -> None:
    """Open visible Edge and wait for the dedicated profile to be signed in."""
    settings = get_settings()
    async with managed_browser() as (_, context):
        page = await get_linkedin_page(context)
        try:
            print("Opening Microsoft Edge and checking LinkedIn sign-in...")
            response = await page.goto(
                f"{settings.linkedin_base_url.rstrip('/')}/feed/",
                wait_until="domcontentloaded",
                timeout=settings.linkedin_navigation_timeout_ms,
            )
            http_status = response.status if response else None
            await page.bring_to_front()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + settings.linkedin_signin_timeout_seconds
            announced_wait = False
            last_barrier_announced = None
            while not linkedin_url_is_signed_in(page.url):
                if loop.time() >= deadline:
                    raise TimeoutError(
                        "LinkedIn sign-in timed out. Sign in to the visible JobApply Edge "
                        "window, then run the command again."
                    )

                # Check for explicit security challenges / verification barriers
                safety = await inspect_page_account_safety(
                    page, stage="preflight_signin", http_status=http_status, fail_closed=False
                )
                if safety.detected and safety.barrier_type != last_barrier_announced:
                    btype = safety.barrier_type.value if safety.barrier_type else "barrier"
                    print(
                        f"🛑 [ACCOUNT SAFETY] LinkedIn requires manual intervention ({btype}): "
                        f"{safety.reason}. Resolve in the visible Edge window..."
                    )
                    last_barrier_announced = safety.barrier_type

                if not announced_wait and not safety.detected:
                    print(
                        "LinkedIn is not signed in. Complete sign-in in the visible Edge window..."
                    )
                    announced_wait = True
                await asyncio.sleep(2)
            print("LinkedIn sign-in: OK")
        finally:
            # Keep the single working tab available to the workflow.
            pass


async def run_preflight() -> None:
    """Validate local files, MongoDB, and the signed-in Edge profile."""
    validate_local_configuration()
    store = DeduplicationStore()
    await asyncio.wait_for(store.client.admin.command("ping"), timeout=10)
    print("MongoDB: OK")
    await wait_for_linkedin_signin()


@keep_system_awake
async def run_live(
    *,
    preflight_only: bool = False,
    max_jobs: int | None = None,
    max_applications: int | None = None,
    close_resources: bool = True,
) -> None:
    """Run preflight and, unless requested otherwise, the real submission workflow."""
    settings = get_settings()
    try:
        await run_preflight()
        if preflight_only:
            print("Preflight complete. No applications were processed.")
            return

        session_cap = (
            max_applications
            if max_applications is not None
            else settings.effective_max_applications
        )
        jobs_scope = "configured limit" if max_jobs is None else str(max_jobs)
        recency_scope = (
            "all dates"
            if settings.search_recency_days is None
            else f"last {settings.search_recency_days} day(s)"
        )
        print("\nLIVE SUBMISSION SCOPE")
        print(f"Search queries: {', '.join(settings.search_queries_list)}")
        print(f"Location: {settings.search_location}")
        print(f"Recency: {recency_scope}")
        print(f"Jobs to evaluate: {jobs_scope}")
        print(f"Maximum applications to submit: {session_cap}")

        await TelegramClient().send_message(
            "🚀 JOB APPLY SESSION STARTED\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "\n"
            "MODE\n"
            "• Live application submission\n"
            "• Qualified jobs proceed automatically\n"
            "• Base resume used without approval prompts\n"
            "\n"
            "SEARCH SCOPE\n"
            f"• Queries: {', '.join(settings.search_queries_list)}\n"
            f"• Location: {settings.search_location}\n"
            f"• Pages per query: {settings.pages_per_query}\n"
            f"• Submission limit: {session_cap}\n"
            "\n"
            "WHAT TO EXPECT\n"
            "• A receipt after each application result\n"
            "• A question only when the form cannot be answered safely\n"
            "• A final summary when the session ends"
        )
        print("Starting LIVE workflow. Confirmed applications may be submitted.")
        await run(
            dry_run=False,
            max_jobs=max_jobs,
            max_applications=max_applications,
            close_resources=close_resources,
        )
    finally:
        if close_resources:
            await DeduplicationStore.close()


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def main() -> None:
    """CLI entry point for ``jobapply-live``."""
    parser = argparse.ArgumentParser(
        description="Open Edge, verify LinkedIn, and run the live JobApply workflow",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify MongoDB and LinkedIn sign-in without processing jobs",
    )
    parser.add_argument(
        "--max-jobs",
        type=_non_negative_int,
        help="optional maximum jobs to evaluate for this run",
    )
    parser.add_argument(
        "--max-applications",
        type=_non_negative_int,
        help="optional submission cap for this run",
    )
    parser.add_argument(
        "--telegram-control",
        action="store_true",
        help="keep a Telegram controller online for /report, /stop, /run, and /status",
    )
    args = parser.parse_args()

    async def _entrypoint() -> None:
        if args.telegram_control and not args.preflight_only:
            from jobapply.utils.telegram_control import TelegramControlService

            async def controlled_run() -> None:
                await run_live(
                    max_jobs=args.max_jobs,
                    max_applications=args.max_applications,
                    close_resources=False,
                )

            try:
                await TelegramControlService(controlled_run).serve(start_immediately=True)
            finally:
                try:
                    await close_llm_client()
                finally:
                    await DeduplicationStore.close()
            return
        await run_live(
            preflight_only=args.preflight_only,
            max_jobs=args.max_jobs,
            max_applications=args.max_applications,
        )

    asyncio.run(_entrypoint())


if __name__ == "__main__":
    main()
