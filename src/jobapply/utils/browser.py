"""Browser automation utilities with Playwright CDP."""

import os
import random
import subprocess
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright, Browser, BrowserContext
from langsmith.run_helpers import trace
from jobapply.settings import get_settings
from jobapply.utils.tracing import get_browser_metadata


def edge_debug_ports(preferred_port: int, active_port_path: Path | None = None) -> list[int]:
    """Return preferred and Edge-discovered CDP ports without reading profile data."""
    ports = [preferred_port]
    if active_port_path is None:
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            active_port_path = Path(local_app_data) / "Microsoft" / "Edge" / "User Data" / "DevToolsActivePort"
    if active_port_path and active_port_path.is_file():
        try:
            discovered = int(active_port_path.read_text(encoding="utf-8").splitlines()[0])
            if 1 <= discovered <= 65535 and discovered not in ports:
                ports.insert(0, discovered)
        except (OSError, ValueError, IndexError):
            pass
    return ports


def edge_profile_path(configured_path: str) -> Path:
    """Resolve and validate the dedicated automation profile path."""
    profile_path = Path(os.path.expandvars(configured_path)).resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        default_data = (Path(local_app_data) / "Microsoft" / "Edge" / "User Data").resolve()
        if profile_path == default_data or default_data in profile_path.parents:
            raise RuntimeError(
                "JOBAPPLY_EDGE_USER_DATA_DIR must be separate from Edge's normal User Data directory"
            )
    return profile_path


def launch_edge_for_jobapply(settings) -> subprocess.Popen:
    """Launch visible Microsoft Edge with a persistent, dedicated profile."""
    executable = Path(settings.edge_executable_path)
    if not executable.is_file():
        raise RuntimeError(f"Microsoft Edge executable not found: {executable}")
    profile_path = edge_profile_path(settings.edge_user_data_dir)
    profile_path.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    return subprocess.Popen(
        [
            str(executable),
            f"--remote-debugging-port={settings.edge_debug_port}",
            f"--user-data-dir={profile_path}",
            "--profile-directory=Default",
            "--no-first-run",
            "--no-default-browser-check",
            "https://www.linkedin.com/feed/",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )


async def connect_to_edge(pw, ports: list[int]):
    """Try each candidate CDP port and return browser, port, and last error."""
    last_error = None
    for port in ports:
        try:
            browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            return browser, port, None
        except Exception as exc:
            last_error = exc
    return None, None, last_error


@asynccontextmanager
async def managed_browser():
    """Async context manager for Playwright CDP connection.

    Handles:
    - Connection to Edge via CDP
    - Empty contexts guard (Edge must have at least one window)
    - Clean shutdown of Playwright on exit or error
    - Screenshot-on-error capture

    Yields:
        tuple: (browser, context) - Playwright browser and context instances.

    Raises:
        RuntimeError: If no browser contexts are found.
    """
    settings = get_settings()
    pw = await async_playwright().start()
    browser = None
    
    try:
        # Trace CDP connection attempt
        async with trace(
            "edge_cdp_connect",
            run_type="tool",
            metadata={"cdp_port": settings.edge_debug_port}
        ) as run_tree:
            try:
                browser, connected_port, last_error = await connect_to_edge(
                    pw,
                    edge_debug_ports(settings.edge_debug_port),
                )
                if browser is None and settings.edge_auto_launch:
                    launch_edge_for_jobapply(settings)
                    for _ in range(20):
                        await asyncio.sleep(0.5)
                        browser, connected_port, last_error = await connect_to_edge(
                            pw,
                            [settings.edge_debug_port],
                        )
                        if browser is not None:
                            break
                if browser is None or connected_port is None:
                    raise RuntimeError(
                        "Cannot connect to the dedicated JobApply Edge profile. "
                        "Check JOBAPPLY_EDGE_EXECUTABLE_PATH, JOBAPPLY_EDGE_USER_DATA_DIR, "
                        "and JOBAPPLY_EDGE_DEBUG_PORT. "
                        f"Last connection error: {last_error}"
                    )
                contexts = browser.contexts
                
                if not contexts:
                    error_msg = (
                        "No browser contexts found. Ensure Edge has at least "
                        "one window open and is launched with "
                        f"--remote-debugging-port={settings.edge_debug_port}"
                    )
                    # Update trace with error metadata
                    if run_tree:
                        run_tree.metadata.update(get_browser_metadata(
                            port=connected_port,
                            connected=False,
                            error=error_msg
                        ))
                    raise RuntimeError(error_msg)
                
                context = contexts[0]
                
                # Update trace with success metadata
                if run_tree:
                    run_tree.metadata.update(get_browser_metadata(
                        port=connected_port,
                        connected=True,
                        contexts_count=len(contexts)
                    ))
                
                yield browser, context
                
            except Exception as e:
                # Update trace with error metadata
                if run_tree:
                    run_tree.metadata.update(get_browser_metadata(
                        port=settings.edge_debug_port,
                        connected=False,
                        error=str(e)
                    ))
                raise
    finally:
        if browser:
            await browser.close()
        await pw.stop()


async def take_error_screenshot(page, run_id: str, label: str) -> str:
    """Save screenshot on error for debugging.

    Args:
        page: Playwright page instance.
        run_id: Run ID for organizing screenshots.
        label: Label for the screenshot file.

    Returns:
        Path to the saved screenshot.
    """
    path = f"outputs/{run_id}/errors/{label}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    await page.screenshot(path=path)
    return path


def get_randomized_delay() -> float:
    """Returns delay in seconds with ±jitter.

    Uses action_delay_ms and delay_jitter_percent from settings.

    Returns:
        Delay duration in seconds with applied jitter.
    """
    settings = get_settings()
    base = settings.action_delay_ms / 1000
    jitter = base * (settings.delay_jitter_percent / 100)
    return base + random.uniform(-jitter, jitter)
