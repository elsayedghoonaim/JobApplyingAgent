"""Browser automation utilities with dedicated Edge profile and CDP isolation."""

import asyncio
import json
import math
import os
import random
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import psutil
from langsmith.run_helpers import trace
from playwright.async_api import async_playwright

from jobapply.settings import get_settings
from jobapply.utils.paths import get_error_screenshot_path
from jobapply.utils.redaction import redact_string
from jobapply.utils.tracing import get_browser_metadata

OWNER_MARKER_FILENAME = ".jobapply_browser_owner.json"
MARKER_SCHEMA_VERSION = 1


def edge_debug_ports(preferred_port: int, active_port_path: Optional[Path] = None) -> list[int]:
    """Return only the configured dedicated CDP port.

    Normal Edge DevToolsActivePort discovery is intentionally disabled to isolate
    JobApply to its dedicated profile and port.
    """
    if not (1 <= preferred_port <= 65535):
        raise ValueError(f"Invalid debug port {preferred_port}: must be between 1 and 65535")
    return [preferred_port]


def edge_profile_path(configured_path: str) -> Path:
    """Resolve and validate the dedicated automation profile path.

    Ensures the automation profile is strictly isolated from default user Edge data
    (neither equal, ancestor, nor descendant).
    """
    profile_path = Path(os.path.expandvars(configured_path)).resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        default_data = (Path(local_app_data) / "Microsoft" / "Edge" / "User Data").resolve()
        if (
            profile_path == default_data
            or default_data in profile_path.parents
            or profile_path in default_data.parents
        ):
            raise RuntimeError(
                "JOBAPPLY_EDGE_USER_DATA_DIR must be strictly separate from Edge's normal User Data directory"
            )
    return profile_path


def get_owner_marker_path(profile_path: Path) -> Path:
    """Get the path to the JobApply browser ownership marker."""
    return profile_path / OWNER_MARKER_FILENAME


def get_port_listener_pids(port: int) -> Optional[set[int]]:
    """Return the set of process IDs currently listening on the specified port.

    Returns None if socket enumeration fails due to permissions or OS errors (fail-closed).
    """
    listener_pids = set()
    try:
        conns = psutil.net_connections(kind="inet")
        for conn in conns:
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                if conn.pid:
                    listener_pids.add(conn.pid)
        return listener_pids
    except Exception:
        try:
            for proc in psutil.process_iter(["pid"]):
                try:
                    for conn in proc.net_connections(kind="inet"):
                        if (
                            conn.laddr
                            and conn.laddr.port == port
                            and conn.status == psutil.CONN_LISTEN
                        ):
                            listener_pids.add(proc.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except (psutil.AccessDenied, Exception):
                    return None
            return listener_pids
        except Exception:
            return None


def is_port_free(port: int) -> bool:
    """Check if the given port is provably free of any listening process.

    Returns True ONLY if enumeration succeeded with zero listeners; returns False on enumeration failure.
    """
    listeners = get_port_listener_pids(port)
    if listeners is None:
        return False
    return len(listeners) == 0


def write_ownership_marker(
    profile_path: Path,
    pid: int,
    create_time: float,
    port: int,
    executable_path: str,
    cmdline: list[str],
) -> Path:
    """Write comprehensive browser ownership marker to the dedicated profile directory."""
    profile_path.mkdir(parents=True, exist_ok=True)
    marker_path = get_owner_marker_path(profile_path)
    marker_data = {
        "version": MARKER_SCHEMA_VERSION,
        "pid": pid,
        "create_time": create_time,
        "port": port,
        "profile_path": str(profile_path.resolve()),
        "executable_path": str(Path(executable_path).resolve()),
        "cmdline": cmdline,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker_data, indent=2), encoding="utf-8")
    return marker_path


def read_ownership_marker(profile_path: Path) -> Optional[dict[str, Any]]:
    """Read and validate the ownership marker schema and field types."""
    marker_path = get_owner_marker_path(profile_path)
    if not marker_path.is_file():
        return None
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        if data.get("version") != MARKER_SCHEMA_VERSION:
            return None
        if not isinstance(data.get("pid"), int) or data["pid"] <= 0:
            return None
        if not isinstance(data.get("port"), int) or not (1 <= data["port"] <= 65535):
            return None
        ctime = data.get("create_time")
        if not isinstance(ctime, (int, float)) or not math.isfinite(ctime) or ctime <= 0:
            return None
        if not isinstance(data.get("profile_path"), str) or not data["profile_path"].strip():
            return None
        if not isinstance(data.get("executable_path"), str) or not data["executable_path"].strip():
            return None
        cmdline = data.get("cmdline")
        if (
            not isinstance(cmdline, list)
            or not cmdline
            or not all(isinstance(a, str) for a in cmdline)
        ):
            return None
        return data
    except Exception:
        return None


def verify_browser_ownership(profile_path: Path, expected_port: int, settings) -> bool:
    """Verify that an Edge instance is owned by JobApply with matching profile/port/pid/ctime/cmdline/listener.

    Cleans up stale marker ONLY if the owning process is provably dead, the marker was structurally valid,
    and the port is provably free.
    Fails closed on any inspection failure, PID reuse, argument mismatch, or unrelated listener.
    """
    marker = read_ownership_marker(profile_path)
    if marker is None:
        return False

    if marker.get("port") != expected_port:
        return False

    # Verify marker recorded profile path matches canonical profile path
    try:
        marker_profile = Path(marker.get("profile_path", "")).resolve()
        if marker_profile.as_posix().lower() != profile_path.resolve().as_posix().lower():
            return False
    except Exception:
        return False

    # Verify marker recorded executable path matches canonical configured executable
    try:
        expected_exe = Path(settings.edge_executable_path).resolve()
        marker_exe = Path(marker.get("executable_path", "")).resolve()
        if marker_exe.as_posix().lower() != expected_exe.as_posix().lower():
            return False
    except Exception:
        return False

    # Verify marker recorded cmdline contains required exact arguments
    marker_cmdline = marker.get("cmdline", [])
    marker_has_port = any(
        arg == f"--remote-debugging-port={expected_port}"
        or (
            arg.startswith("--remote-debugging-port=")
            and arg.split("=", 1)[1].strip() == str(expected_port)
        )
        for arg in marker_cmdline
    )
    marker_has_profile = any(
        arg.startswith("--user-data-dir=")
        and Path(arg.split("=", 1)[1].strip()).resolve().as_posix().lower()
        == profile_path.resolve().as_posix().lower()
        for arg in marker_cmdline
    )
    if not (marker_has_port and marker_has_profile):
        return False

    pid = marker.get("pid", 0)
    marker_ctime = marker.get("create_time", 0.0)

    try:
        proc = psutil.Process(pid)
        if not proc.is_running():
            raise psutil.NoSuchProcess(pid)

        # 1. Verify process creation time tightly (1 millisecond tolerance for float precision)
        proc_ctime = proc.create_time()
        if not math.isfinite(proc_ctime) or abs(proc_ctime - float(marker_ctime)) > 0.001:
            return False

        # 2. Verify live executable path matches canonical configured Edge executable
        proc_exe = Path(proc.exe()).resolve()
        if proc_exe.as_posix().lower() != expected_exe.as_posix().lower():
            return False

        # 3. Verify live launch arguments match exact parameters
        cmdline = proc.cmdline()
        has_port_arg = False
        has_profile_arg = False

        for arg in cmdline:
            if arg == f"--remote-debugging-port={expected_port}":
                has_port_arg = True
            elif arg.startswith("--remote-debugging-port="):
                val = arg.split("=", 1)[1].strip()
                if val == str(expected_port):
                    has_port_arg = True

            if arg.startswith("--user-data-dir="):
                val = arg.split("=", 1)[1].strip()
                try:
                    if (
                        Path(val).resolve().as_posix().lower()
                        == profile_path.resolve().as_posix().lower()
                    ):
                        has_profile_arg = True
                except Exception:
                    pass

        if not (has_port_arg and has_profile_arg):
            return False

        # 4. Verify listener set is non-empty and wholly belongs to the owned PID or verified children
        listeners = get_port_listener_pids(expected_port)
        if listeners is None or len(listeners) == 0:
            return False

        owned_tree = {proc.pid} | {c.pid for c in proc.children(recursive=True)}
        if not listeners.issubset(owned_tree):
            return False

        return True

    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        # Process is dead; if port is provably free, remove stale marker
        if is_port_free(expected_port):
            try:
                get_owner_marker_path(profile_path).unlink(missing_ok=True)
            except Exception:
                pass
        return False
    except Exception:
        # Inspection failed due to AccessDenied or OS error -> fail closed
        return False


def launch_edge_for_jobapply(settings) -> subprocess.Popen:
    """Launch visible Microsoft Edge with a persistent dedicated profile and write ownership marker.

    Fails closed if the configured port is not provably free or an unhandled marker exists.
    """
    profile_path = edge_profile_path(settings.edge_user_data_dir)
    marker_path = get_owner_marker_path(profile_path)

    # Preflight check: if an existing marker file is present, check if it's a dead marker that can be cleaned
    if marker_path.is_file():
        # Running verify_browser_ownership removes valid dead markers when port is free
        if verify_browser_ownership(profile_path, settings.edge_debug_port, settings):
            raise RuntimeError(
                f"Cannot launch dedicated Edge: browser is already running and owned on port {settings.edge_debug_port}."
            )
        # If marker is still present, fail closed (do not overwrite malformed or occupied markers)
        if marker_path.is_file():
            raise RuntimeError(
                f"Cannot launch dedicated Edge: marker file exists at '{marker_path}' and cannot be safely cleared."
            )

    if not is_port_free(settings.edge_debug_port):
        raise RuntimeError(
            f"Cannot launch dedicated Edge: configured debug port {settings.edge_debug_port} is not provably free."
        )

    executable = Path(settings.edge_executable_path)
    if not executable.is_file():
        raise RuntimeError(f"Microsoft Edge executable not found: {executable}")

    profile_path.mkdir(parents=True, exist_ok=True)

    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    cmd = [
        str(executable),
        f"--remote-debugging-port={settings.edge_debug_port}",
        f"--user-data-dir={profile_path.resolve()}",
        "--profile-directory=Default",
        "--no-first-run",
        "--no-default-browser-check",
        "https://www.linkedin.com/feed/",
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )

    try:
        ps_proc = psutil.Process(proc.pid)
        create_time = ps_proc.create_time()
    except Exception:
        create_time = datetime.now(timezone.utc).timestamp()

    write_ownership_marker(
        profile_path=profile_path,
        pid=proc.pid,
        create_time=create_time,
        port=settings.edge_debug_port,
        executable_path=str(executable),
        cmdline=cmd,
    )
    return proc


async def connect_to_edge(pw, port: int, profile_path: Optional[Path] = None, settings=None):
    """Connect strictly to the dedicated Edge CDP port with mandatory ownership verification.

    Never calls connect_over_cdp without successful ownership verification.
    """
    if profile_path is None or settings is None:
        return (
            None,
            port,
            RuntimeError(
                "Ownership verification context (profile_path and settings) is required to connect to Edge."
            ),
        )

    if not verify_browser_ownership(profile_path, port, settings):
        return (
            None,
            port,
            RuntimeError(
                f"Port {port} is active or marker present, but ownership verification failed for profile {profile_path}"
            ),
        )

    try:
        browser = await pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
        return browser, port, None
    except Exception as exc:
        return None, port, exc


@asynccontextmanager
async def managed_browser():
    """Async context manager for Playwright CDP connection to dedicated Edge instance."""
    settings = get_settings()
    profile_path = edge_profile_path(settings.edge_user_data_dir)
    pw = await async_playwright().start()
    browser = None

    try:
        async with trace(
            "edge_cdp_connect", run_type="tool", metadata={"cdp_port": settings.edge_debug_port}
        ) as run_tree:
            try:
                # Check if an owned instance is already alive and verified
                if verify_browser_ownership(profile_path, settings.edge_debug_port, settings):
                    browser, connected_port, last_error = await connect_to_edge(
                        pw,
                        settings.edge_debug_port,
                        profile_path,
                        settings,
                    )
                else:
                    browser, connected_port, last_error = None, settings.edge_debug_port, None

                # Auto-launch if not connected and auto-launch enabled
                if browser is None and settings.edge_auto_launch:
                    launch_edge_for_jobapply(settings)
                    for _ in range(20):
                        await asyncio.sleep(0.5)
                        browser, connected_port, last_error = await connect_to_edge(
                            pw,
                            settings.edge_debug_port,
                            profile_path,
                            settings,
                        )
                        if browser is not None:
                            break

                if browser is None or connected_port is None:
                    error_detail = redact_string(str(last_error)) if last_error else "unknown error"
                    raise RuntimeError(
                        "Cannot connect to the dedicated JobApply Edge profile. "
                        "Check JOBAPPLY_EDGE_EXECUTABLE_PATH, JOBAPPLY_EDGE_USER_DATA_DIR, "
                        "and JOBAPPLY_EDGE_DEBUG_PORT. "
                        f"Last connection error: {error_detail}"
                    )

                contexts = browser.contexts
                if not contexts:
                    error_msg = (
                        "No browser contexts found. Ensure Edge has at least "
                        "one window open and is launched with "
                        f"--remote-debugging-port={settings.edge_debug_port}"
                    )
                    if run_tree:
                        run_tree.metadata.update(
                            get_browser_metadata(
                                port=connected_port, connected=False, error=error_msg
                            )
                        )
                    raise RuntimeError(error_msg)

                context = contexts[0]
                if run_tree:
                    run_tree.metadata.update(
                        get_browser_metadata(
                            port=connected_port, connected=True, contexts_count=len(contexts)
                        )
                    )

                yield browser, context

            except Exception as e:
                if run_tree:
                    run_tree.metadata.update(
                        get_browser_metadata(
                            port=settings.edge_debug_port,
                            connected=False,
                            error=redact_string(str(e)),
                        )
                    )
                raise
    finally:
        if browser:
            try:
                await browser.close()
            except Exception:
                pass
        await pw.stop()


async def take_error_screenshot(page, run_id: str, label: str) -> str:
    """Save screenshot on error for debugging to a safe, validated path.

    Args:
        page: Playwright page instance.
        run_id: Run ID for organizing screenshots.
        label: Label for the screenshot file.

    Returns:
        Safe path to the saved screenshot.
    """
    path = get_error_screenshot_path(run_id=run_id, label=label)
    await page.screenshot(path=str(path))
    return str(path)


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
