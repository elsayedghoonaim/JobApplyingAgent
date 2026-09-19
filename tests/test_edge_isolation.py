"""Tests for dedicated Edge profile and fail-closed CDP isolation."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from jobapply.settings import get_settings
from jobapply.utils.browser import (
    edge_debug_ports,
    edge_profile_path,
    get_linkedin_page,
    get_owner_marker_path,
    get_port_listener_pids,
    is_port_free,
    launch_edge_for_jobapply,
    read_ownership_marker,
    verify_browser_ownership,
    write_ownership_marker,
)


@pytest.mark.asyncio
async def test_linkedin_page_reuses_existing_page_and_enables_dark_mode():
    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.emulate_media = AsyncMock()
    session = MagicMock()
    session.send = AsyncMock()
    session.detach = AsyncMock()
    context = MagicMock()
    context.pages = [page]
    context.new_page = AsyncMock()
    context.new_cdp_session = AsyncMock(return_value=session)

    selected = await get_linkedin_page(context)

    assert selected is page
    context.new_page.assert_not_awaited()
    page.emulate_media.assert_awaited_once_with(color_scheme="dark")
    session.send.assert_awaited_once_with("Emulation.setAutoDarkModeOverride", {"enabled": True})
    session.detach.assert_awaited_once()


@pytest.mark.asyncio
async def test_linkedin_page_closes_stale_automation_tabs_only():
    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.emulate_media = AsyncMock()
    duplicate = MagicMock()
    duplicate.url = "https://www.linkedin.com/jobs/"
    duplicate.close = AsyncMock()
    blank = MagicMock()
    blank.url = "about:blank"
    blank.close = AsyncMock()
    unrelated = MagicMock()
    unrelated.url = "https://example.com/"
    unrelated.close = AsyncMock()
    context = MagicMock()
    context.pages = [page, duplicate, blank, unrelated]
    context.new_page = AsyncMock()
    context.new_cdp_session = AsyncMock(side_effect=RuntimeError("unsupported"))

    selected = await get_linkedin_page(context)

    assert selected is page
    duplicate.close.assert_awaited_once()
    blank.close.assert_awaited_once()
    unrelated.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_linkedin_page_reuses_blank_startup_page():
    page = MagicMock()
    page.url = "about:blank"
    page.emulate_media = AsyncMock()
    context = MagicMock()
    context.pages = [page]
    context.new_page = AsyncMock()
    context.new_cdp_session = AsyncMock(side_effect=RuntimeError("unsupported"))

    selected = await get_linkedin_page(context)

    assert selected is page
    context.new_page.assert_not_awaited()


@pytest.mark.asyncio
async def test_linkedin_page_closes_extra_blank_startup_tab():
    page = MagicMock()
    page.url = "about:blank"
    page.emulate_media = AsyncMock()
    extra = MagicMock()
    extra.url = "edge://newtab/"
    extra.close = AsyncMock()
    context = MagicMock()
    context.pages = [page, extra]
    context.new_page = AsyncMock()
    context.new_cdp_session = AsyncMock(side_effect=RuntimeError("unsupported"))

    selected = await get_linkedin_page(context)

    assert selected is page
    extra.close.assert_awaited_once()


def test_edge_debug_ports_ignores_normal_devtools_marker(tmp_path):
    normal_marker = tmp_path / "DevToolsActivePort"
    normal_marker.write_text("54321\n/devtools/browser/test", encoding="utf-8")

    ports = edge_debug_ports(9222, normal_marker)
    # Must only contain the configured dedicated port, never dynamic normal ports
    assert ports == [9222]


def test_edge_debug_ports_rejects_invalid_range():
    with pytest.raises(ValueError):
        edge_debug_ports(0)
    with pytest.raises(ValueError):
        edge_debug_ports(70000)


def test_edge_profile_path_rejects_normal_profile_ancestor_and_descendant(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    normal_dir = tmp_path / "Microsoft" / "Edge" / "User Data"
    normal_dir.mkdir(parents=True, exist_ok=True)

    # 1. Equals normal user data
    with pytest.raises(RuntimeError, match="JOBAPPLY_EDGE_USER_DATA_DIR must be strictly separate"):
        edge_profile_path(str(normal_dir))

    # 2. Descendant of normal user data
    with pytest.raises(RuntimeError, match="JOBAPPLY_EDGE_USER_DATA_DIR must be strictly separate"):
        edge_profile_path(str(normal_dir / "Default"))

    # 3. Ancestor of normal user data
    with pytest.raises(RuntimeError, match="JOBAPPLY_EDGE_USER_DATA_DIR must be strictly separate"):
        edge_profile_path(str(tmp_path / "Microsoft" / "Edge"))


def test_ownership_marker_write_and_read(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    marker_path = write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )
    assert marker_path.is_file()

    data = read_ownership_marker(profile_dir)
    assert data is not None
    assert data["version"] == 1
    assert data["pid"] == 12345
    assert data["port"] == 9222
    assert data["create_time"] == 1700000000.0
    assert data["profile_path"] == str(profile_dir.resolve())
    assert data["cmdline"] == cmd


def test_ownership_marker_rejects_malformed_and_unknown_versions(tmp_path):
    profile_dir = tmp_path / "edge-dedicated-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    marker_path = get_owner_marker_path(profile_dir)

    # Unknown version
    marker_path.write_text(json.dumps({"version": 999, "pid": 123}), encoding="utf-8")
    assert read_ownership_marker(profile_dir) is None

    # Missing required field / wrong types
    marker_path.write_text(
        json.dumps(
            {
                "version": 1,
                "pid": "not_an_int",
                "port": 9222,
                "create_time": 1700000000.0,
                "profile_path": "C:\\path",
                "executable_path": "C:\\exe",
                "cmdline": ["msedge.exe"],
            }
        ),
        encoding="utf-8",
    )
    assert read_ownership_marker(profile_dir) is None


def test_get_port_listener_pids_fail_closed_on_permission_error():
    with patch("psutil.net_connections", side_effect=PermissionError("AccessDenied")):
        with patch("psutil.process_iter", side_effect=PermissionError("AccessDenied")):
            listeners = get_port_listener_pids(9222)
            assert listeners is None
            assert is_port_free(9222) is False


def test_verify_browser_ownership_stale_pid_cleanup(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=999999,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    with (
        patch("psutil.Process") as mock_proc,
        patch("jobapply.utils.browser.is_port_free", return_value=True),
    ):
        mock_proc.side_effect = psutil.NoSuchProcess(999999)
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)
        # Stale marker should be deleted when process is dead and port is free
        assert not get_owner_marker_path(profile_dir).exists()


def test_verify_browser_ownership_pid_reuse_mismatched_create_time(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    mock_ps_proc = MagicMock()
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700050000.0  # Mismatched create time (PID reused)

    with patch("psutil.Process", return_value=mock_ps_proc):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_fails_closed_on_proc_inspection_errors(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    # 1. exe() raises AccessDenied
    mock_ps_proc1 = MagicMock()
    mock_ps_proc1.is_running.return_value = True
    mock_ps_proc1.create_time.return_value = 1700000000.0
    mock_ps_proc1.exe.side_effect = psutil.AccessDenied()
    with patch("psutil.Process", return_value=mock_ps_proc1):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)

    # 2. cmdline() raises AccessDenied
    mock_ps_proc2 = MagicMock()
    mock_ps_proc2.is_running.return_value = True
    mock_ps_proc2.create_time.return_value = 1700000000.0
    mock_ps_proc2.exe.return_value = str(Path(settings.edge_executable_path).resolve())
    mock_ps_proc2.cmdline.side_effect = psutil.AccessDenied()
    with patch("psutil.Process", return_value=mock_ps_proc2):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_rejects_same_name_different_path(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700000000.0
    # Same name 'msedge.exe', but completely different directory path
    mock_ps_proc.exe.return_value = "C:\\MaliciousFolder\\msedge.exe"
    mock_ps_proc.cmdline.return_value = cmd

    with patch("psutil.Process", return_value=mock_ps_proc):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_rejects_substring_arg_attacks(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    # Port 92220 instead of 9222, and different user data dir containing substring
    attack_cmd = [
        "msedge.exe",
        "--remote-debugging-port=92220",
        f"--user-data-dir={profile_dir.resolve()}_different",
    ]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=attack_cmd,
    )

    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700000000.0
    mock_ps_proc.exe.return_value = str(Path(settings.edge_executable_path).resolve())
    mock_ps_proc.cmdline.return_value = attack_cmd

    with patch("psutil.Process", return_value=mock_ps_proc):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_empty_listeners_fails_closed(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700000000.0
    mock_ps_proc.exe.return_value = str(Path(settings.edge_executable_path).resolve())
    mock_ps_proc.cmdline.return_value = cmd

    with patch("psutil.Process", return_value=mock_ps_proc):
        # Empty listeners set (port not yet active or listening) -> must return False
        with patch("jobapply.utils.browser.get_port_listener_pids", return_value=set()):
            assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_unrelated_listener_on_port(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700000000.0
    mock_ps_proc.cmdline.return_value = cmd
    mock_ps_proc.exe.return_value = str(Path(settings.edge_executable_path).resolve())
    mock_ps_proc.children.return_value = []

    with patch("psutil.Process", return_value=mock_ps_proc):
        # Listener PID 55555 is unrelated to owned PID 12345
        with patch("jobapply.utils.browser.get_port_listener_pids", return_value={55555}):
            assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_owned_child_listener_accepted(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = ["msedge.exe", "--remote-debugging-port=9222", f"--user-data-dir={profile_dir.resolve()}"]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700000000.0
    mock_ps_proc.cmdline.return_value = cmd
    mock_ps_proc.exe.return_value = str(Path(settings.edge_executable_path).resolve())
    mock_child = MagicMock()
    mock_child.pid = 12346
    mock_ps_proc.children.return_value = [mock_child]

    with patch("psutil.Process", return_value=mock_ps_proc):
        # Listener PID 12346 is a child of owned PID 12345
        with patch("jobapply.utils.browser.get_port_listener_pids", return_value={12346}):
            assert verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_launch_edge_refuses_occupied_port_or_existing_marker(tmp_path, monkeypatch):
    profile_dir = tmp_path / "edge-dedicated-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("JOBAPPLY_EDGE_USER_DATA_DIR", str(profile_dir))
    get_settings.cache_clear()
    settings = get_settings()

    try:
        # 1. Refuses when port is occupied
        with patch("jobapply.utils.browser.is_port_free", return_value=False):
            with pytest.raises(RuntimeError, match="not provably free"):
                launch_edge_for_jobapply(settings)

        # 2. Refuses when an existing unhandled marker is present
        marker_path = get_owner_marker_path(profile_dir)
        marker_path.write_text(json.dumps({"malformed": True}), encoding="utf-8")
        with patch("jobapply.utils.browser.is_port_free", return_value=True):
            with pytest.raises(RuntimeError, match="marker file exists"):
                launch_edge_for_jobapply(settings)
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_connect_to_edge_refuses_unowned_browser(tmp_path):
    from jobapply.utils.browser import connect_to_edge

    profile_dir = tmp_path / "edge-dedicated-profile"
    pw = MagicMock()
    settings = get_settings()

    # No ownership marker exists -> connect_over_cdp is never called
    browser, port, err = await connect_to_edge(
        pw, port=9222, profile_path=profile_dir, settings=settings
    )
    assert browser is None
    assert isinstance(err, RuntimeError)
    assert "ownership verification failed" in str(err)
    pw.chromium.connect_over_cdp.assert_not_called()


@pytest.mark.asyncio
async def test_connect_to_edge_refuses_missing_ownership_context():
    from jobapply.utils.browser import connect_to_edge

    pw = MagicMock()
    # Missing profile_path / settings -> connect_over_cdp is never called
    browser1, port1, err1 = await connect_to_edge(pw, port=9222, profile_path=None, settings=None)
    assert browser1 is None
    assert isinstance(err1, RuntimeError)
    assert "required" in str(err1)

    browser2, port2, err2 = await connect_to_edge(
        pw, port=9222, profile_path=Path("C:\\dummy"), settings=None
    )
    assert browser2 is None
    assert isinstance(err2, RuntimeError)
    pw.chromium.connect_over_cdp.assert_not_called()


def test_verify_browser_ownership_rejects_marker_executable_mismatch(tmp_path):
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = [
        "msedge.exe",
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile_dir.resolve()}",
    ]
    # Marker recorded a different executable path than configured in settings
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path="C:\\DifferentPath\\msedge.exe",
        cmdline=cmd,
    )

    settings = get_settings()
    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700000000.0
    mock_ps_proc.cmdline.return_value = cmd
    mock_ps_proc.exe.return_value = str(Path(settings.edge_executable_path).resolve())

    with patch("psutil.Process", return_value=mock_ps_proc):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_rejects_marker_argument_mismatch(tmp_path):
    profile_dir = tmp_path / "edge-dedicated-profile"
    # Marker recorded different arguments
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path="C:\\msedge.exe",
        cmdline=["msedge.exe", "--some-other-flag"],
    )

    settings = get_settings()
    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    mock_ps_proc.create_time.return_value = 1700000000.0
    mock_ps_proc.cmdline.return_value = [
        "msedge.exe",
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile_dir.resolve()}",
    ]
    mock_ps_proc.exe.return_value = str(Path(settings.edge_executable_path).resolve())

    with patch("psutil.Process", return_value=mock_ps_proc):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)


def test_verify_browser_ownership_rejects_create_time_difference_in_old_window(tmp_path):
    settings = get_settings()
    profile_dir = tmp_path / "edge-dedicated-profile"
    cmd = [
        "msedge.exe",
        "--remote-debugging-port=9222",
        f"--user-data-dir={profile_dir.resolve()}",
    ]
    write_ownership_marker(
        profile_path=profile_dir,
        pid=12345,
        create_time=1700000000.0,
        port=9222,
        executable_path=str(Path(settings.edge_executable_path).resolve()),
        cmdline=cmd,
    )

    mock_ps_proc = MagicMock()
    mock_ps_proc.pid = 12345
    mock_ps_proc.is_running.return_value = True
    # 0.5s difference (was accepted under old 2-second window, now rejected under 1ms tolerance)
    mock_ps_proc.create_time.return_value = 1700000000.5
    mock_ps_proc.cmdline.return_value = cmd
    mock_ps_proc.exe.return_value = str(Path(settings.edge_executable_path).resolve())

    with patch("psutil.Process", return_value=mock_ps_proc):
        assert not verify_browser_ownership(profile_dir, expected_port=9222, settings=settings)
