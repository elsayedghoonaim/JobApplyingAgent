"""Tests for JOBAPPLY_DATA_DIR resolution, template tracking, and caller compliance."""

import os
from pathlib import Path

from jobapply.settings import get_settings


def test_relative_data_dir_resolves_from_repo_root(monkeypatch, tmp_path):
    """Verify relative JOBAPPLY_DATA_DIR resolves relative to repo root even after CWD change."""
    monkeypatch.setenv("JOBAPPLY_DATA_DIR", "user-data")
    get_settings.cache_clear()
    settings = get_settings()

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        resolved = settings.resolve_data_path("profile.yaml")
        assert os.path.isabs(resolved)
        assert "user-data" in resolved
        assert not resolved.startswith(str(tmp_path))
    finally:
        os.chdir(original_cwd)
        get_settings.cache_clear()


def test_absolute_data_dir_resolves_as_is(monkeypatch, tmp_path):
    """Verify absolute JOBAPPLY_DATA_DIR resolves directly to the absolute path."""
    custom_dir = str(tmp_path / "custom_data")
    monkeypatch.setenv("JOBAPPLY_DATA_DIR", custom_dir)
    get_settings.cache_clear()
    settings = get_settings()

    resolved = settings.resolve_data_path("profile.yaml")
    assert resolved == os.path.abspath(os.path.join(custom_dir, "profile.yaml"))
    get_settings.cache_clear()


def test_templates_tracked_user_data_ignored():
    """Verify templates directory exists with files and gitignore covers user-data/ and legacy paths."""
    repo_root = Path(__file__).parent.parent
    templates_dir = repo_root / "templates"

    assert (templates_dir / "profile.yaml").exists()
    assert (templates_dir / "resume.md").exists()
    assert (templates_dir / "resume.pdf").exists()

    gitignore_text = (repo_root / ".gitignore").read_text(encoding="utf-8")
    assert "user-data/" in gitignore_text
    assert "src/jobapply/data/profile.yaml" in gitignore_text
    assert "src/jobapply/data/resume.md" in gitignore_text
    assert "src/jobapply/data/resume.pdf" in gitignore_text


def test_all_data_callers_use_settings_resolver():
    """Verify no source code files contain hardcoded src/jobapply/data path strings."""
    repo_root = Path(__file__).parent.parent
    src_dir = repo_root / "src"

    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert 'Path("src/jobapply/data/' not in content


def test_package_version_aligned():
    """Verify package __version__ aligns with pyproject.toml / package metadata."""
    import importlib.metadata

    import jobapply

    assert jobapply.__version__ == "0.3.0"
    try:
        meta_version = importlib.metadata.version("jobapply")
        assert jobapply.__version__ == meta_version
    except importlib.metadata.PackageNotFoundError:
        pass


def test_network_socket_blocked_while_async_initializes():
    """Verify standard AF_INET network socket construction is blocked by SocketBlockedError."""
    import socket

    import pytest
    import pytest_socket

    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
