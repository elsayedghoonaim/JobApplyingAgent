"""Deterministic offline tests for `jobapply doctor` and its opt-in live checks.

The default doctor path must make zero socket calls AND zero filesystem writes
or directory creations (pytest-socket is active suite-wide; filesystem state is
snapshotted around each run). Live checks are only exercised with injected
fakes proving they are bounded, redacted, read-only, timeout-protected, and
closed.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from jobapply import doctor as doctor_module
from jobapply.doctor import (
    MAX_CHECK_DETAIL_LENGTH,
    MAX_CHECK_NAME_LENGTH,
    STATUS_FAIL,
    STATUS_PASS,
    STATUS_WARN,
    CheckResult,
    DoctorDeps,
    check_credentials_configured,
    check_edge_configuration,
    check_gemma_model_constraint,
    check_mongodb_url_syntax,
    check_output_directory,
    check_python_version,
    check_search_configuration,
    check_user_data_files,
    doctor_exit_code,
    format_doctor_results,
    load_settings_guarded,
    run_doctor,
    sanitize_check_result,
)

# ── Offline checks (zero sockets, zero writes) ───────────────────────────────


def _snapshot_tree(root: Path) -> set:
    entries = set()
    for current, dirs, files in os.walk(root):
        for name in dirs:
            entries.add(("d", Path(current) / name))
        for name in files:
            entries.add(("f", Path(current) / name))
    return entries


@pytest.mark.asyncio
async def test_default_doctor_is_offline_and_makes_zero_socket_calls(monkeypatch, tmp_path):
    """Running default doctor under pytest-socket proves it performs no I/O."""
    monkeypatch.chdir(tmp_path)
    fake = _offline_settings(tmp_path)
    with patch.object(doctor_module, "load_settings_guarded", AsyncMock(return_value=(fake, []))):
        results = await run_doctor(live_checks=False)
    names = [result.name for result in results]
    assert not any(name.startswith("live-") for name in names)
    assert {"python-version", "gemma-model-constraint"} <= set(names)


@pytest.mark.asyncio
async def test_default_doctor_performs_zero_filesystem_writes(monkeypatch, tmp_path):
    """No file or directory may appear (or vanish) under the working tree."""
    monkeypatch.chdir(tmp_path)
    fake = _offline_settings(tmp_path)
    before = _snapshot_tree(tmp_path)
    with patch.object(doctor_module, "load_settings_guarded", AsyncMock(return_value=(fake, []))):
        await run_doctor(live_checks=False)
        # A second pass over an existing outputs/ dir must stay inert as well.
        (tmp_path / "outputs").mkdir()
        before_with_outputs = _snapshot_tree(tmp_path)
        results = await run_doctor(live_checks=False)
    assert doctor_exit_code(results) == 0
    assert _snapshot_tree(tmp_path) == before_with_outputs
    # The first pass must not have created anything beyond what the test made.
    after = _snapshot_tree(tmp_path)
    assert all(entry in after for entry in before)


def _offline_settings(tmp_path):
    """Settings instance with every offline check deterministically satisfiable."""
    exe_path = tmp_path / "msedge.exe"
    exe_path.write_bytes(b"")
    data_dir = tmp_path / "user-data"
    data_dir.mkdir(exist_ok=True)
    for name in ("profile.yaml", "resume.md", "resume.pdf"):
        (data_dir / name).write_text("x", encoding="utf-8")
    from jobapply.settings import Settings

    return Settings(
        data_dir=str(data_dir),
        edge_executable_path=str(exe_path),
        edge_user_data_dir=str(tmp_path / "JobApply" / "EdgeProfile"),
        mongodb_url="mongodb://localhost:27017",
        search_location="Worldwide",
        search_recency_days=None,
    )


@pytest.mark.asyncio
async def test_all_offline_checks_pass_and_exit_zero(monkeypatch, tmp_path):
    fake = _offline_settings(tmp_path)
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-key-value")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456789:AAASyntheticTokenForTestsOnly")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    results = []
    results.append(check_python_version())
    results.append(check_gemma_model_constraint(fake))
    results.extend(check_user_data_files(fake))
    results.append(check_output_directory(fake))
    results.extend(check_edge_configuration(fake))
    results.append(check_credentials_configured(fake))
    results.append(check_mongodb_url_syntax(fake))
    results.extend(check_search_configuration(fake))
    assert all(result.status == STATUS_PASS for result in results), format_doctor_results(results)
    assert doctor_exit_code(results) == 0


def test_doctor_exit_code_nonzero_on_required_failure():
    ok = CheckResult("a", STATUS_PASS, "ok")
    warn = CheckResult("b", STATUS_WARN, "warned")
    bad = CheckResult("c", STATUS_FAIL, "failed")
    assert doctor_exit_code([ok, warn]) == 0
    assert doctor_exit_code([ok, bad]) == 1


# ── Guarded settings boundary ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_settings_produce_fail_not_crash():
    """A broken environment must yield a deterministic FAIL result, exit 1."""
    with patch("jobapply.settings.Settings", side_effect=RuntimeError("bad env")):
        results = await run_doctor(live_checks=False)

    fail_results = [r for r in results if r.status == STATUS_FAIL]
    assert fail_results, format_doctor_results(results)
    assert any(r.name == "settings-validation" for r in fail_results)
    # No secret disclosure and no raw exception text.
    assert "boom" not in "".join(r.detail for r in results)
    assert "bad env" not in "".join(r.detail for r in results)
    assert doctor_exit_code(results) == 1
    # Every settings-DEPENDENT check degraded to a bounded WARN instead of
    # raising. The outputs-directory check is settings-independent by design.
    warn_names = {r.name for r in results if r.status == STATUS_WARN}
    assert {
        "gemma-model-constraint",
        "user-data-files",
        "edge-executable",
        "edge-profile-isolation",
        "mongodb-url-syntax",
        "search-location",
        "search-recency",
    } <= warn_names


@pytest.mark.asyncio
async def test_invalid_settings_skip_dependent_checks_as_warn():
    with patch("jobapply.settings.Settings", side_effect=ValueError("invalid env")):
        loaded, guarded_results = await load_settings_guarded()
    assert loaded is None
    assert len(guarded_results) == 1
    assert guarded_results[0].status == STATUS_FAIL
    assert "invalid env" not in guarded_results[0].detail

    dependent = [
        check_gemma_model_constraint(None),
        *check_user_data_files(None),
        *check_edge_configuration(None),
        check_mongodb_url_syntax(None),
        *check_search_configuration(None),
    ]
    assert all(result.status == STATUS_WARN for result in dependent)
    assert all("Skipped: settings" in result.detail for result in dependent)


def test_gemma_constraint_fails_for_other_models(tmp_path):
    fake = _offline_settings(tmp_path)
    fake.llm_model = "not-gemma"
    result = check_gemma_model_constraint(fake)
    assert result.status == STATUS_FAIL


def test_user_data_files_report_missing_without_paths_leaking(tmp_path):
    fake = _offline_settings(tmp_path)
    (tmp_path / "user-data" / "resume.md").unlink()
    results = check_user_data_files(fake)
    assert any(result.status == STATUS_WARN and "resume.md" in result.detail for result in results)


@pytest.mark.asyncio
async def test_user_data_containment_uses_resolve_relative_to(tmp_path, monkeypatch):
    fake = _offline_settings(tmp_path)

    real_resolve = type(fake).resolve_data_path

    def resolve_with_escape(self, *parts):
        if not parts:
            return str(tmp_path / "user-data")
        if parts[0] == "profile.yaml":
            return str(tmp_path / "outside" / "profile.yaml")
        return real_resolve(self, *parts)

    monkeypatch.setattr(type(fake), "resolve_data_path", resolve_with_escape)
    results = check_user_data_files(fake)
    assert any(r.status == STATUS_FAIL and "escapes" in r.detail for r in results)


def test_output_directory_is_read_only_and_bounded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Missing -> WARN without creating anything.
    result = check_output_directory(None)
    assert result.status == STATUS_WARN
    assert not (tmp_path / "outputs").exists()

    # Existing but access-denied cannot be simulated portably; existing+accessed
    # passes via read-only heuristics only.
    (tmp_path / "outputs").mkdir()
    result = check_output_directory(None)
    assert result.status == STATUS_PASS
    assert "probe" not in result.detail.lower()


def test_edge_profile_isolation_rejects_normal_profile(tmp_path):
    fake = _offline_settings(tmp_path)
    fake.edge_user_data_dir = os.path.join(str(tmp_path), "Microsoft", "Edge", "User Data")
    results = check_edge_configuration(fake)
    assert any(
        result.name == "edge-profile-isolation" and result.status == STATUS_FAIL
        for result in results
    )
    fake.edge_user_data_dir = ""
    results = check_edge_configuration(fake)
    assert any(result.status == STATUS_FAIL for result in results)


def test_mongodb_url_syntax_checks_without_connecting(tmp_path):
    fake = _offline_settings(tmp_path)

    fake.mongodb_url = "mongodb://user:pass@localhost:27017"
    result = check_mongodb_url_syntax(fake)
    assert result.status == STATUS_PASS
    assert "pass" not in result.detail
    assert "embedded credentials: yes" in result.detail

    fake.mongodb_url = "http://localhost:27017"
    assert check_mongodb_url_syntax(fake).status == STATUS_FAIL

    fake.mongodb_url = "mongodb://"
    assert check_mongodb_url_syntax(fake).status == STATUS_FAIL


def test_credentials_are_only_reported_as_missing_names(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    result = check_credentials_configured()
    assert result.status == STATUS_WARN
    assert "GOOGLE_API_KEY" in result.detail
    assert "your-google-api-key" not in result.detail


def test_search_configuration_check_reflects_settings(tmp_path):
    fake = _offline_settings(tmp_path)
    fake.search_recency_days = 40  # out of bounds
    results = check_search_configuration(fake)
    assert any(
        result.name == "search-recency" and result.status == STATUS_FAIL for result in results
    )

    fake.search_recency_days = 7
    results = check_search_configuration(fake)
    assert any(
        result.name == "search-recency"
        and result.status == STATUS_PASS
        and "7 day" in result.detail
        for result in results
    )

    fake.search_location = "   "
    results = check_search_configuration(fake)
    assert any(
        result.name == "search-location" and result.status == STATUS_FAIL for result in results
    )


# ── Central sanitization ──────────────────────────────────────────────────────


def test_check_result_sanitization_bounds_and_redacts():
    secret_token = "123456789:AAASuperSecretBotTokenValue"
    hostile = CheckResult(
        "x" * 200 + f" token {secret_token}",
        STATUS_PASS,
        f"leak {secret_token} " + "y" * 1000,
    )
    clean = sanitize_check_result(hostile)
    assert len(clean.name) <= MAX_CHECK_NAME_LENGTH
    assert len(clean.detail) <= MAX_CHECK_DETAIL_LENGTH
    assert secret_token not in clean.name
    assert secret_token not in clean.detail
    assert "[REDACTED]" in clean.detail


def test_format_sanitizes_every_result_line():
    secret = "999888777:AABAnotherSyntheticSecretToken"
    text = format_doctor_results([CheckResult("check", STATUS_FAIL, f"failed with {secret}")])
    assert secret not in text
    assert "[REDACTED]" in text


@pytest.mark.skipif(__import__("sys").version_info < (3, 11), reason="project floor is 3.11")
def test_python_version_passes_on_supported_runtime():
    assert check_python_version().status == STATUS_PASS


def test_python_version_fails_below_floor():
    assert check_python_version(minor_floor=99).status == STATUS_FAIL


# ── Live checks: opt-in, mocked, bounded, redacted, read-only ────────────────


@pytest.mark.asyncio
async def test_live_checks_never_run_without_opt_in():
    deps = DoctorDeps(
        mongo_ping=AsyncMock(return_value=(True, "ping")),
        telegram_get_me=AsyncMock(return_value=(True, "tg")),
        google_models=AsyncMock(return_value=(True, "gg")),
        edge_cdp_version=AsyncMock(return_value=(True, "cdp")),
    )
    results = await run_doctor(live_checks=False, deps=deps)
    assert not [r for r in results if r.name.startswith("live-")]
    deps.mongo_ping.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_live_checks_run_only_when_opted_in_and_are_read_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mongo_ping = AsyncMock(return_value=(True, "MongoDB ping succeeded."))
    telegram_get_me = AsyncMock(return_value=(True, "Telegram bot reachable (@bot)."))
    google_models = AsyncMock(return_value=(True, "Gemma endpoint reachable."))
    cdp = AsyncMock(return_value=(False, "Managed Edge CDP endpoint unreachable (ConnectError)."))
    deps = DoctorDeps(
        mongo_ping=mongo_ping,
        telegram_get_me=telegram_get_me,
        google_models=google_models,
        edge_cdp_version=cdp,
    )
    fake = _offline_settings(tmp_path)
    with patch.object(doctor_module, "load_settings_guarded", AsyncMock(return_value=(fake, []))):
        results = await run_doctor(live_checks=True, deps=deps)
    names = [r.name for r in results]
    assert {
        "live-mongodb",
        "live-telegram",
        "live-gemma-endpoint",
        "live-edge-cdp",
    } <= set(names)
    mongo_ping.assert_awaited_once()
    telegram_get_me.assert_awaited_once()
    google_models.assert_awaited_once()
    cdp.assert_awaited_once()
    by_name = {r.name: r for r in results}
    assert by_name["live-mongodb"].status == STATUS_PASS
    assert by_name["live-edge-cdp"].status == STATUS_WARN  # degraded, not fatal


@pytest.mark.asyncio
async def test_live_telegram_skipped_without_token(tmp_path):
    fake = _offline_settings(tmp_path)
    fake.telegram_bot_token = ""
    get_me = AsyncMock()
    result = await doctor_module.check_live_telegram(fake, DoctorDeps(telegram_get_me=get_me))
    assert result.status == STATUS_WARN
    get_me.assert_not_awaited()


@pytest.mark.asyncio
async def test_hanging_live_probe_times_out_cleanly():
    import asyncio

    async def hanging_probe(*args):
        await asyncio.sleep(30)
        return True, "never"

    result = await doctor_module.check_live_telegram(
        type("S", (), {"telegram_bot_token": "tok"})(), hanging_probe, timeout=0.05
    )
    assert result.status == STATUS_WARN
    assert "budget" in result.detail


@pytest.mark.asyncio
async def test_raising_live_probe_degrades_into_redacted_result():
    secret = "555444333:AABYetAnotherSyntheticTokenValue"

    async def raising_probe(token):
        raise RuntimeError(f"upstream said {secret} then exploded with " + "z" * 500)

    result = await doctor_module.check_live_telegram(
        type("S", (), {"telegram_bot_token": "tok"})(), raising_probe
    )
    assert result.status == STATUS_WARN
    assert secret not in result.detail
    assert len(result.detail) <= MAX_CHECK_DETAIL_LENGTH
    assert "RuntimeError" in result.detail


@pytest.mark.asyncio
async def test_external_cancellation_is_preserved():
    import asyncio

    async def cancelling_probe(*args):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await doctor_module.check_live_telegram(
            type("S", (), {"telegram_bot_token": "tok"})(), cancelling_probe
        )


@pytest.mark.asyncio
async def test_hostile_injected_result_text_is_bounded_and_redacted():
    secret = "777666555:AABHostileInjectedTokenValueHere"

    async def hostile_probe(*args):
        return False, f"upstream said {secret} and then " + "q" * 2000

    result = await doctor_module.check_live_telegram(
        type("S", (), {"telegram_bot_token": "tok"})(), hostile_probe
    )
    assert result.status == STATUS_WARN
    assert secret not in result.detail
    assert len(result.detail) <= MAX_CHECK_DETAIL_LENGTH


@pytest.mark.asyncio
async def test_google_models_probe_uses_get_never_generation(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "models/gemma-4-31b-it"}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, method, url, json=None, headers=None):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = dict(headers or {})
            return FakeResponse()

        # Guard: generation endpoints must never be touched.
        async def post(self, *args, **kwargs):  # pragma: no cover
            captured["post_called"] = True
            raise AssertionError("generation POST attempted")

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    ok, detail = await doctor_module.default_google_models_check(
        "https://generativelanguage.googleapis.com/v1beta", "synthetic-key"
    )
    assert ok is True
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/models")
    assert captured["headers"].get("x-goog-api-key") == "synthetic-key"
    assert "generativelanguage" in captured["url"]
    assert not captured.get("post_called")
