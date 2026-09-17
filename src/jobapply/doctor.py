"""`jobapply doctor` — offline-first configuration diagnostics.

Default behavior is offline and strictly non-mutating: no filesystem writes or
directory creation, no network calls, subprocesses, or browser launches. Every
offline check inspects only local configuration, files, and access bits.

``--live-checks`` is the only authorization for bounded connectivity checks:
short-timeout, read-only probes of the configured MongoDB deployment, the
Telegram ``getMe`` endpoint, the Gemma/Google endpoint (GET models list —
never content generation), and the managed Edge CDP endpoint. Live checks
never navigate to LinkedIn, never send Telegram messages, never write MongoDB
data, never expose secrets or credential-bearing URLs, run under an outer
timeout boundary, degrade every failure into a bounded recursively-redacted
diagnostic result, and close every resource they own.

Individual check functions are dependency-injectable and independently
testable; exit codes are deterministic (0 when all required checks pass,
nonzero when any required check fails). Every ``CheckResult`` is centrally
sanitized (redacted and bounded) before formatting.
"""

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

from jobapply.utils.redaction import redact_string

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"

EXIT_OK = 0
EXIT_CHECKS_FAILED = 1

LIVE_TIMEOUT_SECONDS = 5.0

MAX_CHECK_NAME_LENGTH = 60
MAX_CHECK_DETAIL_LENGTH = 300

# Normal Edge profile locations that must never be used as the dedicated profile.
_NORMAL_EDGE_PROFILE_MARKERS = (
    "microsoft\\edge\\user data",
    "microsoft/edge/user data",
)

REQUIRED_USER_DATA_FILES = ("profile.yaml", "resume.md", "resume.pdf")
REQUIRED_CREDENTIAL_KEYS = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")

_SETTINGS_UNAVAILABLE_DETAIL = "Skipped: settings could not be loaded."


@dataclass(frozen=True)
class CheckResult:
    """One bounded doctor check outcome."""

    name: str
    status: str
    detail: str


def sanitize_check_result(result: CheckResult) -> CheckResult:
    """Centrally redact and bound a result's name and detail before output."""
    return CheckResult(
        name=redact_string(str(result.name or "unnamed"))[:MAX_CHECK_NAME_LENGTH],
        status=str(result.status)[:8],
        detail=redact_string(str(result.detail or ""))[:MAX_CHECK_DETAIL_LENGTH],
    )


@dataclass
class DoctorDeps:
    """Injectable connectivity dependencies for live checks.

    Each dependency performs one bounded read-only probe. Tests substitute
    fakes here; the defaults use short-timeout HTTP clients that are always
    closed. Every invocation is additionally wrapped in an outer
    ``asyncio.wait_for`` boundary by :func:`run_doctor`.
    """

    mongo_ping: Optional[Callable[[Any], Awaitable[tuple[bool, str]]]] = None
    telegram_get_me: Optional[Callable[[str], Awaitable[tuple[bool, str]]]] = None
    google_models: Optional[Callable[[str, str], Awaitable[tuple[bool, str]]]] = None
    edge_cdp_version: Optional[Callable[[int], Awaitable[tuple[bool, str]]]] = None

    def __post_init__(self) -> None:
        if self.mongo_ping is None:
            self.mongo_ping = default_mongo_ping
        if self.telegram_get_me is None:
            self.telegram_get_me = default_telegram_get_me
        if self.google_models is None:
            self.google_models = default_google_models_check
        if self.edge_cdp_version is None:
            self.edge_cdp_version = default_edge_cdp_version


# ── Guarded settings boundary ─────────────────────────────────────────────────


async def load_settings_guarded() -> tuple[Optional[Any], list[CheckResult]]:
    """Load settings without ever raising; invalid env yields a deterministic FAIL."""
    from jobapply.settings import Settings

    try:
        return Settings(), []
    except Exception as exc:
        # Bounded class-level diagnosis only; never echo raw values/messages,
        # which can carry environment contents.
        detail = f"Settings failed validation ({type(exc).__name__}); dependent checks skipped."
        return None, [CheckResult("settings-validation", STATUS_FAIL, detail)]


# ── Offline checks ────────────────────────────────────────────────────────────


def check_python_version(minor_floor: int = 11) -> CheckResult:
    """Require the supported Python floor declared by the project."""
    version = sys.version_info
    if version >= (3, minor_floor):
        return CheckResult(
            "python-version",
            STATUS_PASS,
            f"Python {version.major}.{version.minor}.{version.micro} is supported.",
        )
    return CheckResult(
        "python-version",
        STATUS_FAIL,
        f"Python 3.{minor_floor}+ is required; found {version.major}.{version.minor}.",
    )


def _skipped_result(name: str) -> CheckResult:
    return CheckResult(name, STATUS_WARN, _SETTINGS_UNAVAILABLE_DETAIL)


def check_gemma_model_constraint(settings: Optional[Any]) -> CheckResult:
    """Validate the selected LLM provider, model, and endpoint combination."""
    if settings is None:
        return _skipped_result("gemma-model-constraint")
    provider = getattr(settings, "llm_provider", "gemini")
    model = getattr(settings, "llm_model", "")
    base_url = getattr(settings, "llm_base_url", "")
    if provider == "gemini" and model == "gemma-4-31b-it":
        return CheckResult(
            "gemma-model-constraint", STATUS_PASS, "Gemini model is gemma-4-31b-it."
        )
    if provider == "openrouter" and model.strip() and "openrouter.ai" in base_url:
        return CheckResult(
            "gemma-model-constraint",
            STATUS_PASS,
            f"OpenRouter model is configured ({model[:100]}).",
        )
    return CheckResult(
        "gemma-model-constraint",
        STATUS_FAIL,
        "LLM provider, model, and base URL are inconsistent.",
    )


def check_user_data_files(settings: Optional[Any]) -> list[CheckResult]:
    """Required user-data files exist under safe contained paths (read-only)."""
    if settings is None:
        return [_skipped_result("user-data-files")]
    results: list[CheckResult] = []
    missing: list[str] = []
    base_raw = settings.resolve_data_path()
    base_path = Path(base_raw).resolve()
    for name in REQUIRED_USER_DATA_FILES:
        path = Path(settings.resolve_data_path(name))
        try:
            contained = path.resolve().relative_to(base_path) is not None
        except ValueError:
            contained = False
        except OSError:
            contained = False
        if not contained:
            results.append(
                CheckResult(
                    "user-data-files",
                    STATUS_FAIL,
                    f"Resolved '{name}' escapes the configured data directory.",
                )
            )
            continue
        if not path.is_file():
            missing.append(name)
    if missing:
        results.append(
            CheckResult(
                "user-data-files",
                STATUS_WARN,
                f"Missing user-data file(s): {', '.join(missing)}.",
            )
        )
    elif not results:
        results.append(
            CheckResult(
                "user-data-files",
                STATUS_PASS,
                f"All required files present in '{settings.data_dir}'.",
            )
        )
    return results


def check_output_directory(settings: Optional[Any]) -> CheckResult:
    """Outputs directory accessibility using read-only inspection only.

    Writability cannot be proven without mutation, so this check relies on
    existence plus OS access bits and reports a bounded WARN whenever that
    heuristic cannot establish readiness. It never creates directories or
    writes probe files.
    """
    del settings  # outputs root depends only on CWD by contract
    root = Path("outputs").resolve()
    if not root.exists():
        return CheckResult(
            "output-directory",
            STATUS_WARN,
            "Outputs directory does not exist yet; it is created on first run.",
        )
    if not os.access(root, os.R_OK | os.W_OK):
        return CheckResult(
            "output-directory",
            STATUS_FAIL,
            "Outputs directory exists but read/write access was denied.",
        )
    return CheckResult(
        "output-directory",
        STATUS_PASS,
        f"Outputs directory '{root.name}/' exists with read/write access bits set.",
    )


def check_edge_configuration(settings: Optional[Any]) -> list[CheckResult]:
    """Edge executable exists and the profile stays dedicated/isolated."""
    if settings is None:
        return [
            _skipped_result("edge-executable"),
            _skipped_result("edge-profile-isolation"),
        ]
    results: list[CheckResult] = []

    exe = settings.edge_executable_path
    if exe and Path(exe).is_file():
        results.append(
            CheckResult("edge-executable", STATUS_PASS, "Configured Edge executable exists.")
        )
    else:
        results.append(
            CheckResult(
                "edge-executable",
                STATUS_FAIL,
                "Configured Edge executable was not found on disk.",
            )
        )

    normalized = (settings.edge_user_data_dir or "").replace("%LOCALAPPDATA%", "").strip().lower()
    if any(marker in normalized for marker in _NORMAL_EDGE_PROFILE_MARKERS) or normalized in (
        "",
        "default",
    ):
        results.append(
            CheckResult(
                "edge-profile-isolation",
                STATUS_FAIL,
                "Edge profile must be the dedicated JobApply profile, not the normal Edge profile.",
            )
        )
    else:
        results.append(
            CheckResult(
                "edge-profile-isolation",
                STATUS_PASS,
                "Dedicated Edge profile configured; normal Edge profile untouched.",
            )
        )
    return results


def _credential_is_configured(key: str) -> bool:
    """A credential counts when either its plain or JOBAPPLY_-prefixed name is set."""
    return bool((os.getenv(key) or "").strip()) or bool(
        (os.getenv(f"JOBAPPLY_{key}") or "").strip()
    )


def check_credentials_configured(settings: Optional[Any] = None) -> CheckResult:
    """Report required credentials only as configured/missing; values are never printed."""
    provider = getattr(settings, "llm_provider", None) or os.getenv(
        "JOBAPPLY_LLM_PROVIDER", "gemini"
    )
    llm_key = "OPENROUTER_API_KEY" if provider == "openrouter" else "GOOGLE_API_KEY"
    required = (llm_key, *REQUIRED_CREDENTIAL_KEYS)
    missing = [key for key in required if not _credential_is_configured(key)]
    if not missing:
        return CheckResult("credentials", STATUS_PASS, "All required credentials are configured.")
    return CheckResult(
        "credentials",
        STATUS_WARN,
        f"Missing credential(s): {', '.join(missing)} (values are never printed).",
    )


def check_mongodb_url_syntax(settings: Optional[Any]) -> CheckResult:
    """Validate MongoDB URL syntax without connecting; credentials never printed."""
    if settings is None:
        return _skipped_result("mongodb-url-syntax")
    url = settings.mongodb_url
    parsed = urlparse(url)
    if parsed.scheme not in ("mongodb", "mongodb+srv"):
        return CheckResult(
            "mongodb-url-syntax",
            STATUS_FAIL,
            "MongoDB URL must start with mongodb:// or mongodb+srv://.",
        )
    if not parsed.hostname:
        return CheckResult("mongodb-url-syntax", STATUS_FAIL, "MongoDB URL has no host.")
    has_creds = parsed.username is not None
    detail = f"Scheme and host are valid (embedded credentials: {'yes' if has_creds else 'no'})."
    return CheckResult("mongodb-url-syntax", STATUS_PASS, detail)


def check_search_configuration(settings: Optional[Any]) -> list[CheckResult]:
    """Validate search scope and per-user eligibility preferences."""
    from jobapply.utils.search_config import normalize_recency_days, normalize_search_location

    if settings is None:
        return [
            _skipped_result("search-location"),
            _skipped_result("search-recency"),
            _skipped_result("search-queries"),
            _skipped_result("target-title-keywords"),
            _skipped_result("allowed-languages"),
        ]
    results: list[CheckResult] = []

    location, loc_err = normalize_search_location(settings.search_location)
    if loc_err or location is None:
        results.append(CheckResult("search-location", STATUS_FAIL, loc_err or "Invalid location."))
    else:
        results.append(CheckResult("search-location", STATUS_PASS, f"Location filter: {location}"))

    recency, rec_err = normalize_recency_days(settings.search_recency_days)
    if rec_err:
        results.append(CheckResult("search-recency", STATUS_FAIL, rec_err))
    elif recency is None:
        results.append(
            CheckResult(
                "search-recency",
                STATUS_PASS,
                "Recency filter disabled (all posting dates). Set 1-30 days to enable.",
            )
        )
    else:
        results.append(
            CheckResult("search-recency", STATUS_PASS, f"Recency filter: last {recency} day(s).")
        )

    queries = list(settings.search_queries_list)
    targets = list(settings.target_title_keywords_list)
    languages = list(settings.allowed_languages_list)
    results.append(
        CheckResult(
            "search-queries",
            STATUS_PASS if queries else STATUS_FAIL,
            f"Discovery queries: {', '.join(queries)}"
            if queries
            else "No search queries configured.",
        )
    )
    results.append(
        CheckResult(
            "target-title-keywords",
            STATUS_PASS if targets else STATUS_FAIL,
            (
                f"Strict title phrases: {', '.join(targets)}; "
                f"exclude senior titles: {str(settings.exclude_senior_titles).lower()}."
                if targets
                else "No target title phrases configured."
            ),
        )
    )
    results.append(
        CheckResult(
            "allowed-languages",
            STATUS_PASS if languages else STATUS_FAIL,
            f"Allowed required languages: {', '.join(languages)}"
            if languages
            else "No allowed languages configured.",
        )
    )
    return results


# ── Live checks (opt-in via --live-checks only) ──────────────────────────────


async def default_mongo_ping(settings: Any) -> tuple[bool, str]:
    """Bounded read-only admin ping against the configured MongoDB deployment.

    Uses a short-lived client with a strict server-selection timeout and always
    closes it; the shared process-owned client lifecycle is untouched.
    """
    from motor.motor_asyncio import AsyncIOMotorClient

    client: Optional[AsyncIOMotorClient] = None
    try:
        client = AsyncIOMotorClient(
            settings.mongodb_url, serverSelectionTimeoutMS=int(LIVE_TIMEOUT_SECONDS * 1000)
        )
        await client.admin.command("ping")
        return True, "MongoDB ping succeeded."
    except Exception as exc:
        return False, f"MongoDB ping failed ({type(exc).__name__})."
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


async def _bounded_http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> tuple[Optional[Any], Optional[str]]:
    """One bounded HTTP request with short timeout; returns (json, redacted_error)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=LIVE_TIMEOUT_SECONDS) as client:
            response = await client.request(method, url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json(), None
    except Exception as exc:
        return (
            None,
            f"{type(exc).__name__}: {redact_string(str(exc))[:160]}",
        )


async def default_telegram_get_me(bot_token: str) -> tuple[bool, str]:
    """Read-only Telegram getMe probe; token never appears in output."""
    data, error = await _bounded_http_json(
        f"https://api.telegram.org/bot{bot_token}/getMe", method="POST", payload={}
    )
    if error is not None:
        reason = error.split(":", 1)[-1].strip() or "transport_error"
        return False, f"Telegram getMe unavailable ({reason})."
    if data and data.get("ok"):
        username = str((data.get("result") or {}).get("username") or "unknown")
        return True, f"Telegram bot reachable (@{username[:32]})."
    return False, "Telegram getMe rejected the request."


async def default_google_models_check(base_url: str, api_key: str) -> tuple[bool, str]:
    """Read-only Gemma availability probe: GET models list with header authentication.

    This never performs content generation and never puts the key in a URL.
    """
    data, error = await _bounded_http_json(
        f"{base_url.rstrip('/')}/models",
        method="GET",
        headers={"x-goog-api-key": api_key},
    )
    if error is not None:
        reason = error.split(":", 1)[-1].strip() or "transport_error"
        return False, f"Gemma endpoint unavailable ({reason})."
    if isinstance(data, dict) and data.get("models") is not None:
        return True, "Gemma endpoint reachable (models list available)."
    return False, "Gemma endpoint responded unexpectedly."


async def default_openrouter_models_check(base_url: str, api_key: str) -> tuple[bool, str]:
    """Read-only OpenRouter availability probe using the models endpoint."""
    data, error = await _bounded_http_json(
        f"{base_url.rstrip('/')}/models",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if error is not None:
        reason = error.split(":", 1)[-1].strip() or "transport_error"
        return False, f"OpenRouter endpoint unavailable ({reason})."
    if isinstance(data, dict) and data.get("data") is not None:
        return True, "OpenRouter endpoint reachable (models list available)."
    return False, "OpenRouter endpoint responded unexpectedly."


async def default_edge_cdp_version(port: int) -> tuple[bool, str]:
    """Read-only managed Edge CDP endpoint probe on localhost."""
    data, error = await _bounded_http_json(f"http://127.0.0.1:{port}/json/version", method="GET")
    if error is not None:
        reason = error.split(":", 1)[-1].strip() or "transport_error"
        return False, f"Managed Edge CDP endpoint unreachable ({reason})."
    browser = str((data or {}).get("Browser") or "unknown")[:60]
    return True, f"Managed Edge CDP endpoint answered ({browser})."


async def _guarded_live_probe(
    name: str, severity_fail: bool, probe, *args, timeout: float
) -> CheckResult:
    """Run one live dependency under an outer timeout/exception boundary.

    Timeouts and unexpected exceptions become bounded, redacted diagnostic
    results; external task cancellation is always preserved.
    """
    try:
        ok, detail = await asyncio.wait_for(probe(*args), timeout)
    except asyncio.TimeoutError:
        ok, detail = False, f"Live probe exceeded its {timeout:.1f}s budget."
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        ok = False
        detail = f"Live probe raised {type(exc).__name__}: {redact_string(str(exc))[:160]}"
    status = STATUS_PASS if ok else (STATUS_FAIL if severity_fail else STATUS_WARN)
    return sanitize_check_result(CheckResult(name, status, str(detail)))


async def check_live_mongodb(
    settings: Any, mongo_fn, *, timeout: float = LIVE_TIMEOUT_SECONDS
) -> CheckResult:
    return await _guarded_live_probe("live-mongodb", True, mongo_fn, settings, timeout=timeout)


async def check_live_telegram(
    settings: Any, telegram_fn, *, timeout: float = LIVE_TIMEOUT_SECONDS
) -> CheckResult:
    token = settings.telegram_bot_token if settings is not None else ""
    if not token.strip():
        return sanitize_check_result(
            CheckResult("live-telegram", STATUS_WARN, "Telegram bot token not configured; skipped.")
        )
    return await _guarded_live_probe("live-telegram", False, telegram_fn, token, timeout=timeout)


async def check_live_google(
    settings: Any, google_fn, *, timeout: float = LIVE_TIMEOUT_SECONDS
) -> CheckResult:
    provider = getattr(settings, "llm_provider", "gemini")
    key_name = "OPENROUTER_API_KEY" if provider == "openrouter" else "GOOGLE_API_KEY"
    api_key = os.getenv(key_name) or os.getenv(f"JOBAPPLY_{key_name}") or ""
    if not api_key.strip():
        return sanitize_check_result(
            CheckResult(
                "live-gemma-endpoint", STATUS_WARN, f"{key_name} not configured; skipped."
            )
        )
    if provider == "openrouter" and google_fn is default_google_models_check:
        google_fn = default_openrouter_models_check
    return await _guarded_live_probe(
        "live-gemma-endpoint", False, google_fn, settings.llm_base_url, api_key, timeout=timeout
    )


async def check_live_edge_cdp(
    settings: Any, cdp_fn, *, timeout: float = LIVE_TIMEOUT_SECONDS
) -> CheckResult:
    return await _guarded_live_probe(
        "live-edge-cdp", False, cdp_fn, settings.edge_debug_port, timeout=timeout
    )


# ── Orchestration ────────────────────────────────────────────────────────────


async def run_doctor(
    live_checks: bool = False,
    deps: Optional[DoctorDeps] = None,
    *,
    live_timeout_seconds: float = LIVE_TIMEOUT_SECONDS,
) -> list[CheckResult]:
    """Run all doctor checks and return sanitized deterministic results.

    With ``live_checks=False`` this function makes zero socket calls and zero
    filesystem writes. Settings load through a guarded boundary: invalid
    environment values yield a deterministic FAIL (exit code 1) while every
    dependent check degrades to a bounded WARN instead of raising.
    """
    settings, guarded_results = await load_settings_guarded()

    results: list[CheckResult] = []
    results.append(check_python_version())
    results.extend(guarded_results)
    results.append(check_gemma_model_constraint(settings))
    results.extend(check_user_data_files(settings))
    results.append(check_output_directory(settings))
    results.extend(check_edge_configuration(settings))
    results.append(check_credentials_configured(settings))
    results.append(check_mongodb_url_syntax(settings))
    results.extend(check_search_configuration(settings))

    if live_checks:
        injected = deps or DoctorDeps()
        mongo_fn = injected.mongo_ping or default_mongo_ping
        telegram_fn = injected.telegram_get_me or default_telegram_get_me
        google_fn = injected.google_models or default_google_models_check
        cdp_fn = injected.edge_cdp_version or default_edge_cdp_version
        effective_settings = settings
        if effective_settings is None:
            # Live probes need settings; degrade uniformly without touching env.
            for name in ("live-mongodb", "live-gemma-endpoint", "live-edge-cdp"):
                results.append(_skipped_result(name))
            results.append(_skipped_result("live-telegram"))
        else:
            results.append(
                await check_live_mongodb(effective_settings, mongo_fn, timeout=live_timeout_seconds)
            )
            results.append(
                await check_live_telegram(
                    effective_settings, telegram_fn, timeout=live_timeout_seconds
                )
            )
            results.append(
                await check_live_google(effective_settings, google_fn, timeout=live_timeout_seconds)
            )
            results.append(
                await check_live_edge_cdp(effective_settings, cdp_fn, timeout=live_timeout_seconds)
            )

    return [sanitize_check_result(result) for result in results]


def doctor_exit_code(results: list[CheckResult]) -> int:
    """Zero when all required checks pass; nonzero when any required check fails."""
    return (
        EXIT_CHECKS_FAILED if any(result.status == STATUS_FAIL for result in results) else EXIT_OK
    )


def format_doctor_results(results: list[CheckResult]) -> str:
    sanitized = [sanitize_check_result(result) for result in results]
    lines = ["JobApply doctor", ""]
    for result in sanitized:
        lines.append(f"[{result.status}] {result.name}: {result.detail}")
    passed = sum(1 for r in sanitized if r.status == STATUS_PASS)
    warned = sum(1 for r in sanitized if r.status == STATUS_WARN)
    failed = sum(1 for r in sanitized if r.status == STATUS_FAIL)
    lines.extend(["", f"Summary: {passed} passed, {warned} warnings, {failed} failed"])
    return "\n".join(lines)


def run_doctor_cli(live_checks: bool = False) -> int:
    """CLI wrapper: print bounded results and return the deterministic exit code."""
    results = asyncio.run(run_doctor(live_checks=live_checks))
    print(format_doctor_results(results))
    return doctor_exit_code(results)


if __name__ == "__main__":  # pragma: no cover
    parser_live = "--live-checks" in sys.argv[1:]
    raise SystemExit(run_doctor_cli(live_checks=parser_live))
