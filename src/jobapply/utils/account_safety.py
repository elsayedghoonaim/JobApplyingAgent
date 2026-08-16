"""Account-safety detection module for LinkedIn automation.

Detects security challenges, CAPTCHAs, 2FA/OTP prompts, account restrictions,
authwalls, and rate limits. Never solves, evades, or bypasses any safety barriers.
"""

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

from jobapply.utils.redaction import redact_string


class AccountSafetyBarrierType(str, Enum):
    """Classified types of LinkedIn account safety barriers."""

    CAPTCHA = "captcha"
    CHECKPOINT = "checkpoint"
    TWO_FACTOR = "two_factor"
    RESTRICTION = "restriction"
    AUTHWALL = "authwall"
    RATE_LIMIT = "rate_limit"
    INSPECTION_UNAVAILABLE = "inspection_unavailable"


SAFE_RESUME_INSTRUCTIONS = (
    "Open the visible dedicated Microsoft Edge profile, resolve the security barrier "
    "manually in the browser, verify your LinkedIn session is active and normal, "
    "then start a NEW JobApply session. Do not reuse the paused run ID. "
    "Never share OTPs, passwords, or CAPTCHA answers."
)

# LinkedIn-specific URL path markers (checked ONLY on linkedin.com / *.linkedin.com paths)
LINKEDIN_CAPTCHA_PATH_PREFIXES = (
    "/checkpoint/challenge/captcha",
    "/arkose",
    "/funcaptcha",
)

LINKEDIN_2FA_PATH_PREFIXES = (
    "/checkpoint/challenge/two-step-verification",
    "/checkpoint/challenge/totp",
    "/checkpoint/challenge/sms",
    "/checkpoint/challenge/email-pin",
    "/checkpoint/challenge/device-verification",
    "/checkpoint/challenge/phone-pin",
    "/checkpoint/challenge/app-verification",
)

LINKEDIN_CHECKPOINT_PATH_PREFIXES = (
    "/checkpoint/challenge",
    "/checkpoint/lg",
    "/checkpoint/rp",
    "/checkpoint/id",
    "/checkpoint/v2",
    "/checkpoint",
    "/litms/vendor",
    "/security/verification",
    "/security/checkpoint",
    "/checkpoint-challenge",
)

LINKEDIN_RESTRICTION_PATH_PREFIXES = (
    "/identity/restricted",
    "/feed/restriction",
    "/help/linkedin/answer/restriction",
    "/account-restricted",
    "/suspended",
    "/restriction",
)

LINKEDIN_AUTHWALL_PATH_PREFIXES = (
    "/authwall",
    "/uas/login",
    "/login",
    "/signup",
    "/uas/consumer-login",
)

LINKEDIN_RATE_LIMIT_PATH_PREFIXES = (
    "/429",
    "/throttle",
    "/too-many-requests",
    "/rate-limit",
)

# Approved external CAPTCHA provider hosts
EXTERNAL_CAPTCHA_HOSTS = (
    "arkoselabs.com",
    "client-api.arkoselabs.com",
    "hcaptcha.com",
    "recaptcha.net",
)

# Known static route segments that should not be redacted in URLs
KNOWN_SAFE_PATH_SEGMENTS = {
    "",
    "checkpoint",
    "challenge",
    "two-step-verification",
    "totp",
    "sms",
    "email-pin",
    "device-verification",
    "phone-pin",
    "app-verification",
    "captcha",
    "lg",
    "rp",
    "id",
    "v2",
    "identity",
    "restricted",
    "feed",
    "restriction",
    "help",
    "linkedin",
    "answer",
    "account-restricted",
    "suspended",
    "authwall",
    "uas",
    "login",
    "signup",
    "consumer-login",
    "429",
    "throttle",
    "too-many-requests",
    "rate-limit",
    "litms",
    "vendor",
    "security",
    "verification",
    "jobs",
    "view",
    "search",
    "arkose",
    "funcaptcha",
}

# High-confidence page title patterns
TITLE_PATTERNS: list[tuple[AccountSafetyBarrierType, list[str]]] = [
    (
        AccountSafetyBarrierType.TWO_FACTOR,
        [
            "two-step verification",
            "two factor authentication",
            "2-step verification",
            "enter verification code",
            "enter the code",
            "confirm your identity with a code",
            "device verification",
        ],
    ),
    (
        AccountSafetyBarrierType.CAPTCHA,
        [
            "security challenge",
            "human verification",
            "bot detection",
            "solve the puzzle",
            "solve the puzzle to continue",
            "captcha challenge",
            "recaptcha challenge",
            "arkose challenge",
            "captcha | linkedin",
        ],
    ),
    (
        AccountSafetyBarrierType.RESTRICTION,
        [
            "account restricted",
            "account suspended",
            "temporary restriction",
            "your account has been restricted",
            "account has been temporarily restricted",
        ],
    ),
    (
        AccountSafetyBarrierType.RATE_LIMIT,
        [
            "429 too many requests",
            "too many requests",
            "rate limit exceeded",
            "request rate exceeded",
        ],
    ),
    (
        AccountSafetyBarrierType.CHECKPOINT,
        [
            "security verification",
            "quick verification",
            "linkedin security",
            "security check",
            "checkpoint | linkedin",
            "verify your identity",
            "identity verification",
            "challenge | linkedin",
        ],
    ),
    (
        AccountSafetyBarrierType.AUTHWALL,
        [
            "linkedin login, sign in | linkedin",
            "sign in | linkedin",
            "log in, sign in | linkedin",
            "linkedin: log in or sign up",
            "join linkedin",
        ],
    ),
]

# Known selectors for safety barriers
SELECTOR_PATTERNS: list[tuple[AccountSafetyBarrierType, list[str]]] = [
    (
        AccountSafetyBarrierType.CAPTCHA,
        [
            "iframe[src*='arkoselabs.com']",
            "iframe[src*='recaptcha']",
            "iframe[src*='hcaptcha']",
            "#captcha-internal",
            "#arkose",
            ".captcha-container",
            "div[data-callback*='recaptcha']",
            ".g-recaptcha",
            "#captcha",
            "#captcha-challenge",
            "div[class*='arkose']",
        ],
    ),
    (
        AccountSafetyBarrierType.TWO_FACTOR,
        [
            "input[name='pin']",
            "input[name='verification-code']",
            "input[id*='two-step']",
            "input[id*='pin-input']",
            "input[aria-label*='verification code' i]",
            "input[aria-label*='6-digit' i]",
            "form[action*='two-step']",
            ".two-step-verification",
            "#two-step-challenge",
        ],
    ),
    (
        AccountSafetyBarrierType.CHECKPOINT,
        [
            "#checkpoint-challenge",
            ".checkpoint-challenge",
            "form[action*='/checkpoint/']",
            "#challenge-form",
            ".security-challenge",
            "#security-verification-card",
            "div[data-control-name='security_check']",
            "div[class*='checkpoint-challenge']",
        ],
    ),
    (
        AccountSafetyBarrierType.RESTRICTION,
        [
            ".account-restricted",
            "#account-restricted-card",
            ".restriction-notice",
            "div[data-test-account-restriction]",
            "div[class*='account-restricted']",
        ],
    ),
    (
        AccountSafetyBarrierType.RATE_LIMIT,
        [
            ".too-many-requests",
            "#rate-limit-error",
            "div[data-test-rate-limit]",
        ],
    ),
    (
        AccountSafetyBarrierType.AUTHWALL,
        [
            ".authwall-join-form",
            "#login-submit",
            ".sign-in-form",
            "form[action*='/uas/login-submit']",
            "form.login__form",
        ],
    ),
]

# High-confidence platform alert texts / banners
PLATFORM_MESSAGES: list[tuple[AccountSafetyBarrierType, list[str]]] = [
    (
        AccountSafetyBarrierType.TWO_FACTOR,
        [
            "enter the 6-digit code",
            "enter verification code",
            "we sent a verification code",
            "check your phone for a code",
            "check your authenticator app",
            "two-step verification required",
            "two-factor authentication",
        ],
    ),
    (
        AccountSafetyBarrierType.CAPTCHA,
        [
            "please verify that you are a human",
            "solve this challenge to continue",
            "quick security check: please solve the puzzle",
            "solve the puzzle",
            "let's make sure you're not a bot",
            "please solve the challenge",
        ],
    ),
    (
        AccountSafetyBarrierType.CHECKPOINT,
        [
            "let's do a quick security check",
            "help us keep your account safe",
            "verify your identity",
            "security verification required",
            "we've noticed unusual activity",
            "unusual activity detected",
        ],
    ),
    (
        AccountSafetyBarrierType.RESTRICTION,
        [
            "your account has been temporarily restricted",
            "your account has been restricted",
            "access to your account has been restricted",
            "account has been restricted",
            "we've temporarily restricted your account",
        ],
    ),
    (
        AccountSafetyBarrierType.RATE_LIMIT,
        [
            "http 429 too many requests",
            "too many requests. please try again later",
            "you've reached the request limit",
            "rate limit exceeded",
            "too many requests",
        ],
    ),
    (
        AccountSafetyBarrierType.AUTHWALL,
        [
            "please sign in to view this page",
            "sign in to see who's hiring",
            "join linkedin or sign in",
            "join or sign in to find your next job",
        ],
    ),
]


def path_matches_route_prefix(path: str, prefix: str) -> bool:
    """Return True if path matches prefix with valid segment boundaries.

    Prevents partial slug matches such as '/jobs/view/checkpoint-specialist'
    from matching prefix '/checkpoint'.
    """
    clean_path = "/" + path.strip("/") if path else "/"
    clean_prefix = "/" + prefix.strip("/") if prefix else "/"
    if clean_path == clean_prefix:
        return True
    if clean_path.startswith(clean_prefix + "/"):
        return True
    return False


def sanitize_url_for_evidence(url: str, max_length: int = 160) -> str:
    """Return a safe, redacted, length-bounded URL for logging and state evidence.

    Strips userinfo, query parameters, fragments, and credentials.
    Masks opaque payload segments (OTPs, challenge tokens, hashes) on safety routes.
    Enforces http/https protocol and bounds length.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            return "https://www.linkedin.com"

        netloc = parsed.netloc or ""
        # Strip userinfo (username:password@)
        if "@" in netloc:
            netloc = netloc.split("@")[-1]

        raw_path = parsed.path or "/"
        segments = [s for s in raw_path.split("/") if s]
        sanitized_segments: list[str] = []

        is_challenge_route = any(
            path_matches_route_prefix(raw_path, p)
            for p in (
                "/checkpoint",
                "/security",
                "/identity/restricted",
                "/feed/restriction",
                "/arkose",
                "/funcaptcha",
            )
        )

        for seg in segments:
            seg_lower = seg.lower()
            if is_challenge_route and seg_lower not in KNOWN_SAFE_PATH_SEGMENTS:
                # Opaque challenge payload, OTP, secret token, or hash -> redact
                if not sanitized_segments or sanitized_segments[-1] != "[REDACTED]":
                    sanitized_segments.append("[REDACTED]")
            else:
                sanitized_segments.append(seg)

        clean_path = "/" + "/".join(sanitized_segments) if sanitized_segments else "/"
        safe_url = f"{scheme}://{netloc}{clean_path}"
        redacted = redact_string(safe_url)
        if len(redacted) > max_length:
            return redacted[: max_length - 3] + "..."
        return redacted
    except Exception:
        return "https://www.linkedin.com"


def sanitize_evidence_string(text: str, max_length: int = 160) -> str:
    """Return length-bounded, redacted evidence string."""
    if not text:
        return ""
    redacted = redact_string(text)
    masked = re.sub(r"\b\d{4,8}\b", "[REDACTED_CODE]", redacted)
    collapsed = " ".join(masked.split())
    if len(collapsed) > max_length:
        return collapsed[: max_length - 3] + "..."
    return collapsed


@dataclass(frozen=True)
class AccountSafetyDetection:
    """Immutable detection result for LinkedIn account safety barriers."""

    detected: bool
    barrier_type: Optional[AccountSafetyBarrierType] = None
    reason: Optional[str] = None
    stage: Optional[str] = None
    url: Optional[str] = None
    detected_at: Optional[str] = None
    evidence: Optional[str] = None
    resume_instructions: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert detection to a JSON-serializable dict."""
        return {
            "detected": self.detected,
            "barrier_type": self.barrier_type.value if self.barrier_type else None,
            "reason": self.reason,
            "stage": self.stage,
            "url": self.url,
            "detected_at": self.detected_at,
            "evidence": self.evidence,
            "resume_instructions": self.resume_instructions,
        }


class AccountSafetyBarrierError(Exception):
    """Exception raised when an account-safety barrier is detected.

    Must not be caught by generic retry or infrastructure failure handlers.
    """

    def __init__(self, detection: AccountSafetyDetection):
        self.detection = detection
        barrier = detection.barrier_type.value if detection.barrier_type else "barrier"
        msg = f"LinkedIn account-safety {barrier} detected at stage '{detection.stage}': {detection.reason}"
        super().__init__(msg)


def is_linkedin_host(hostname: str) -> bool:
    """Return True if hostname is linkedin.com or a subdomain."""
    h = hostname.lower()
    return h == "linkedin.com" or h.endswith(".linkedin.com")


def classify_url(url: str) -> Optional[tuple[AccountSafetyBarrierType, str]]:
    """Origin- and segment-boundary aware classifier for account safety barriers.

    Strictly separates LinkedIn path routing from external CAPTCHA providers,
    and ignores query parameters and fragments to avoid false positives.
    """
    if not url:
        return None

    try:
        parsed = urlparse(url.strip())
    except Exception:
        return None

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return None

    hostname = (parsed.hostname or "").lower()
    path = (parsed.path or "/").lower()

    # 1. External CAPTCHA provider hosts
    for host in EXTERNAL_CAPTCHA_HOSTS:
        if hostname == host or hostname.endswith(f".{host}"):
            return (
                AccountSafetyBarrierType.CAPTCHA,
                f"External CAPTCHA provider origin detected: {hostname}",
            )

    # Google reCAPTCHA endpoints
    if (hostname == "google.com" or hostname.endswith(".google.com")) and path_matches_route_prefix(
        path, "/recaptcha"
    ):
        return (
            AccountSafetyBarrierType.CAPTCHA,
            "Google reCAPTCHA challenge endpoint detected",
        )

    # 2. LinkedIn endpoints (checked strictly on linkedin.com origins and normalized path segments)
    if is_linkedin_host(hostname):
        # Captcha endpoints
        for marker in LINKEDIN_CAPTCHA_PATH_PREFIXES:
            if path_matches_route_prefix(path, marker):
                return (
                    AccountSafetyBarrierType.CAPTCHA,
                    f"LinkedIn CAPTCHA endpoint: {marker}",
                )

        # 2FA endpoints
        for marker in LINKEDIN_2FA_PATH_PREFIXES:
            if path_matches_route_prefix(path, marker):
                return (
                    AccountSafetyBarrierType.TWO_FACTOR,
                    f"LinkedIn 2FA / OTP verification endpoint: {marker}",
                )

        # Checkpoint endpoints
        for marker in LINKEDIN_CHECKPOINT_PATH_PREFIXES:
            if path_matches_route_prefix(path, marker):
                return (
                    AccountSafetyBarrierType.CHECKPOINT,
                    f"LinkedIn security checkpoint endpoint: {marker}",
                )

        # Restriction endpoints
        for marker in LINKEDIN_RESTRICTION_PATH_PREFIXES:
            if path_matches_route_prefix(path, marker):
                return (
                    AccountSafetyBarrierType.RESTRICTION,
                    f"LinkedIn account restriction endpoint: {marker}",
                )

        # Rate limit endpoints
        for marker in LINKEDIN_RATE_LIMIT_PATH_PREFIXES:
            if path_matches_route_prefix(path, marker):
                return (
                    AccountSafetyBarrierType.RATE_LIMIT,
                    f"LinkedIn rate-limit endpoint: {marker}",
                )

        # Authwall endpoints
        for marker in LINKEDIN_AUTHWALL_PATH_PREFIXES:
            if path_matches_route_prefix(path, marker):
                return (
                    AccountSafetyBarrierType.AUTHWALL,
                    f"LinkedIn authwall / login redirect endpoint: {marker}",
                )

    return None


def classify_http_status(
    http_status: Optional[int],
    url: Optional[str] = None,
    page_evidence: Optional[str] = None,
) -> Optional[tuple[AccountSafetyBarrierType, str]]:
    """Classify account-safety conditions from HTTP response status codes.

    Detects 429 rate limits unconditionally.
    Classifies 403 ONLY when combined with strong restriction/challenge evidence
    (known safety barrier URL or explicit restriction/challenge page text).
    Does NOT classify a plain 403 on ordinary job URLs as an account restriction.
    """
    if http_status is None:
        return None

    if http_status == 429:
        return (
            AccountSafetyBarrierType.RATE_LIMIT,
            "HTTP 429 Too Many Requests response received from platform",
        )

    if http_status == 403:
        has_barrier_url = False
        if url:
            url_match = classify_url(url)
            if url_match is not None:
                has_barrier_url = True

        has_barrier_evidence = False
        if page_evidence:
            ev_match = classify_platform_message(page_evidence)
            if ev_match is not None:
                has_barrier_evidence = True
            else:
                ev_lower = page_evidence.lower()
                if any(
                    w in ev_lower
                    for w in (
                        "account restricted",
                        "account suspended",
                        "temporary restriction",
                        "security checkpoint",
                        "verify your identity",
                        "security challenge",
                    )
                ):
                    has_barrier_evidence = True

        if has_barrier_url or has_barrier_evidence:
            return (
                AccountSafetyBarrierType.RESTRICTION,
                "HTTP 403 Forbidden platform restriction response received with barrier evidence",
            )

    return None


def classify_title(title: str) -> Optional[tuple[AccountSafetyBarrierType, str]]:
    """Pure classifier: identify account safety barrier from page title."""
    if not title:
        return None
    title_lower = " ".join(title.lower().split()).strip()

    # Exact or normalized title matching for LinkedIn platform challenge pages
    for barrier_type, patterns in TITLE_PATTERNS:
        for pattern in patterns:
            if (
                title_lower == pattern
                or title_lower.startswith(pattern)
                or title_lower.endswith(pattern)
                or title_lower == f"{pattern} | linkedin"
                or title_lower.startswith(f"{pattern} |")
                or title_lower.endswith(f"| {pattern}")
            ):
                return (
                    barrier_type,
                    f"Page title contains barrier phrase: '{pattern}'",
                )

    return None


def classify_platform_message(
    text: str,
) -> Optional[tuple[AccountSafetyBarrierType, str]]:
    """Pure classifier: identify account safety barrier from platform alert text."""
    if not text:
        return None
    text_lower = " ".join(text.lower().split()).strip()

    for barrier_type, phrases in PLATFORM_MESSAGES:
        for phrase in phrases:
            if (
                text_lower == phrase
                or phrase in text_lower
                or text_lower.startswith(phrase)
                or text_lower.endswith(phrase)
            ):
                return (
                    barrier_type,
                    f"Platform alert contains: '{phrase}'",
                )

    return None


async def inspect_page_account_safety(
    page: Any,
    stage: str = "unknown",
    http_status: Optional[int] = None,
    fail_closed: bool = True,
) -> AccountSafetyDetection:
    """Async inspection of a Playwright page for account-safety barriers.

    Checks HTTP response status, origin-aware URL, page title, DOM selectors,
    and top-level platform alerts. Fails closed by default on inspection errors.
    """
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    current_url = ""
    safe_url = "https://www.linkedin.com"

    try:
        raw_url = getattr(page, "url", "")
        current_url = raw_url if isinstance(raw_url, str) and not raw_url.startswith("<") else ""
        safe_url = sanitize_url_for_evidence(current_url)

        # 1. HTTP status code check
        status_match = classify_http_status(http_status, url=current_url)
        if status_match:
            btype, reason = status_match
            evidence = sanitize_evidence_string(f"HTTP Status {http_status} at {safe_url}")
            return AccountSafetyDetection(
                detected=True,
                barrier_type=btype,
                reason=reason,
                stage=stage,
                url=safe_url,
                detected_at=now_iso,
                evidence=evidence,
                resume_instructions=SAFE_RESUME_INSTRUCTIONS,
            )

        # 2. URL classification (origin/path aware)
        url_match = classify_url(current_url)
        if url_match:
            btype, reason = url_match
            evidence = sanitize_evidence_string(f"Detected via URL: {safe_url}")
            return AccountSafetyDetection(
                detected=True,
                barrier_type=btype,
                reason=reason,
                stage=stage,
                url=safe_url,
                detected_at=now_iso,
                evidence=evidence,
                resume_instructions=SAFE_RESUME_INSTRUCTIONS,
            )

        # 3. Page title classification
        title = ""
        if hasattr(page, "title") and callable(page.title):
            try:
                title_res = page.title()
                raw_title = await title_res if asyncio.iscoroutine(title_res) else title_res
                title = (
                    raw_title
                    if isinstance(raw_title, str) and not raw_title.startswith("<")
                    else ""
                )
            except Exception as title_exc:
                if fail_closed:
                    raise title_exc
                title = ""

        title_match = classify_title(title)
        if title_match:
            btype, reason = title_match
            evidence = sanitize_evidence_string(f"Detected via title: {title}")
            return AccountSafetyDetection(
                detected=True,
                barrier_type=btype,
                reason=reason,
                stage=stage,
                url=safe_url,
                detected_at=now_iso,
                evidence=evidence,
                resume_instructions=SAFE_RESUME_INSTRUCTIONS,
            )

        # 4. Known safety barrier selectors
        if hasattr(page, "query_selector") and callable(page.query_selector):
            for barrier_type, selectors in SELECTOR_PATTERNS:
                for selector in selectors:
                    try:
                        elem_res = page.query_selector(selector)
                        elem = await elem_res if asyncio.iscoroutine(elem_res) else elem_res
                        if elem is not None:
                            is_vis_fn = getattr(elem, "is_visible", None)
                            if callable(is_vis_fn):
                                vis_res = is_vis_fn()
                                vis_val = await vis_res if asyncio.iscoroutine(vis_res) else vis_res
                                if vis_val is True:
                                    evidence = sanitize_evidence_string(
                                        f"Visible element matching selector: {selector}"
                                    )
                                    return AccountSafetyDetection(
                                        detected=True,
                                        barrier_type=barrier_type,
                                        reason=f"Page contains visible security selector: {selector}",
                                        stage=stage,
                                        url=safe_url,
                                        detected_at=now_iso,
                                        evidence=evidence,
                                        resume_instructions=SAFE_RESUME_INSTRUCTIONS,
                                    )
                    except Exception:
                        continue

        # 5. Top-level platform message inspection
        if hasattr(page, "evaluate") and callable(page.evaluate):
            try:
                eval_res = page.evaluate(
                    r"""() => {
                        const excluded = '.jobs-description, .jobs-description-content__text, ' +
                                         '.jobs-search__job-details, article, .job-details-jobs-unified-top-card';
                        const candidates = document.querySelectorAll(
                            'h1, h2, .error-message, .alert, .system-alert, .artdeco-inline-feedback--error, ' +
                            '[role="alert"], .checkpoint-challenge, .authwall'
                        );
                        const texts = [];
                        for (const el of candidates) {
                            if (!el.closest(excluded) && el.offsetParent !== null) {
                                const t = (el.innerText || '').trim();
                                if (t && t.length < 300) {
                                    texts.push(t);
                                }
                            }
                        }
                        return texts;
                    }"""
                )
                raw_alerts = await eval_res if asyncio.iscoroutine(eval_res) else eval_res
                alert_texts = raw_alerts if isinstance(raw_alerts, list) else []
                for alert_text in alert_texts:
                    if isinstance(alert_text, str):
                        msg_match = classify_platform_message(alert_text)
                        if msg_match:
                            btype, reason = msg_match
                            evidence = sanitize_evidence_string(f"Platform text: {alert_text}")
                            return AccountSafetyDetection(
                                detected=True,
                                barrier_type=btype,
                                reason=reason,
                                stage=stage,
                                url=safe_url,
                                detected_at=now_iso,
                                evidence=evidence,
                                resume_instructions=SAFE_RESUME_INSTRUCTIONS,
                            )
            except Exception as eval_exc:
                if fail_closed:
                    raise eval_exc

    except Exception as exc:
        if fail_closed:
            return AccountSafetyDetection(
                detected=True,
                barrier_type=AccountSafetyBarrierType.INSPECTION_UNAVAILABLE,
                reason=f"Account safety inspection unavailable ({type(exc).__name__}); pausing for account safety",
                stage=stage,
                url=safe_url,
                detected_at=now_iso,
                evidence=sanitize_evidence_string(f"Inspection error: {str(exc)}"),
                resume_instructions=SAFE_RESUME_INSTRUCTIONS,
            )

    return AccountSafetyDetection(detected=False)


async def guard_page_account_safety(
    page: Any,
    stage: str = "unknown",
    http_status: Optional[int] = None,
    fail_closed: bool = True,
) -> None:
    """Perform page account-safety inspection and raise AccountSafetyBarrierError if detected."""
    detection = await inspect_page_account_safety(
        page, stage=stage, http_status=http_status, fail_closed=fail_closed
    )
    if detection.detected:
        raise AccountSafetyBarrierError(detection)


def normalize_safety_log_payload(
    run_id: str,
    barrier_type: Optional[str] = None,
    stage: Optional[str] = None,
    reason: Optional[str] = None,
    url: Optional[str] = None,
    resume_instructions: Optional[str] = None,
) -> dict[str, str]:
    """Return a sanitized, length-bounded dict of safety barrier fields."""
    btype = sanitize_evidence_string(barrier_type or "SECURITY BARRIER", max_length=40).upper()
    return {
        "run_id": sanitize_evidence_string(run_id, max_length=64),
        "barrier_type": btype,
        "stage": sanitize_evidence_string(stage or "unknown", max_length=60),
        "reason": sanitize_evidence_string(
            reason or "Account safety challenge detected", max_length=160
        ),
        "url": sanitize_url_for_evidence(url or "https://www.linkedin.com"),
        "resume_instructions": sanitize_evidence_string(
            resume_instructions or SAFE_RESUME_INSTRUCTIONS, max_length=300
        ),
    }


def format_account_safety_notification(run_id: str, detection: AccountSafetyDetection) -> str:
    """Format a concise, redacted operator alert for console and Telegram."""
    norm = normalize_safety_log_payload(
        run_id=run_id,
        barrier_type=detection.barrier_type.value if detection.barrier_type else None,
        stage=detection.stage,
        reason=detection.reason,
        url=detection.url,
        resume_instructions=detection.resume_instructions,
    )
    lines = [
        "🛑 ACCOUNT SAFETY BARRIER DETECTED — ACTION REQUIRED",
        "",
        f"Run ID: {norm['run_id']}",
        f"Barrier Type: {norm['barrier_type']}",
        f"Stage: {norm['stage']}",
        f"URL: {norm['url']}",
        f"Reason: {norm['reason']}",
        "",
        "OPERATOR ACTION REQUIRED:",
        f"{norm['resume_instructions']}",
        "",
        "⚠️ SECURITY NOTICE: Never share OTPs, 2FA codes, passwords, or CAPTCHA answers.",
    ]
    return "\n".join(lines)
