"""Centralized secret redaction and credential hygiene utilities."""

import os
import re
from typing import Any, Mapping, Optional, Sequence, Set

_REDACTED_REPLACEMENT = "[REDACTED]"

# Regex patterns for credential exposure
_GOOGLE_API_KEY_QUERY_RE = re.compile(r"(?i)([?&]key=)(?:AIza[0-9A-Za-z_-]{20,}|[^&\s\"'>]+)")
_TELEGRAM_BOT_TOKEN_URL_RE = re.compile(
    r"(?i)((?:https?://)?api\.telegram\.org/bot)([0-9]+:[A-Za-z0-9_-]+)"
)
_BEARER_AUTH_RE = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9_.\-~+/=]{8,})")
_HEADER_AUTH_RE = re.compile(
    r"(?i)((?:authorization|x-goog-api-key|proxy-authorization):\s*)([^\r\n]+)"
)
_GENERIC_URL_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:api_?key|bot_token|token|secret|password|auth|access_token)=)([^&\s\"'>]+)"
)
_STANDALONE_GOOGLE_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")
_STANDALONE_TELEGRAM_TOKEN_RE = re.compile(r"\b[0-9]{8,12}:[A-Za-z0-9_-]{20,}\b")
_MONGODB_SCHEME_RE = re.compile(r"(?i)\b(mongodb(?:\+srv)?://)([^\s\"'<>/?#]+)([/#'?][^\s\"'<>]*)?")
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"client[_-]?secret|bot[_-]?token)\b(\s*[=:]\s*)([^\s&\"']+)"
)

# Exact normalized sensitive key names
_EXACT_SENSITIVE_KEYS: Set[str] = {
    "api_key",
    "apikey",
    "google_api_key",
    "telegram_bot_token",
    "bot_token",
    "token",
    "secret",
    "client_secret",
    "password",
    "pass",
    "passwd",
    "auth",
    "authorization",
    "x_goog_api_key",
    "cookie",
    "cookies",
    "proxy_url",
    "mongodb_url",
    "access_token",
    "refresh_token",
    "private_key",
    "secret_key",
    "app_secret",
}


def _redact_mongodb_uri(text: str) -> str:
    """Scrub entire userinfo credentials from MongoDB connection URIs."""

    def replace_mongo(match: re.Match) -> str:
        scheme = match.group(1)
        authority = match.group(2)
        rest = match.group(3) or ""

        if "@" in authority:
            _, host_part = authority.rsplit("@", 1)
            return f"{scheme}{_REDACTED_REPLACEMENT}@{host_part}{rest}"
        return match.group(0)

    return _MONGODB_SCHEME_RE.sub(replace_mongo, text)


def _get_dynamic_secret_literals() -> list[str]:
    """Retrieve non-empty configured secret strings for exact matching."""
    secrets = set()
    env_keys = [
        "GOOGLE_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "LANGSMITH_API_KEY",
        "MONGODB_URL",
    ]
    for key in env_keys:
        val = os.getenv(key)
        if val and isinstance(val, str) and len(val.strip()) >= 6:
            val_clean = val.strip()
            if "://" in val_clean and "@" in val_clean:
                try:
                    after_scheme = val_clean.split("://", 1)[1]
                    userinfo = after_scheme.split("@", 1)[0]
                    if ":" in userinfo:
                        password = userinfo.split(":", 1)[1]
                        if len(password) >= 4:
                            secrets.add(password)
                except Exception:
                    pass
            secrets.add(val_clean)

    return sorted(secrets, key=len, reverse=True)


# Credential-bearing key suffixes (component forms such as database_password).
# Suffix matching avoids broad substring rules that would redact harmless
# fields like token_count or password_policy.
_SENSITIVE_KEY_SUFFIXES = (
    "_password",
    "_passwd",
    "_pwd",
    "_secret",
    "_api_key",
    "_access_token",
    "_auth_token",
    "_bot_token",
    "_client_secret",
    "_private_key",
    "_secret_key",
)


def is_sensitive_key(key: str) -> bool:
    """Return True when a normalized mapping key denotes sensitive credentials.

    Recognizes exact credential key names plus component/suffix forms such as
    ``database_password`` or ``service_access_token``, while leaving harmless
    keys like ``token_count`` or ``password_policy`` untouched.
    """
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _EXACT_SENSITIVE_KEYS:
        return True
    return any(normalized.endswith(suffix) for suffix in _SENSITIVE_KEY_SUFFIXES)


def redact_string(text: str, extra_secrets: Optional[Sequence[str]] = None) -> str:
    """Redact secrets from a string idempotently.

    Args:
        text: Untrusted or sensitive string.
        extra_secrets: Optional list of additional literal secrets to scrub.

    Returns:
        Redacted string with sensitive values replaced by [REDACTED].
    """
    if not isinstance(text, str) or not text:
        return text

    result = text

    # Redact MongoDB URIs first to completely eliminate credentials
    result = _redact_mongodb_uri(result)

    # Redact credential-style assignments (password=..., secret: ..., etc.)
    result = _CREDENTIAL_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{_REDACTED_REPLACEMENT}", result
    )

    # Redact Google API key in query strings
    result = _GOOGLE_API_KEY_QUERY_RE.sub(rf"\g<1>{_REDACTED_REPLACEMENT}", result)

    # Redact Telegram bot token in URLs
    result = _TELEGRAM_BOT_TOKEN_URL_RE.sub(rf"\g<1>{_REDACTED_REPLACEMENT}", result)

    # Redact Bearer tokens
    result = _BEARER_AUTH_RE.sub(rf"\g<1>{_REDACTED_REPLACEMENT}", result)

    # Redact Header credentials
    result = _HEADER_AUTH_RE.sub(rf"\g<1>{_REDACTED_REPLACEMENT}", result)

    # Redact generic query parameter secrets
    result = _GENERIC_URL_SECRET_QUERY_RE.sub(rf"\g<1>{_REDACTED_REPLACEMENT}", result)

    # Redact standalone Google API keys
    result = _STANDALONE_GOOGLE_KEY_RE.sub(_REDACTED_REPLACEMENT, result)

    # Redact standalone Telegram tokens
    result = _STANDALONE_TELEGRAM_TOKEN_RE.sub(_REDACTED_REPLACEMENT, result)

    # Redact configured secret literals
    all_secrets = _get_dynamic_secret_literals()
    if extra_secrets:
        all_secrets = sorted(
            set(all_secrets) | {s for s in extra_secrets if isinstance(s, str) and len(s) >= 4},
            key=len,
            reverse=True,
        )

    for secret in all_secrets:
        if secret and secret in result and secret != _REDACTED_REPLACEMENT:
            result = result.replace(secret, _REDACTED_REPLACEMENT)

    return result


def redact_data(data: Any, extra_secrets: Optional[Sequence[str]] = None) -> Any:
    """Recursively scrub sensitive keys and secret values from arbitrary data structures.

    If a mapping key is in _EXACT_SENSITIVE_KEYS, the entire value (scalar, dict, list,
    tuple, set) is replaced with [REDACTED].

    Args:
        data: Arbitrary object (metadata dict, list, error message, etc.).
        extra_secrets: Optional additional secret strings to redact.

    Returns:
        New data structure with secrets redacted.
    """
    if isinstance(data, str):
        return redact_string(data, extra_secrets)

    if isinstance(data, Exception):
        return redact_string(str(data), extra_secrets)

    if isinstance(data, Mapping):
        redacted_dict = {}
        for key, value in data.items():
            key_norm = str(key).strip().lower().replace("-", "_")
            if key_norm in _EXACT_SENSITIVE_KEYS:
                redacted_dict[key] = _REDACTED_REPLACEMENT
            else:
                redacted_dict[key] = redact_data(value, extra_secrets)
        return redacted_dict

    if isinstance(data, list):
        return [redact_data(item, extra_secrets) for item in data]

    if isinstance(data, tuple):
        return tuple(redact_data(item, extra_secrets) for item in data)

    if isinstance(data, set):
        return {redact_data(item, extra_secrets) for item in data}

    return data


def redact_exception(exc: Exception) -> str:
    """Convert an exception to a sanitized string with all credentials redacted."""
    return redact_string(str(exc))
