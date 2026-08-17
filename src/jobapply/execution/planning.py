"""Pure answer/field planning, classification, and normalization helpers."""

from dataclasses import dataclass, field

# Field patterns the agent can auto-fill
KNOWN_FIELD_PATTERNS: list[str] = [
    "phone",
    "email",
    "name",
    "first name",
    "last name",
    "city",
    "linkedin",
    "github",
    "portfolio",
    "education",
    "resume",
    "cv",
]

# Fields to auto-skip (not critical)
AUTO_SKIP_PATTERNS: list[str] = [
    "gender",
    "race",
    "ethnicity",
    "veteran",
    "disability",
    "diverse",
    "protected",
    "voluntary",
]

CHOICE_PLACEHOLDERS: tuple[str, ...] = (
    "select an option",
    "select option",
    "choose an option",
    "choose option",
    "please select",
    "-- select --",
)

CHOICE_SENTINEL_VALUES: frozenset[str] = frozenset(
    {
        "",
        "none",
        "placeholder",
        "select",
        "-- select --",
    }
)

STANDARD_TEXT_FIELD_SELECTOR: str = (
    "input:not([type]), input[type='text'], input[type='tel'], "
    "input[type='email'], input[type='number'], input[type='url'], textarea, "
    "[role='textbox'][contenteditable='true']"
)


@dataclass(frozen=True)
class RequiredFieldValidationResult:
    """Result of validating visible required form fields before navigation or submission."""

    is_valid: bool
    unresolved_fields: list[str] = field(default_factory=list)
    reason: str | None = None


def is_skip_job_reply(reply: str | None) -> bool:
    """Recognize only explicit job-skip commands and labels."""
    normalized = " ".join((reply or "").casefold().split()).strip()
    return normalized in {"/skip", "skip", "skip job", "skip this job"} or (
        normalized.startswith("/skip@") and " " not in normalized
    )


def _is_placeholder_text(text: str) -> bool:
    normalized = " ".join(text.casefold().split()).strip()
    if not normalized or normalized in CHOICE_SENTINEL_VALUES:
        return True
    return any(marker in normalized for marker in CHOICE_PLACEHOLDERS)


def choice_is_unanswered(value: str | None, visible_text: str | None = None) -> bool:
    """Recognize empty and placeholder values in native/custom choice controls."""
    v_norm = " ".join(str(value or "").casefold().split()).strip()
    t_norm = " ".join(str(visible_text or "").casefold().split()).strip()

    # 1. Both empty
    if not v_norm and not t_norm:
        return True

    # 2. Visible selected text matches a configured placeholder marker or exact sentinel
    if t_norm and _is_placeholder_text(t_norm):
        return True

    # 3. Raw value is empty, exact bounded sentinel, or matches a configured placeholder marker
    if _is_placeholder_text(v_norm):
        if not t_norm or _is_placeholder_text(t_norm):
            return True

    return False


def match_choice_index(answer: str, options: list[str]) -> int | None:
    """Return the best case-insensitive exact/containment option match."""
    normalized_answer = " ".join(answer.casefold().split())
    normalized_options = [" ".join(option.casefold().split()) for option in options]
    for index, option in enumerate(normalized_options):
        if normalized_answer == option:
            return index
    for index, option in enumerate(normalized_options):
        if (
            option
            and normalized_answer
            and (normalized_answer in option or option in normalized_answer)
        ):
            return index
    return None


def _normalized_choice_text(value: str | None) -> str:
    """Normalize question and option text for live DOM re-resolution."""
    return " ".join((value or "").casefold().split())


def is_known_field(field_label: str) -> bool:
    """Check if field can be auto-filled."""
    label_lower = field_label.lower()
    # Experience-by-skill questions need a factual, user-specific answer and
    # must go through Telegram rather than reuse a generic total-years value.
    if "years" in label_lower and "experience" in label_lower:
        return False
    return any(pattern in label_lower for pattern in KNOWN_FIELD_PATTERNS)


def is_required_field(
    required_attribute: str | None,
    aria_required: str | None,
    label: str,
) -> bool:
    """Recognize native and accessible required-field markers."""
    return (
        required_attribute is not None
        or (aria_required or "").lower() == "true"
        or "*" in label
        or "(required)" in label.lower()
    )


def is_auto_skip_field(field_label: str) -> bool:
    """Check if field should be auto-skipped."""
    label_lower = field_label.lower()
    return any(pattern in label_lower for pattern in AUTO_SKIP_PATTERNS)


def get_auto_fill_value(field_label: str, profile: dict) -> str | None:
    """Get auto-fill value from profile for known fields."""
    label_lower = field_label.lower()

    if "email" in label_lower:
        return profile.get("email")
    if "phone" in label_lower:
        return profile.get("phone")
    if "first name" in label_lower or "first_name" in label_lower:
        return profile.get("name", "").split()[0] if profile.get("name") else None
    if "last name" in label_lower or "last_name" in label_lower:
        parts = profile.get("name", "").split()
        return parts[-1] if len(parts) > 1 else None
    if "name" in label_lower:
        name_val = profile.get("name")
        return str(name_val).strip() if name_val else None
    if "linkedin" in label_lower:
        return profile.get("linkedin")
    if "github" in label_lower:
        return profile.get("github")
    if "city" in label_lower or "location" in label_lower:
        return profile.get("location")
    if "years" in label_lower and "experience" in label_lower:
        return profile.get("form_defaults", {}).get("years_of_experience")
    if "education" in label_lower:
        return profile.get("form_defaults", {}).get("highest_education")

    return None
