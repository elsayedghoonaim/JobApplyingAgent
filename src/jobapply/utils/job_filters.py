"""Deterministic eligibility filters applied before job qualification."""

import re

_SENIOR_TITLE_PATTERN = re.compile(
    r"\b(?:senior|sr\.?|lead|principal|staff|manager|director|head|chief|"
    r"vice\s+president|vp|architect|ceo|cto|cio|cfo|coo)\b",
    re.IGNORECASE,
)
_MIXED_MID_SENIOR_PATTERN = re.compile(
    r"\bmid(?:dle)?(?:\s*[- ]\s*level)?\s*"
    r"(?:(?:[-–—/]|\bto\b)\s*)?(?:senior|sr\.?)\b",
    re.IGNORECASE,
)
_MACHINE_LEARNING_TITLE_PATTERN = re.compile(
    r"\bmachine[\s-]+learning\b|\bml(?:ops)?\b|\bdeep[\s-]+learning\b",
    re.IGNORECASE,
)

_ALLOWED_LANGUAGE_WORDS = {
    "arabic",
    "english",
    "العربية",
    "الانجليزية",
    "الإنجليزية",
}

_LANGUAGE_NAMES = (
    "afrikaans",
    "albanian",
    "amharic",
    "armenian",
    "azerbaijani",
    "basque",
    "bengali",
    "bosnian",
    "bulgarian",
    "burmese",
    "catalan",
    "cantonese",
    "chinese",
    "croatian",
    "czech",
    "danish",
    "dutch",
    "estonian",
    "farsi",
    "finnish",
    "french",
    "georgian",
    "german",
    "greek",
    "gujarati",
    "hebrew",
    "hindi",
    "hungarian",
    "icelandic",
    "indonesian",
    "italian",
    "japanese",
    "kannada",
    "korean",
    "latvian",
    "lithuanian",
    "malay",
    "malayalam",
    "mandarin",
    "marathi",
    "norwegian",
    "persian",
    "polish",
    "portuguese",
    "punjabi",
    "romanian",
    "russian",
    "serbian",
    "slovak",
    "slovenian",
    "spanish",
    "swahili",
    "swedish",
    "tamil",
    "telugu",
    "thai",
    "turkish",
    "ukrainian",
    "urdu",
    "vietnamese",
)
_LANGUAGE_PATTERN = "|".join(
    re.escape(language) for language in sorted(_LANGUAGE_NAMES, key=len, reverse=True)
)
_REQUIRED_LANGUAGE_PATTERNS = (
    re.compile(
        rf"\b(?:must\s+(?:be\s+able\s+to\s+)?(?:speak|read|write|communicate\s+in)|"
        rf"fluent|proficient|native|bilingual)\s+(?:in\s+)?(?P<language>{_LANGUAGE_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:fluency|proficiency)\s+in\s+(?P<language>{_LANGUAGE_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<language>{_LANGUAGE_PATTERN})(?:\s+language)?\s+"
        rf"(?:is\s+)?(?:required|mandatory|essential)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:required|mandatory)\s+(?:spoken\s+|written\s+)?languages?\s*[:\-]\s*"
        rf"(?P<language>{_LANGUAGE_PATTERN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<language>{_LANGUAGE_PATTERN})[-\s]+speaking\b",
        re.IGNORECASE,
    ),
)
_REQUIRED_LANGUAGE_LIST_PATTERN = re.compile(
    r"\b(?:fluent|proficient)\s+in\s+(?P<languages>[^.;\n]{1,120})|"
    r"\b(?:fluency|proficiency)\s+in\s+(?P<languages_noun>[^.;\n]{1,120})|"
    r"\b(?:required|mandatory)\s+languages?\s*[:\-]\s*(?P<languages_label>[^.;\n]{1,120})",
    re.IGNORECASE,
)


def is_senior_position_title(title: str | None) -> bool:
    """Exclude pure senior titles while allowing mixed mid-to-senior ranges."""
    normalized = " ".join((title or "").casefold().split())
    normalized = _MIXED_MID_SENIOR_PATTERN.sub("mid", normalized)
    # "Lead generation" describes a sales activity rather than seniority.
    normalized = normalized.replace("lead generation", "")
    return _SENIOR_TITLE_PATTERN.search(normalized) is not None


def is_machine_learning_position_title(title: str | None) -> bool:
    """Return whether a title explicitly identifies a machine-learning role."""
    return _MACHINE_LEARNING_TITLE_PATTERN.search(title or "") is not None


def get_title_exclusion_reason(title: str | None) -> str | None:
    """Return a deterministic title-only exclusion reason."""
    if is_senior_position_title(title):
        return "Senior-level position excluded by user preference"
    if not is_machine_learning_position_title(title):
        return "Non-machine-learning title excluded by user preference"
    return None


def _declared_language_is_allowed(language: str) -> bool:
    """Allow declarations made up only of Arabic, English, and joiner words."""
    remaining = language.casefold()
    for allowed in sorted(_ALLOWED_LANGUAGE_WORDS, key=len, reverse=True):
        remaining = re.sub(rf"(?<!\w){re.escape(allowed)}(?!\w)", " ", remaining)
    remaining = re.sub(
        r"\b(?:and|or|either|both|language|languages|required|mandatory|native|"
        r"fluent|proficient|proficiency|advanced|business|working|level|"
        r"a1|a2|b1|b2|c1|c2)\b",
        " ",
        remaining,
    )
    remaining = re.sub(r"[^\w\u0600-\u06ff]+", "", remaining)
    return not remaining


def find_disallowed_required_languages(job: dict) -> list[str]:
    """Return explicitly required languages other than Arabic and English."""
    disallowed: list[str] = []
    declared_languages = job.get("required_languages") or []
    if isinstance(declared_languages, str):
        declared_languages = [declared_languages]
    for language in declared_languages:
        value = " ".join(str(language).split()).strip()
        if not value or value.casefold() in {"none", "not specified", "n/a"}:
            continue
        if not _declared_language_is_allowed(value):
            disallowed.append(value)

    description = str(job.get("description") or "")
    for pattern in _REQUIRED_LANGUAGE_PATTERNS:
        for match in pattern.finditer(description):
            language = match.group("language").title()
            if language.casefold() not in _ALLOWED_LANGUAGE_WORDS:
                disallowed.append(language)

    for match in _REQUIRED_LANGUAGE_LIST_PATTERN.finditer(description):
        language_list = next(value for value in match.groupdict().values() if value is not None)
        for language in _LANGUAGE_NAMES:
            if re.search(rf"(?<!\w){re.escape(language)}(?!\w)", language_list, re.IGNORECASE):
                disallowed.append(language.title())

    return list(dict.fromkeys(disallowed))


def get_job_exclusion_reason(job: dict) -> str | None:
    """Return the first deterministic reason a job must not be processed."""
    title_reason = get_title_exclusion_reason(job.get("title"))
    if title_reason:
        return title_reason
    disallowed_languages = find_disallowed_required_languages(job)
    if disallowed_languages:
        return "Job requires language(s) outside Arabic and English: " + ", ".join(
            disallowed_languages
        )
    return None
