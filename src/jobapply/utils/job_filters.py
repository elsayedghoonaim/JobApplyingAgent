"""Deterministic eligibility filters applied before job qualification."""

import re
from collections.abc import Sequence

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
_DEFAULT_TARGET_TITLE_KEYWORDS = ("Machine Learning", "ML", "MLOps", "Deep Learning")
_DEFAULT_EXCLUDED_LOCATIONS = (
    "Palestine",
    "Palastin",
    "Palastine",
    "State of Palestine",
    "Palestinian Territory",
    "Palestinian Territories",
    "Israel",
    "Israeli",
    "West Bank",
    "Gaza",
    "Gaza Strip",
    "Tel Aviv",
    "Jerusalem",
    "Haifa",
    "Beer Sheva",
    "Beersheba",
    "Ashdod",
    "Netanya",
    "Petah Tikva",
    "Rishon LeZion",
    "Ramat Gan",
    "Herzliya",
    "فلسطين",
    "إسرائيل",
    "ישראל",
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


def _normalized_keyword_pattern(keyword: str) -> re.Pattern[str] | None:
    """Build a literal, word-bounded title phrase pattern with flexible separators."""
    parts = [part for part in re.split(r"[\s\-]+", keyword.casefold().strip()) if part]
    if not parts:
        return None
    phrase = r"[\s\-/]+".join(re.escape(part) for part in parts)
    return re.compile(rf"(?<!\w){phrase}(?!\w)", re.IGNORECASE)


def is_target_position_title(
    title: str | None,
    target_keywords: Sequence[str] | None = None,
) -> bool:
    """Return whether a title contains one of the configured literal target phrases."""
    keywords = target_keywords or _DEFAULT_TARGET_TITLE_KEYWORDS
    return any(
        pattern.search(title or "") is not None
        for keyword in keywords
        if (pattern := _normalized_keyword_pattern(keyword)) is not None
    )


def get_title_exclusion_reason(
    title: str | None,
    target_keywords: Sequence[str] | None = None,
    *,
    exclude_senior_titles: bool = True,
) -> str | None:
    """Return a deterministic title-only exclusion reason."""
    if exclude_senior_titles and is_senior_position_title(title):
        return "Senior-level position excluded by user preference"
    if not is_target_position_title(title, target_keywords):
        return "Title does not match configured target keywords"
    return None


def _declared_language_is_allowed(
    language: str, allowed_languages: Sequence[str] | None = None
) -> bool:
    """Allow declarations made up only of configured languages and joiner words."""
    allowed_words = _normalized_allowed_languages(allowed_languages)
    remaining = language.casefold()
    for allowed in sorted(allowed_words, key=len, reverse=True):
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


def _normalized_allowed_languages(allowed_languages: Sequence[str] | None) -> set[str]:
    """Normalize configured language names and include known Arabic-script aliases."""
    allowed_words = {
        value.casefold().strip() for value in (allowed_languages or ("Arabic", "English")) if value
    }
    if "arabic" in allowed_words:
        allowed_words.update(word for word in _ALLOWED_LANGUAGE_WORDS if word != "english")
    return allowed_words


def find_disallowed_required_languages(
    job: dict, allowed_languages: Sequence[str] | None = None
) -> list[str]:
    """Return explicitly required languages outside the configured allowlist."""
    allowed_words = _normalized_allowed_languages(allowed_languages)
    disallowed: list[str] = []
    declared_languages = job.get("required_languages") or []
    if isinstance(declared_languages, str):
        declared_languages = [declared_languages]
    for language in declared_languages:
        value = " ".join(str(language).split()).strip()
        if not value or value.casefold() in {"none", "not specified", "n/a"}:
            continue
        if not _declared_language_is_allowed(value, allowed_languages):
            disallowed.append(value)

    description = str(job.get("description") or "")
    for pattern in _REQUIRED_LANGUAGE_PATTERNS:
        for match in pattern.finditer(description):
            language = match.group("language").title()
            if language.casefold() not in allowed_words:
                disallowed.append(language)

    for match in _REQUIRED_LANGUAGE_LIST_PATTERN.finditer(description):
        language_list = next(value for value in match.groupdict().values() if value is not None)
        for language in _LANGUAGE_NAMES:
            if language.casefold() not in allowed_words and re.search(
                rf"(?<!\w){re.escape(language)}(?!\w)", language_list, re.IGNORECASE
            ):
                disallowed.append(language.title())

    return list(dict.fromkeys(disallowed))


def get_location_exclusion_reason(
    job: dict,
    excluded_locations: Sequence[str] | None = None,
) -> str | None:
    """Return an exclusion reason when a job matches a configured blocked location."""
    blocked = (
        _DEFAULT_EXCLUDED_LOCATIONS if excluded_locations is None else tuple(excluded_locations)
    )
    patterns = [
        pattern
        for value in blocked
        if (pattern := _normalized_keyword_pattern(str(value))) is not None
    ]
    for field_name in ("location", "parsed_location"):
        location = str(job.get(field_name) or "").strip()
        if location and any(pattern.search(location) for pattern in patterns):
            return "Job location excluded by user preference"
    return None


def get_job_exclusion_reason(
    job: dict,
    target_keywords: Sequence[str] | None = None,
    *,
    exclude_senior_titles: bool = True,
    allowed_languages: Sequence[str] | None = None,
    excluded_locations: Sequence[str] | None = None,
) -> str | None:
    """Return the first deterministic reason a job must not be processed."""
    location_reason = get_location_exclusion_reason(job, excluded_locations)
    if location_reason:
        return location_reason
    title_reason = get_title_exclusion_reason(
        job.get("title"),
        target_keywords,
        exclude_senior_titles=exclude_senior_titles,
    )
    if title_reason:
        return title_reason
    disallowed_languages = find_disallowed_required_languages(job, allowed_languages)
    if disallowed_languages:
        return "Job requires language(s) outside configured allowlist: " + ", ".join(
            disallowed_languages
        )
    return None
