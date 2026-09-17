"""Durable, dated candidate facts learned from Telegram form answers."""

from __future__ import annotations

import hashlib
import inspect
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from jobapply.settings import get_settings
from jobapply.utils.mongo import MongoClientManager
from jobapply.utils.observability import log_event

SENSITIVE_QUESTION_MARKERS = (
    "password",
    "passcode",
    "one-time code",
    "one time code",
    "otp",
    "captcha",
    "bank account",
    "credit card",
    "social security",
    "national id",
    "passport number",
    "driver license number",
    "date of birth",
    "birth date",
    "gender",
    "race",
    "ethnicity",
    "veteran",
    "disability",
    "sexual orientation",
    "religion",
)


@dataclass(frozen=True)
class FactIdentity:
    """Stable identity and reuse scope for one candidate fact."""

    fact_key: str
    scope: str
    remember: bool = True

    @property
    def fact_id(self) -> str:
        raw = f"{self.fact_key}|{self.scope}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CandidateFact:
    """One current candidate fact returned by the repository."""

    fact_id: str
    fact_key: str
    scope: str
    value: str
    original_question: str
    confirmed_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at <= (now or datetime.now(timezone.utc))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s.+#/-]", " ", normalized)
    return " ".join(normalized.split())


def _slug(value: str, fallback: str = "general") -> str:
    value = _normalize_text(value).replace("+", " plus ").replace("#", " sharp ")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:80] or fallback


def _location_scope(job: dict[str, Any]) -> str:
    location = str(job.get("location") or "").strip()
    return f"location:{_slug(location)}" if location else "global"


def classify_candidate_fact(question: str, job: dict[str, Any] | None = None) -> FactIdentity:
    """Map a form question to a conservative, deterministic candidate-fact identity.

    Unknown questions use a normalized-question fingerprint. This safely remembers
    repeated wording without guessing that two semantically different questions are
    equivalent. Common facts receive semantic keys so small wording changes can reuse
    the same confirmed answer.
    """

    job = job or {}
    normalized = _normalize_text(question)
    if not normalized or any(marker in normalized for marker in SENSITIVE_QUESTION_MARKERS):
        return FactIdentity("sensitive.unstored", "none", remember=False)

    location_scope = _location_scope(job)
    if "sponsor" in normalized or "sponsorship" in normalized:
        return FactIdentity("employment.requires_sponsorship", location_scope)
    if "authorized" in normalized and ("work" in normalized or "employment" in normalized):
        return FactIdentity("employment.work_authorization", location_scope)
    if "relocat" in normalized:
        return FactIdentity("employment.willing_to_relocate", location_scope)
    if "notice period" in normalized or "available to start" in normalized:
        return FactIdentity("employment.availability", "global")
    if "education" in normalized or "degree" in normalized:
        return FactIdentity("education.highest", "global")

    if "year" in normalized and "experience" in normalized:
        skill = re.sub(
            r"\b(?:how many|number of|years?|of|professional|work|working|experience|do you have|with|in)\b",
            " ",
            normalized,
        )
        return FactIdentity(f"experience.{_slug(skill)}.years", "global")

    if any(word in normalized for word in ("salary", "compensation", "pay", "wage")):
        title = _slug(str(job.get("title") or "role"))
        return FactIdentity("employment.compensation_expectation", f"{location_scope}|role:{title}")

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return FactIdentity(f"question.{digest}", "exact")


def compatible_saved_value(value: str, options: list[str] | None) -> str | None:
    """Return the exact current form value when a saved answer is compatible."""

    clean_options = [str(option).strip() for option in (options or []) if str(option).strip()]
    if not clean_options:
        return str(value).strip() or None
    normalized_value = " ".join(str(value).casefold().split())
    normalized_options = [" ".join(option.casefold().split()) for option in clean_options]
    index = next(
        (i for i, option in enumerate(normalized_options) if option == normalized_value),
        None,
    )
    if index is None:
        index = next(
            (
                i
                for i, option in enumerate(normalized_options)
                if option
                and normalized_value
                and (normalized_value in option or option in normalized_value)
            ),
            None,
        )
    return clean_options[index] if index is not None else None


def _as_utc_datetime(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("candidate fact timestamp is invalid")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class CandidateFactsRepository:
    """MongoDB repository for current facts and bounded confirmation history."""

    def __init__(self, collection: Any | None = None, settings: Any | None = None):
        self.settings = settings or get_settings()
        if collection is None:
            client = MongoClientManager.get_client()
            collection = client[self.settings.mongodb_db][self.settings.candidate_facts_collection]
        self.collection = collection
        self._indexes_ready = False

    async def ensure_indexes(self) -> None:
        if self._indexes_ready or not hasattr(self.collection, "create_index"):
            self._indexes_ready = True
            return
        result = self.collection.create_index("fact_id", unique=True)
        if inspect.isawaitable(result):
            await result
        result = self.collection.create_index([("expires_at", 1), ("fact_key", 1)])
        if inspect.isawaitable(result):
            await result
        self._indexes_ready = True

    async def get(self, identity: FactIdentity) -> CandidateFact | None:
        if not identity.remember:
            return None
        await self.ensure_indexes()
        result = self.collection.find_one({"fact_id": identity.fact_id, "status": "active"})
        document = await result if inspect.isawaitable(result) else result
        if not document:
            return None
        try:
            return CandidateFact(
                fact_id=str(document["fact_id"]),
                fact_key=str(document["fact_key"]),
                scope=str(document["scope"]),
                value=str(document["value"]),
                original_question=str(document.get("original_question") or ""),
                confirmed_at=_as_utc_datetime(document["confirmed_at"]),
                expires_at=_as_utc_datetime(document["expires_at"]),
            )
        except (KeyError, TypeError, ValueError):
            log_event(
                "warning",
                "candidate_facts.malformed",
                "Stored candidate fact was malformed and will not be reused.",
                node="candidate_facts",
            )
            return None

    async def remember(
        self,
        identity: FactIdentity,
        value: str,
        question: str,
        *,
        now: datetime | None = None,
        source: str = "telegram",
    ) -> None:
        if not identity.remember or not str(value).strip():
            return
        await self.ensure_indexes()
        confirmed_at = now or datetime.now(timezone.utc)
        confirmed_at = (
            confirmed_at.replace(tzinfo=timezone.utc)
            if confirmed_at.tzinfo is None
            else confirmed_at.astimezone(timezone.utc)
        )
        expires_at = confirmed_at + timedelta(days=self.settings.candidate_fact_validity_days)
        history_entry = {
            "value": str(value).strip(),
            "question": str(question)[:1000],
            "source": source,
            "confirmed_at": confirmed_at,
            "expires_at": expires_at,
        }
        update = {
            "$set": {
                "fact_key": identity.fact_key,
                "scope": identity.scope,
                "value": str(value).strip(),
                "original_question": str(question)[:1000],
                "source": source,
                "confirmed_at": confirmed_at,
                "expires_at": expires_at,
                "updated_at": confirmed_at,
                "status": "active",
            },
            "$setOnInsert": {"fact_id": identity.fact_id, "created_at": confirmed_at},
            "$inc": {"revision": 1},
            "$push": {
                "history": {
                    "$each": [history_entry],
                    "$slice": -self.settings.candidate_fact_history_limit,
                }
            },
        }
        result = self.collection.update_one({"fact_id": identity.fact_id}, update, upsert=True)
        if inspect.isawaitable(result):
            await result


async def resolve_answer_from_memory(
    question: str,
    job: dict[str, Any],
    options: list[str] | None,
    repository: CandidateFactsRepository,
) -> tuple[FactIdentity, CandidateFact | None, str | None]:
    """Return identity, stored record, and reusable answer for one question."""

    identity = classify_candidate_fact(question, job)
    if not identity.remember:
        return identity, None, None
    fact = await repository.get(identity)
    if fact is None:
        return identity, None, None
    return identity, fact, compatible_saved_value(fact.value, options)
