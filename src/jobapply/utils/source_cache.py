"""Process-local, bounded, concurrency-safe source file cache for user profile and resume data."""

import copy
import hashlib
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import yaml

from jobapply.settings import get_settings


@dataclass(frozen=True)
class _CachedEntry:
    mtime_ns: int
    size: int
    digest: str
    raw_text: str
    parsed_yaml: Optional[dict[str, Any]] = None


class SourceCache:
    """Process-local, concurrency-safe LRU cache for profile.yaml and resume.md."""

    _global_instance: Optional["SourceCache"] = None
    _global_lock: threading.Lock = threading.Lock()

    def __init__(self, max_entries: int = 32):
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _CachedEntry] = OrderedDict()
        self._max_entries = max_entries

    @classmethod
    def _get_global(cls) -> "SourceCache":
        with cls._global_lock:
            if cls._global_instance is None:
                cls._global_instance = SourceCache(max_entries=32)
            return cls._global_instance

    @classmethod
    def reset(cls) -> None:
        """Reset the global source cache (for tests and application shutdown)."""
        with cls._global_lock:
            if cls._global_instance is not None:
                cls._global_instance.clear()

    def clear(self) -> None:
        """Clear entries in this cache instance."""
        with self._lock:
            self._entries.clear()

    def read_profile(self, path: Optional[str] = None) -> tuple[dict[str, Any], str]:
        """Read and parse profile.yaml with mtime/size/digest validation."""
        resolved_path = os.path.abspath(path or get_settings().resolve_data_path("profile.yaml"))
        with self._lock:
            cached = self._check_cache_locked(resolved_path)
            if cached and cached.parsed_yaml is not None:
                return copy.deepcopy(cached.parsed_yaml), cached.raw_text

        try:
            stat = os.stat(resolved_path)
        except OSError:
            with self._lock:
                self._entries.pop(resolved_path, None)
            raise

        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        with self._lock:
            # Check if previous cached entry had identical content digest (e.g. metadata-only touch)
            prev_entry = self._entries.get(resolved_path)
            if (
                prev_entry is not None
                and prev_entry.digest == digest
                and prev_entry.parsed_yaml is not None
            ):
                new_entry = _CachedEntry(
                    mtime_ns=stat.st_mtime_ns,
                    size=stat.st_size,
                    digest=digest,
                    raw_text=raw_text,
                    parsed_yaml=prev_entry.parsed_yaml,
                )
                self._store_entry_locked(resolved_path, new_entry)
                return copy.deepcopy(prev_entry.parsed_yaml), raw_text

        # Digest differed or no previous entry; parse YAML
        parsed_data = yaml.safe_load(raw_text) or {}
        if not isinstance(parsed_data, dict):
            parsed_data = {}

        entry = _CachedEntry(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            digest=digest,
            raw_text=raw_text,
            parsed_yaml=parsed_data,
        )

        with self._lock:
            self._store_entry_locked(resolved_path, entry)
            return copy.deepcopy(parsed_data), raw_text

    def read_resume_markdown(self, path: Optional[str] = None) -> str:
        """Read resume.md with mtime/size/digest validation."""
        resolved_path = os.path.abspath(path or get_settings().resolve_data_path("resume.md"))
        with self._lock:
            cached = self._check_cache_locked(resolved_path)
            if cached:
                return cached.raw_text

        try:
            stat = os.stat(resolved_path)
        except OSError:
            with self._lock:
                self._entries.pop(resolved_path, None)
            raise

        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        with self._lock:
            prev_entry = self._entries.get(resolved_path)
            if prev_entry is not None and prev_entry.digest == digest:
                new_entry = _CachedEntry(
                    mtime_ns=stat.st_mtime_ns,
                    size=stat.st_size,
                    digest=digest,
                    raw_text=prev_entry.raw_text,
                    parsed_yaml=None,
                )
                self._store_entry_locked(resolved_path, new_entry)
                return prev_entry.raw_text

        entry = _CachedEntry(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            digest=digest,
            raw_text=raw_text,
            parsed_yaml=None,
        )

        with self._lock:
            self._store_entry_locked(resolved_path, entry)
            return raw_text

    def get_profile(self, path: Optional[str] = None) -> tuple[dict[str, Any], str]:
        return self.read_profile(path)

    def get_resume_markdown(self, path: Optional[str] = None) -> str:
        return self.read_resume_markdown(path)

    def _check_cache_locked(self, resolved_path: str) -> Optional[_CachedEntry]:
        """Check if cached entry for resolved_path matches stat fast-path."""
        if resolved_path not in self._entries:
            return None

        try:
            stat = os.stat(resolved_path)
        except OSError:
            self._entries.pop(resolved_path, None)
            return None

        cached = self._entries[resolved_path]
        if cached.mtime_ns == stat.st_mtime_ns and cached.size == stat.st_size:
            self._entries.move_to_end(resolved_path)
            return cached

        return None

    def _store_entry_locked(self, resolved_path: str, entry: _CachedEntry) -> None:
        """Store an entry ensuring the cache is bounded."""
        self._entries[resolved_path] = entry
        self._entries.move_to_end(resolved_path)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


def get_cached_profile(path: Optional[str] = None) -> tuple[dict[str, Any], str]:
    """Public helper to get cached profile dictionary and YAML string."""
    return SourceCache._get_global().get_profile(path)


def get_cached_resume_markdown(path: Optional[str] = None) -> str:
    """Public helper to get cached resume markdown text."""
    return SourceCache._get_global().get_resume_markdown(path)


def reset_source_cache() -> None:
    """Public helper to reset source cache."""
    SourceCache.reset()
