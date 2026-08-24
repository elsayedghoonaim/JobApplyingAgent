"""Offline deterministic tests for structured, redacted, bounded logging."""

import io
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Mapping as AbcMapping
from pathlib import Path

import pytest

import jobapply
from jobapply.utils.monitoring import RedactingFormatter, setup_logging, shutdown_logging
from jobapply.utils.observability import (
    LOGGER_NAME,
    MAX_CONSOLE_MESSAGE_LENGTH,
    MAX_DETAIL_DEPTH,
    MAX_DETAIL_ITEMS,
    MAX_DETAIL_KEYS,
    MAX_DETAIL_STRING_LENGTH,
    MAX_MESSAGE_LENGTH,
    MAX_TOTAL_DETAIL_ENTRIES,
    StructuredJsonlFormatter,
    bound_text,
    fingerprint,
    log_event,
    sanitize_log_value,
)
from jobapply.utils.redaction import is_sensitive_key

REQUIRED_RECORD_FIELDS = (
    "timestamp",
    "level",
    "event",
    "message",
    "run_id",
    "job_id",
    "attempt_id",
    "node",
    "stage",
    "outcome",
    "status",
    "details",
    "exception",
)


@pytest.fixture()
def jsonl_log_path(tmp_path):
    """Configure the workflow logger against a temp dir and yield its JSONL file."""
    shutdown_logging(LOGGER_NAME)
    setup_logging("obs-run", log_dir=str(tmp_path / "outputs"))
    log_dir = tmp_path / "outputs" / "obs-run"
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) == 1
    yield log_files[0]
    shutdown_logging(LOGGER_NAME)


def _read_records(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_jsonl_file_records_are_valid_with_required_fields(jsonl_log_path):
    log_event(
        "info",
        "test.event",
        "Structured event emitted",
        run_id="run-abc",
        job_id="job-42",
        attempt_id="attempt-7",
        stage="search_node",
        status="qualified",
        details={"page": 2, "count": 5},
    )
    log_event("debug", "test.minimal", "Minimal event")

    records = _read_records(jsonl_log_path)
    assert len(records) == 2
    for record in records:
        for field in REQUIRED_RECORD_FIELDS:
            assert field in record

    first = records[0]
    assert first["level"] == "INFO"
    assert first["event"] == "test.event"
    assert first["message"] == "Structured event emitted"
    assert first["run_id"] == "run-abc"
    assert first["job_id"] == "job-42"
    assert first["attempt_id"] == "attempt-7"
    assert first["node"] == "search_node"
    assert first["stage"] == "search_node"
    assert first["status"] == "qualified"
    assert first["details"] == {"page": 2, "count": 5}
    assert first["exception"] is None
    assert first["timestamp"].endswith("Z")

    second = records[1]
    assert second["level"] == "DEBUG"
    assert second["run_id"] is None


def test_nested_secrets_and_hostile_exception_text_are_redacted_in_logs(
    jsonl_log_path,
):
    token = "123456789:AAFakeSyntheticTelegramTokenXYZ123456"
    google_key = "AIzaSyFakeKeyForTestingPurposes1234567890"
    log_event(
        "warning",
        "test.hostile",
        f"Leaked bot token {token} and key {google_key} in message",
        details={
            "api_key": google_key,
            "nested": {"password": "super-secret-password", "note": "safe text"},
            "authorization": {"scheme": "bearer"},
            "url": f"https://api.telegram.org/bot{token}/sendMessage",
        },
        exc=RuntimeError(f"upstream failure at https://api.telegram.org/bot{token}/getUpdates"),
    )

    raw = jsonl_log_path.read_text(encoding="utf-8")
    assert token not in raw
    assert "AIzaSy" not in raw
    assert "super-secret-password" not in raw
    assert "[REDACTED]" in raw

    (record,) = [r for r in _read_records(jsonl_log_path) if r["event"] == "test.hostile"]
    assert record["exception"]["type"] == "RuntimeError"
    assert "bot[REDACTED]" in record["exception"]["message"]
    assert record["details"]["api_key"] == "[REDACTED]"
    assert record["details"]["nested"]["password"] == "[REDACTED]"
    assert record["details"]["nested"]["note"] == "safe text"
    assert record["details"]["authorization"] == "[REDACTED]"


def test_logging_bounds_depth_list_size_key_count_and_string_size():
    hostile = {
        "long_string": "x" * 5000,
        "long_key_" + "k" * 300: "value",
        "giant_list": [{"index": i} for i in range(500)],
        "deep": {"l1": {"l2": {"l3": {"l4": {"l5": {"l6": "too deep"}}}}}},
        **{f"key_{i}": i for i in range(200)},
    }
    bounded = sanitize_log_value(hostile)

    assert len(bounded["long_string"]) <= MAX_DETAIL_STRING_LENGTH
    assert bounded["long_string"].endswith("...")
    assert len(bounded["giant_list"]) <= MAX_DETAIL_ITEMS + 1
    assert any(str(item).startswith("[TRUNCATED_ITEMS") for item in bounded["giant_list"])
    assert len([key for key in bounded if key.startswith("key_")]) <= MAX_DETAIL_KEYS
    assert "[truncated_keys]" in bounded
    assert bounded["deep"]["l1"]["l2"]["l3"] == "[TRUNCATED_DEPTH]"
    assert MAX_DETAIL_DEPTH >= 3


def test_message_length_is_bounded_in_file_records(jsonl_log_path):
    log_event("info", "test.huge", "y" * 10000)
    (record,) = _read_records(jsonl_log_path)
    assert len(record["message"]) <= MAX_MESSAGE_LENGTH


def test_setup_logging_never_duplicates_handlers_and_disables_propagation(tmp_path):
    shutdown_logging(LOGGER_NAME)
    try:
        first = setup_logging("obs-dup", log_dir=str(tmp_path / "outputs"))
        assert len(first.handlers) == 2
        assert first.propagate is False

        second = setup_logging("obs-dup", log_dir=str(tmp_path / "outputs"))
        assert second is first
        assert len(second.handlers) == 2
        assert isinstance(second.handlers[0], logging.FileHandler)
        assert isinstance(second.handlers[1], logging.StreamHandler)
    finally:
        shutdown_logging(LOGGER_NAME)


def test_console_output_stays_human_readable_while_file_is_jsonl(tmp_path):
    shutdown_logging(LOGGER_NAME)
    logger = setup_logging("obs-console", log_dir=str(tmp_path / "outputs"))

    stream = io.StringIO()
    console_handler = next(
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    console_handler.setStream(stream)

    log_event(
        "info",
        "test.console",
        "Operator friendly message with token 123456789:AAFakeSyntheticTelegramTokenXYZ123456",
        run_id="obs-console-run",
    )
    console_text = stream.getvalue()
    assert "Operator friendly message" in console_text
    assert '"event"' not in console_text
    assert "AAFakeSyntheticTelegramToken" not in console_text

    shutdown_logging(LOGGER_NAME)


def test_bound_text_redacts_and_bounds():
    text = bound_text(
        "secret AIzaSyFakeKeyForTestingPurposes1234567890 " + "z" * 400, max_length=80
    )
    assert "AIzaSy" not in text
    assert len(text) <= 80
    assert bound_text("", 10) == ""


def test_fingerprint_is_stable_and_short():
    assert fingerprint("How many years of experience?*") == fingerprint(
        " How Many Years of Experience?* "
    )
    assert len(fingerprint("anything")) == 16
    assert fingerprint("a") != fingerprint("b")


@pytest.mark.parametrize("level_name", ["debug", "info", "warning", "error"])
def test_log_event_accepts_standard_level_names(level_name, tmp_path):
    shutdown_logging(LOGGER_NAME)
    try:
        setup_logging("obs-levels", log_dir=str(tmp_path / "outputs"))
        log_event(level_name, "test.levels", "Level check")
        log_file = next((tmp_path / "outputs" / "obs-levels").glob("*.log"))
        (record,) = _read_records(log_file)
        assert record["level"] == level_name.upper()
    finally:
        shutdown_logging(LOGGER_NAME)


def test_console_formatter_remains_redacting():
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name=LOGGER_NAME,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="connect %s",
        args=("https://api.telegram.org/bot123456789:AAFakeSyntheticTelegramTokenXYZ/sendMessage",),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "AAFakeSyntheticTelegramToken" not in rendered
    assert "bot[REDACTED]" in rendered


def test_console_output_is_bounded_for_very_large_messages(tmp_path):
    shutdown_logging(LOGGER_NAME)
    logger = setup_logging("obs-console-bound", log_dir=str(tmp_path / "outputs"))
    try:
        stream = io.StringIO()
        console_handler = next(
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        )
        console_handler.setStream(stream)

        huge = (
            "Operator message with token 123456789:AAFakeSyntheticTelegramTokenXYZ123456 inside "
            + "y" * 20000
        )
        log_event("info", "test.console.bound", huge, run_id="obs-bound")

        console_text = stream.getvalue().rstrip("\n")
        assert len(console_text) <= MAX_CONSOLE_MESSAGE_LENGTH
        assert console_text.endswith("...")
        assert "AAFakeSyntheticTelegramToken" not in console_text
    finally:
        shutdown_logging(LOGGER_NAME)


class _LazyCountingMapping(AbcMapping):
    """Hostile mapping that yields 10k pairs lazily and counts consumption."""

    def __init__(self):
        self.consumed = 0
        self.total = 10000

    def __getitem__(self, key):
        return 0

    def __len__(self):
        return self.total

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        return (f"k{i}" for i in range(self.total))

    def items(self):
        def generator():
            for i in range(self.total):
                self.consumed += 1
                yield f"k{i}", {"index": i}

        return generator()


class _ExplodingMapping(AbcMapping):
    def __getitem__(self, key):
        raise RuntimeError("getitem exploded")

    def __len__(self):
        raise RuntimeError("len exploded")

    def __iter__(self):
        raise RuntimeError("iteration exploded")

    def items(self):
        raise RuntimeError("items exploded with password=super-secret-value")


class _LazyCountingSequence:
    """Hostile sequence that yields 10k items lazily and counts consumption."""

    def __init__(self):
        self.consumed = 0

    def __iter__(self):
        def generator():
            for i in range(10000):
                self.consumed += 1
                yield {"index": i}

        return generator()

    def __len__(self):
        return 10000


class _ExplodingIterable:
    def __iter__(self):
        raise RuntimeError("iteration exploded")

    def __repr__(self):
        raise RuntimeError("repr exploded")


class _ExplodingReprObject:
    def __repr__(self):
        raise RuntimeError("repr exploded")


def test_sanitizer_consumes_only_bounded_entries_from_hostile_containers():
    lazy_mapping = _LazyCountingMapping()
    bounded_mapping = sanitize_log_value(lazy_mapping)
    assert lazy_mapping.consumed <= MAX_DETAIL_KEYS + 1
    assert len([k for k in bounded_mapping if k.startswith("k")]) == MAX_DETAIL_KEYS
    assert bounded_mapping["[truncated_keys]"] == str(10000 - MAX_DETAIL_KEYS)

    lazy_sequence = _LazyCountingSequence()
    bounded_sequence = sanitize_log_value(lazy_sequence)
    assert lazy_sequence.consumed <= MAX_DETAIL_ITEMS + 1
    assert len(bounded_sequence) == MAX_DETAIL_ITEMS + 1
    assert bounded_sequence[-1] == f"[TRUNCATED_ITEMS {10000 - MAX_DETAIL_ITEMS}]"

    unsized_generator = ({"i": i} for i in range(10000))
    bounded_generator = sanitize_log_value(unsized_generator)
    assert len(bounded_generator) == MAX_DETAIL_ITEMS + 1
    assert bounded_generator[-1] == "[TRUNCATED_ITEMS unknown]"


def test_sanitizer_survives_hostile_iteration_and_repr_failures():
    assert sanitize_log_value(_ExplodingMapping()) == "[UNSERIALIZABLE_MAPPING]"
    assert sanitize_log_value(_ExplodingIterable()) == "[UNSERIALIZABLE_SEQUENCE]"
    assert sanitize_log_value(_ExplodingReprObject()) == "[UNSERIALIZABLE_VALUE]"


def test_sets_serialize_deterministically():
    first = sanitize_log_value({"delta", "alpha", "charlie", "bravo"})
    second = sanitize_log_value({"charlie", "alpha", "bravo", "delta"})
    assert first == sorted(first)
    assert first == second


def test_hostile_container_log_record_stays_valid_and_bounded(jsonl_log_path):
    log_event(
        "warning",
        "test.hostile_containers",
        "Hostile containers emitted",
        details={
            "lazy": _LazyCountingMapping(),
            "exploding": _ExplodingIterable(),
            "secret_key": "ignored-because-sensitive",
            "password": "super-secret-password",
        },
        exc=RuntimeError("boom password=super-secret-db-pass"),
    )

    raw = jsonl_log_path.read_text(encoding="utf-8")
    assert "super-secret-password" not in raw
    assert "super-secret-db-pass" not in raw
    lines = raw.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "test.hostile_containers"
    details = record["details"]
    assert len([k for k in details["lazy"] if k.startswith("k")]) <= MAX_DETAIL_KEYS
    assert details["exploding"] == "[UNSERIALIZABLE_SEQUENCE]"
    assert record["exception"]["type"] == "RuntimeError"
    assert len(json.dumps(record)) < 20000


PRODUCTION_PRINT_FREE_TARGETS = ("nodes", "execution", "graph.py")


def test_production_workflow_modules_contain_no_direct_print_calls():
    package_dir = Path(jobapply.__file__).parent
    targets = sorted((package_dir / "nodes").glob("*.py"))
    targets += sorted((package_dir / "execution").glob("*.py"))
    targets.append(package_dir / "graph.py")

    offenders = []
    for path in targets:
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(?<![\w.])print\s*\(", source):
            line_no = source.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(package_dir)}:{line_no}")

    assert offenders == []


_SUBPROCESS_SET_SCRIPT = (
    "import json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from jobapply.utils.observability import sanitize_log_value\n"
    "size = int(sys.argv[2])\n"
    "payload = sanitize_log_value({f'elem_{i:04d}' for i in range(size)})\n"
    "print(json.dumps(payload, sort_keys=True))\n"
)


def _run_set_snippet(src_path: str, size: int, hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SET_SCRIPT, src_path, str(size)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout.strip()


def test_prefixed_credential_keys_are_redacted_in_logs(jsonl_log_path):
    log_event(
        "info",
        "test.prefixed_credentials",
        "Credential material handled",
        details={
            "database_password": "plain-secret-value",
            "service_access_token": "opaque-value-123",
            "db_passwd": "another-secret-value",
            "user_pwd": "pwd-secret-value",
            "aws_secret_key": "aws-secret-value",
            "nested": {"client_auth_token": "nested-token-secret"},
            "token_count": 42,
            "password_policy": "strict",
        },
    )

    raw = jsonl_log_path.read_text(encoding="utf-8")
    assert "plain-secret-value" not in raw
    assert "opaque-value-123" not in raw
    assert "another-secret-value" not in raw
    assert "pwd-secret-value" not in raw
    assert "aws-secret-value" not in raw
    assert "nested-token-secret" not in raw

    (record,) = [
        r for r in _read_records(jsonl_log_path) if r["event"] == "test.prefixed_credentials"
    ]
    details = record["details"]
    assert details["database_password"] == "[REDACTED]"
    assert details["service_access_token"] == "[REDACTED]"
    assert details["db_passwd"] == "[REDACTED]"
    assert details["user_pwd"] == "[REDACTED]"
    assert details["aws_secret_key"] == "[REDACTED]"
    assert details["nested"]["client_auth_token"] == "[REDACTED]"
    # Harmless near-miss fields stay untouched.
    assert details["token_count"] == 42
    assert details["password_policy"] == "strict"


def test_sensitive_key_matching_is_precise():
    for key in (
        "database_password",
        "service_access_token",
        "db_passwd",
        "user_pwd",
        "app_secret",
        "openai_api_key",
        "session_auth_token",
        "telegram_bot_token",
        "google_client_secret",
        "aws_private_key",
        "vault_secret_key",
        "Service-Access-Token",
    ):
        assert is_sensitive_key(key), key

    for harmless in ("token_count", "password_policy", "tokenizer", "secret_sauce_recipe"):
        assert not is_sensitive_key(harmless), harmless


@pytest.mark.parametrize("hash_seed", ["1", "2"])
def test_set_serialization_deterministic_across_hash_seeds(tmp_path, hash_seed):
    src_dir = str(Path(jobapply.__file__).parent.parent)

    oversized_first = _run_set_snippet(src_dir, 300, hash_seed)
    oversized_second = _run_set_snippet(src_dir, 300, hash_seed)
    # Oversized sets never depend on iteration order: cardinality marker only.
    assert json.loads(oversized_first) == f"[TRUNCATED_SET {MAX_DETAIL_ITEMS + 250}]"
    assert oversized_first == oversized_second

    small = json.loads(_run_set_snippet(src_dir, 10, hash_seed))
    assert small == sorted(small)
    assert len(small) == 10


def test_global_budget_caps_exponentially_branching_structures():
    def branch(level: int):
        if level == 0:
            return {f"leaf{i}": f"leaf-value-{i}" for i in range(8)}
        return {f"node{i}": branch(level - 1) for i in range(8)}

    tree = {f"root{i}": branch(5) for i in range(8)}

    bounded = sanitize_log_value(tree)
    serialized = json.dumps(bounded)

    leaf_hits = serialized.count("leaf-value-")
    assert leaf_hits <= MAX_TOTAL_DETAIL_ENTRIES
    assert len(serialized) < 40_000
    assert "[TRUNCATED_BUDGET]" in serialized

    # Fresh budget per call: repeated sanitization is identical, never degraded.
    assert json.dumps(sanitize_log_value(tree)) == serialized


def test_formatter_sanitizes_externally_supplied_exception_summary(tmp_path):
    shutdown_logging(LOGGER_NAME)
    setup_logging("obs-exception-field", log_dir=str(tmp_path / "outputs"))
    try:
        log_event(
            "warning",
            "test.external_exception",
            "External exception summary path",
            exc=RuntimeError(
                "leak https://api.telegram.org/bot123456789:AAFakeSyntheticTelegramTokenXYZ123/getUpdates"
            ),
        )
        log_file = next((tmp_path / "outputs" / "obs-exception-field").glob("*.log"))
        raw = log_file.read_text(encoding="utf-8")
        assert "AAFakeSyntheticTelegramToken" not in raw
        parsed = json.loads(raw.splitlines()[0])
        assert parsed["exception"]["type"] == "RuntimeError"
        assert "bot[REDACTED]" in parsed["exception"]["message"]

        # Direct formatter probe with a hostile externally supplied field.
        formatter = StructuredJsonlFormatter()
        record = logging.LogRecord(
            name=LOGGER_NAME,
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="probe",
            args=None,
            exc_info=None,
        )
        record.event = "test.hostile_exception_field"
        record.exception_summary = {
            "type": "RuntimeError",
            "message": "leak https://api.telegram.org/bot123456789:AAFakeSyntheticTelegramTokenXYZ123/getUpdates",
            "password": "super-secret-password",
        }
        line = formatter.format(record)
        assert "AAFakeSyntheticTelegramToken" not in line
        assert "super-secret-password" not in line
        hostile_parsed = json.loads(line)
        assert hostile_parsed["exception"]["message"].endswith("bot[REDACTED]/getUpdates")
        assert hostile_parsed["exception"]["password"] == "[REDACTED]"
    finally:
        shutdown_logging(LOGGER_NAME)


class _CountingMapping(AbcMapping):
    """Hostile mapping that counts every yielded pair in a shared counter."""

    def __init__(self, entries, counter):
        self._entries = list(entries)
        self.counter = counter

    def __getitem__(self, key):
        raise KeyError(key)

    def __len__(self):
        return 4096

    def __iter__(self):
        return iter(key for key, _ in self._entries)

    def items(self):
        def generator():
            for pair in self._entries:
                self.counter["yields"] += 1
                yield pair

        return generator()


class _CountingSequence:
    """Hostile sequence that counts every yielded element in a shared counter."""

    def __init__(self, items, counter):
        self._items = list(items)
        self.counter = counter

    def __len__(self):
        return 4096

    def __iter__(self):
        def generator():
            for item in self._items:
                self.counter["yields"] += 1
                yield item

        return generator()


def _build_counting_tree(counter, depth):
    if depth == 0:
        return [f"leaf-{i}" for i in range(8)]
    if depth % 2 == 0:
        return _CountingMapping(
            [(f"k{i}", _build_counting_tree(counter, depth - 1)) for i in range(8)],
            counter,
        )
    return _CountingSequence(
        [_build_counting_tree(counter, depth - 1) for _ in range(8)],
        counter,
    )


def test_global_budget_limits_actual_iterator_yields_across_nesting():
    counter = {"yields": 0}
    tree = _build_counting_tree(counter, 5)

    bounded = sanitize_log_value(tree)
    serialized = json.dumps(bounded)

    # Actual yields across the whole structure respect the shared budget.
    assert counter["yields"] <= MAX_TOTAL_DETAIL_ENTRIES
    assert "[TRUNCATED_BUDGET]" in serialized
    assert len(serialized) < 40_000

    # Fresh budget per call: a second sanitization performs the same bounded work.
    second_counter = {"yields": 0}
    second_tree = _build_counting_tree(second_counter, 5)
    assert json.dumps(sanitize_log_value(second_tree)) == serialized
    assert second_counter["yields"] <= MAX_TOTAL_DETAIL_ENTRIES
