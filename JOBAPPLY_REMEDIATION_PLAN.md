# JobApply Reliability, Safety, and Performance Plan

## Summary

The review found a syntactically valid project with 61 tests, but tests cannot currently collect because required dependencies are missing. Highest-risk findings are personal documents not being ignored, possible attachment to the normal Edge profile, secret-bearing URLs reaching errors, unsafe artifact paths, incomplete job-page scanning, and non-idempotent submission recovery.

Gemini 3.6 Flash High critiqued the draft through Antigravity using only redacted findings and no repository access. Its useful corrections are incorporated below.

## Phase 0 — Privacy and Green Baseline

- Move real application documents into ignored `user-data/`, commit sanitized examples, and add `JOBAPPLY_DATA_DIR=user-data`.
- Verify secrets, outputs, caches, browser profiles, and personal documents are ignored before recording the baseline.
- Adopt `uv`, add `uv.lock`, explicit build/package configuration, Ruff, Pyright, pytest-cov, and build validation.
- Restore the environment and make existing tests pass before refactoring; add offline-only Windows CI for Python 3.11/3.12.

## Phase 1 — Security and Account Safety

- Centralize safe artifact paths and validate run/job identifiers.
- Harden Playwright PDF rendering with sanitization, CSP, disabled JavaScript, and blocked external/file requests.
- Redact Telegram/Google credentials before errors reach logs, state, MongoDB, or traces; use Google header authentication where supported.
- Discover CDP only from the dedicated profile, validate ownership, and never automate or close an unrelated Edge process.
- Detect CAPTCHA, challenge, checkpoint, 2FA, restriction, and rate-limit pages; pause safely and never bypass them.

## Phase 2 — Search and Idempotent Submission

- Inspect all cards within page/max-job bounds rather than limiting extraction from the remaining application cap.
- Batch-query seen IDs, persist deterministic exclusions, and return immutable state updates.
- Add indexed attempts/quotas with typed states including `submission_unknown`.
- Atomically reserve quotas before Submit, persist unknown status before clicking, and mark submitted only after explicit confirmation.
- Never auto-retry unknown attempts; persist Telegram correlation and queue failed notifications safely.

## Phase 3 — Refactor, Performance, and Observability

- Split execution into a DOM adapter, answer planner, Telegram service, form navigator, repositories, and outcome formatting.
- Remove duplicate field processing, validate required fields, and use typed bounded failures.
- Combine description parsing and qualification; keep document-generation calls separate.
- Bound/cache LLM inputs, reuse owned resources, batch database work, and avoid loading the full seen collection.
- Replace node prints with structured redacted logging and write `outputs/<run-id>/summary.json`.

## Phase 4 — Operator Enhancements

- Add `jobapply doctor [--live-checks]` while preserving existing CLI behavior.
- Add Telegram buttons, a manual-review queue, configurable location/recency, dry-run metrics, and non-destructive repost flags.

## Test and Acceptance

- Gates: `uv sync --extra dev`, Ruff, Pyright, pytest with branch coverage, `uv build`, wheel installation, and CLI smoke tests.
- Test traversal, redaction, Edge isolation, checkpoint pausing, full scanning, quota concurrency, Submit crash recovery, Telegram recovery, and synthetic Playwright forms.
- Automated tests block external network access. Real LinkedIn submission remains manual and explicitly authorized.

## Defaults

- Use `uv`, ignored repository-local `user-data/`, hardened Playwright PDF rendering, and Gemma as the only production LLM.
- Use atomic single-document quota updates without requiring MongoDB transactions.
- Never attempt CAPTCHA or anti-bot evasion.
