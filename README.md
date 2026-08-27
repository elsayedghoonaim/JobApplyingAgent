# JobApply

Autonomous, checkpointed LinkedIn Easy Apply workflow powered exclusively by
`gemma-4-31b-it`. The agent searches recent Easy Apply jobs, evaluates fit,
generates application documents, requests Telegram answers when needed, and
uses an existing signed-in Microsoft Edge session.

## Safety defaults

- Dry-run mode is enabled by default and never clicks Submit.
- Session and daily caps are checked before browser work and again before live submission.
- A live submission counts only after LinkedIn displays an explicit success message.
- Resume edits require Telegram approval.
- Application agreements require confirmation unless explicitly enabled in `.env`.
- Job descriptions are treated as untrusted LLM input.
- Personal application data is stored outside package code in an ignored `user-data/` directory.

## Requirements

- Python 3.11+ with `uv`
- MongoDB on `mongodb://localhost:27017`
- Microsoft Edge
- Google API key with access to `gemma-4-31b-it`
- Telegram bot and chat ID

## Installation

For a first-time personal setup, follow [CUSTOMIZE.md](CUSTOMIZE.md). It explains
the candidate files, target titles, location, language preferences, safety
limits, dry-run verification, and the first live run.

```powershell
# Install dependencies with uv
uv sync --extra dev

# Install Playwright browser binaries
uv run playwright install chromium

# Copy environment template
Copy-Item .env.example .env

# Initialize ignored user-data directory from tracked templates
New-Item -ItemType Directory -Force user-data
Copy-Item templates\* user-data\
```

Fill every required value in `.env`. The LLM settings are:

```env
GOOGLE_API_KEY=...
JOBAPPLY_LLM_MODEL=gemma-4-31b-it
JOBAPPLY_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
JOBAPPLY_DATA_DIR=user-data
```

`JOBAPPLY_LLM_MODEL` is validated and cannot select another model.
`JOBAPPLY_DATA_DIR` defaults to `user-data` and points to the repository-local or absolute directory containing `profile.yaml`, `resume.md`, and `resume.pdf`.

## User Data & Templates

- Tracked template files live in `templates/` (`templates/profile.yaml`, `templates/resume.md`, `templates/resume.pdf`).
- Real candidate documents live in `user-data/` (or the location configured by `JOBAPPLY_DATA_DIR`).
- `user-data/` and legacy paths under `src/jobapply/data/` are ignored by Git and excluded from package wheels.

## Start the signed-in Edge session

Current Edge versions protect the normal Default profile from command-line
automation. JobApply therefore auto-launches Microsoft Edge with a separate,
persistent profile under `%LOCALAPPDATA%\JobApply\EdgeProfile`.

Run the bounded dry-run command once. A visible Edge window opens on LinkedIn.
Sign in manually in that window; the dedicated profile preserves the session
for future runs. JobApply never copies or inspects the normal Edge profile's
cookies or passwords.

## Telegram form questions

When Easy Apply exposes an unanswered required field, the bot sends only the
question (and available options, when applicable). Small bounded option lists
also arrive as tappable inline buttons (plus a persistent **Skip Job** button);
ordinary text replies remain fully supported at any time — no nonce or special
format is required. Gemma extracts the exact numeric value or option and fills
the form. Questions are asked one at a time as LinkedIn reveals each
application page. Button presses are validated against the exact durable
prompt/nonce, persisted before acknowledgment, and duplicate presses are
idempotent.

## Search location and recency

Configure the search scope in `.env` or per run on the CLI:

```env
JOBAPPLY_SEARCH_LOCATION=Worldwide
JOBAPPLY_SEARCH_RECENCY_DAYS=
```

- `JOBAPPLY_SEARCH_LOCATION` is trimmed, bounded, and safely URL-encoded; the default is `Worldwide`.
- `JOBAPPLY_SEARCH_RECENCY_DAYS` accepts a bounded positive number of days (1–30). Unset, empty, or `0` disables the recency filter entirely.
- CLI overrides win over `.env`: `uv run jobapply --location "Berlin, Germany" --recency-days 7`.
- Use `--recency-days 0` to explicitly disable filtering for one run.
- Effective values are stored in run/checkpoint state so resumed runs keep identical search behavior.
- Reposted listings are flagged non-destructively (`is_repost`, bounded redacted evidence) and never filtered, deprioritized, or counted against quotas.

Search queries discover listings, while `JOBAPPLY_TARGET_TITLE_KEYWORDS` is the
strict title allowlist. For example, a broad `AI Engineer` search can still be
used while only titles containing `Machine Learning`, `ML`, `MLOps`, or
`Deep Learning` are processed. See [CUSTOMIZE.md](CUSTOMIZE.md) for examples.

## Doctor diagnostics

```powershell
uv run jobapply doctor                # offline, non-mutating; zero network calls
uv run jobapply doctor --live-checks  # opt-in bounded read-only probes
```

Offline checks cover Python version, settings validation, the Gemma-only model
constraint, user-data files and safe resolved paths, output directory access,
Edge executable configuration and dedicated-profile containment, credential
presence (reported only as configured/missing), MongoDB URL syntax (no
connection), and search configuration. Results print as PASS/WARN/FAIL with a
final count; the exit code is zero when all required checks pass.

`--live-checks` additionally performs short-timeout, read-only probes of the
configured MongoDB deployment (admin ping), Telegram (`getMe` only — no
messages are sent), the Gemma endpoint availability (header authentication),
and the managed Edge CDP endpoint. Live checks never navigate to LinkedIn,
never invoke the LLM, never write data, and always close their resources.

## Manual review queue

Every `needs_manual_review` outcome is durably queued (idempotent across
resume/replay) in the configured collection
(`JOBAPPLY_MANUAL_REVIEW_COLLECTION=manual_review_queue`). Ambiguous
`submission_unknown` items stay visibly distinct, are never auto-retried or
auto-resolved, and require an explicit safe resolution label. Operator
commands:

```powershell
uv run jobapply review list [--limit N]
uv run jobapply review acknowledge <item-id> [--note "..."]
uv run jobapply review resolve <item-id> [--note "..."] --label confirmed_not_submitted|confirmed_submitted|requires_followup
```

Acknowledging or resolving never increments application counts or alters
submission truth, and no browser retry is ever triggered from the queue.

## One-command live run

This command performs preflight checks, starts the dedicated Edge profile,
waits for LinkedIn sign-in when necessary, and then submits real applications:

```powershell
.\run-live.cmd
```

It continues through every job in the configured search queries and pages until
the search scope is exhausted or an application cap is reached. Telegram is
used for unanswered fields, consent, per-application receipts, and the final
structured session summary. This is live mode: Submit may be clicked.

To verify Edge, LinkedIn, MongoDB, and configuration without processing jobs:

```powershell
.\run-live.cmd --preflight-only
```

## Verification

Run all repository quality gates offline:

```powershell
uv lock --check
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q
uv build
```

Run bounded dry-run checks:

```powershell
uv run python scripts/component_check.py --telegram
uv run python -m jobapply.main --dry-run --max-jobs 1 --max-applications 1
```

Only use `--no-dry-run` after reviewing a successful dry run.

## Resume an interrupted run

```powershell
uv run jobapply --run-id <run-id>
```

Pending checkpoints resume without reinitializing the graph. A completed run
returns its saved result instead of starting again.

## Project layout

- `templates/`: tracked sanitized profile and resume templates
- `user-data/`: ignored directory for real profile.yaml, resume.md, and resume.pdf
- `src/jobapply/graph.py`: workflow and routing
- `src/jobapply/nodes/`: search, qualification, generation, approval, execution, notification
- `src/jobapply/utils/llm.py`: Gemma-only REST client
- `src/jobapply/utils/telegram.py`: checked Telegram messaging
- `src/jobapply/utils/dedup.py`: shared MongoDB persistence
- `scripts/component_check.py`: bounded live component checks
- `tests/`: isolated workflow tests

Runtime logs, generated letters, edited resumes, and screenshots are written
under `outputs/<run-id>/` and are excluded from source control.
