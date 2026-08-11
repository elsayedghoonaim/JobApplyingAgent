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

## Requirements

- Python 3.11+
- MongoDB on `mongodb://localhost:27017`
- Microsoft Edge
- Google API key with access to `gemma-4-31b-it`
- Telegram bot and chat ID

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
playwright install chromium
Copy-Item .env.example .env
```

Fill every value in `.env`. The only LLM variables are:

```env
GOOGLE_API_KEY=...
JOBAPPLY_LLM_MODEL=gemma-4-31b-it
JOBAPPLY_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

`JOBAPPLY_LLM_MODEL` is validated and cannot select another model.

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
question (and available options, when applicable). Send your answer normally as
the next Telegram message; no nonce or special format is required. Gemma
extracts the exact numeric value or option and fills the form. Questions are
asked one at a time as LinkedIn reveals each application page.

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

Run isolated unit/component tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\component_check.py --telegram
```

Run the Gemma rate-limit probe:

```powershell
.\.venv\Scripts\python.exe scripts\stress_test.py --phase A
```

Run one bounded dry-run job:

```powershell
.\.venv\Scripts\python.exe -m jobapply.main --dry-run --max-jobs 1 --max-applications 1
```

Only use `--no-dry-run` after reviewing a successful dry run.

## Resume an interrupted run

```powershell
jobapply --run-id <run-id>
```

Pending checkpoints resume without reinitializing the graph. A completed run
returns its saved result instead of starting again.

## Project layout

- `src/jobapply/graph.py`: workflow and routing
- `src/jobapply/nodes/`: search, qualification, generation, approval, execution, notification
- `src/jobapply/utils/llm.py`: Gemma-only REST client
- `src/jobapply/utils/telegram.py`: checked Telegram messaging
- `src/jobapply/utils/dedup.py`: shared MongoDB persistence
- `scripts/component_check.py`: bounded live component checks
- `scripts/gemma_rate_limit_test.py`: rate-limit diagnostics
- `tests/`: isolated workflow tests

Runtime logs, generated letters, edited resumes, and screenshots are written
under `outputs/<run-id>/` and are excluded from source control.
