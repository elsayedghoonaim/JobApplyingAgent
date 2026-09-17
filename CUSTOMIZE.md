# Customize JobApply for a New Person

This setup keeps each person's real information in the ignored `user-data/`
directory and all preferences in the ignored `.env` file. Neither should be
committed to Git.

## 1. Install and create private files

From the project directory in PowerShell:

```powershell
uv sync --extra dev
uv run playwright install chromium
Copy-Item .env.example .env
New-Item -ItemType Directory -Force user-data
Copy-Item templates\profile.yaml user-data\profile.yaml
Copy-Item templates\resume.md user-data\resume.md
Copy-Item templates\resume.pdf user-data\resume.pdf
```

If `.env` or `user-data/` already contains real information, do not copy over
it. Back it up before changing users.

## 2. Add the person's resume

Replace `user-data/resume.pdf` with the exact PDF that should be submitted.
LinkedIn's already-selected saved resume is reused; this PDF is the fallback
when no saved resume is selected.

Replace `user-data/resume.md` with a faithful Markdown text version of that
same resume. The agent reads this text to judge fit and prepare application
documents. Do not add skills or experience that are absent from the real PDF.

## 3. Fill in the profile

Edit `user-data/profile.yaml`. At minimum, replace every example value:

```yaml
name: "Full Name"
email: "person@example.com"
phone: "+201234567890"
location: "Cairo, Egypt"
linkedin: "https://www.linkedin.com/in/profile"
github: "https://github.com/profile"
summary: "A factual two- or three-sentence professional summary."
skills:
  - Python
  - Machine Learning
form_defaults:
  years_of_experience: "3"
  highest_education: "Bachelor's Degree"
```

Add other truthful facts that help answer forms, such as work experience,
education, certifications, portfolio, notice period, or languages. Leave
unknown or sensitive answers out; the Telegram bot will ask when a form needs
them.

## 4. Set the target jobs

Open `.env` and edit these values. They are comma-separated lists:

```env
# Searches LinkedIn performs (broad discovery is useful)
JOBAPPLY_SEARCH_QUERIES=Machine Learning Engineer,AI Engineer,MLOps Engineer

# Strict title gate: at least one phrase must appear in the actual job title
JOBAPPLY_TARGET_TITLE_KEYWORDS=Machine Learning,ML,MLOps,Deep Learning

# true skips Senior, Lead, Staff, Principal, Manager, Director, and similar titles
JOBAPPLY_EXCLUDE_SENIOR_TITLES=true

# A job requiring a language outside this list is skipped
JOBAPPLY_ALLOWED_LANGUAGES=Arabic,English
```

The title phrases are literal and case-insensitive. `ML` matches a standalone
word, not unrelated words. Keep the list precise: adding a broad term such as
`Engineer` permits almost every engineering title.

Examples for other candidates:

```env
# Data analyst
JOBAPPLY_SEARCH_QUERIES=Data Analyst,Business Intelligence Analyst
JOBAPPLY_TARGET_TITLE_KEYWORDS=Data Analyst,BI Analyst,Business Intelligence Analyst

# Backend developer
JOBAPPLY_SEARCH_QUERIES=Python Backend Developer,Backend Engineer
JOBAPPLY_TARGET_TITLE_KEYWORDS=Backend Developer,Backend Engineer,Python Developer
```

## 5. Set location, freshness, and limits

Edit the same `.env` file:

```env
JOBAPPLY_SEARCH_LOCATION=Worldwide
JOBAPPLY_SEARCH_RECENCY_DAYS=7
JOBAPPLY_QUALIFICATION_THRESHOLD=0.6
JOBAPPLY_MAX_APPLICATIONS_PER_SESSION=5
JOBAPPLY_DAILY_APPLICATION_CAP=20
JOBAPPLY_DRY_RUN=true
```

Use `Worldwide`, a country, or a city supported by LinkedIn. Recency accepts
1-30 days; leave it blank or use `0` for all dates. Start with a small session
limit and keep dry-run enabled until the results look correct.

## 6. Add service credentials

Fill the required placeholders in `.env`:

```env
JOBAPPLY_LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
JOBAPPLY_TELEGRAM_BOT_TOKEN=...
JOBAPPLY_TELEGRAM_CHAT_ID=...
```

MongoDB must also be running at the configured URL. Keep `.env` private.

## 7. Validate before opening LinkedIn

```powershell
uv run jobapply doctor
```

Fix every `FAIL`. Review warnings. The output should show the configured target
title phrases, senior-title rule, languages, location, and recency.

Then start the managed Edge session and check sign-in without processing jobs:

```powershell
.\run-live.cmd --preflight-only
```

Sign in to LinkedIn in the Edge window if prompted. This dedicated browser
profile preserves the login for later runs.

## 8. Test one job safely

Keep `JOBAPPLY_DRY_RUN=true`, then run:

```powershell
uv run jobapply --dry-run --max-jobs 1 --max-applications 1
```

Confirm that the title is in scope, the candidate details are correct, and the
Telegram questions work. Dry-run never clicks Submit.

## 9. Enable live applications

Only after the dry run is correct, set:

```env
JOBAPPLY_DRY_RUN=false
```

Then run:

```powershell
.\run-live.cmd
```

This command can submit real applications. Monitor the first full run and use
small limits until the setup has been proven for that person's account.

## Switching to another person later

Stop the agent, back up the old `.env` and `user-data/`, replace all three user
files, update every preference and credential, run `jobapply doctor`, and use
a new dedicated Edge profile path if the LinkedIn account is different. Never
mix one person's browser session, resume, profile, or Telegram credentials with
another person's setup.
