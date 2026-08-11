# -*- coding: utf-8 -*-
"""Gemma API rate-limit tester.

Reads credentials directly from .env (no jobapply package needed on the path).

Uses the native Gemini REST endpoint:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=KEY

This endpoint supports ALL models including Gemma variants.

Burst phases
------------
  Phase A :  5 concurrent requests   (baseline)
  Phase B : 15 concurrent requests   (moderate load)
  Phase C : 30 concurrent requests   (aggressive burst)

Usage
-----
    python scripts/gemma_rate_limit_test.py                  # all 3 phases
    python scripts/gemma_rate_limit_test.py --phase A        # single phase
    python scripts/gemma_rate_limit_test.py --concurrency 50 # custom burst
    python scripts/gemma_rate_limit_test.py --prompt "Explain AI in one sentence"
"""

import argparse
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env(path: Path) -> dict[str, str]:
    """Minimal .env parser - no external dependencies."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split("#")[0].strip().strip("'\"")
        env[key.strip()] = val
    return env


def _mask(key: str) -> str:
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:5]}...{key[-5:]}"


def _banner(title: str, width: int = 66) -> str:
    bar = "=" * width
    return f"\n+{bar}+\n|  {title:<{width - 2}}|\n+{bar}+"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class Result:
    req_id: int
    phase: str
    status: int | None
    latency: float
    success: bool
    rate_limited: bool
    error_msg: str = ""
    retry_after: str = ""
    rl_headers: dict[str, str] = field(default_factory=dict)
    reply: str = ""


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

async def fire(
    req_id: int,
    phase: str,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 64,
    timeout: int = 30,
) -> Result:
    """Fire one request to the native Gemini generateContent endpoint.

    URL:  {base_url}/models/{model}:generateContent?key={api_key}
    Auth: query parameter (no Authorization header needed).
    """
    # Strip any accidental 'models/' prefix from model name
    model_id = model.removeprefix("models/")
    url = f"{base_url.rstrip('/')}/models/{model_id}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.1},
    }
    headers = {"Content-Type": "application/json"}

    def _send() -> tuple[int, dict[str, str], str]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (
                resp.status,
                {k.lower(): v for k, v in resp.headers.items()},
                resp.read().decode("utf-8", errors="replace"),
            )

    start = time.perf_counter()
    try:
        status, resp_hdrs, body_text = await asyncio.to_thread(_send)
        latency = time.perf_counter() - start

        rl_hdrs = {k: v for k, v in resp_hdrs.items() if "ratelimit" in k or "retry" in k}
        retry_after = resp_hdrs.get("retry-after", "")

        # Gemini response: {candidates: [{content: {parts: [{text: ...}]}}]}
        try:
            data = json.loads(body_text)
            reply = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")[:80]
            )
        except Exception:
            reply = body_text[:80]

        return Result(req_id=req_id, phase=phase, status=status, latency=latency,
                      success=True, rate_limited=False,
                      retry_after=retry_after, rl_headers=rl_hdrs, reply=reply)

    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - start
        err_body = exc.read()[:300].decode("utf-8", errors="replace")
        resp_hdrs = {k.lower(): v for k, v in exc.headers.items()}
        rl_hdrs = {k: v for k, v in resp_hdrs.items() if "ratelimit" in k or "retry" in k}
        retry_after = resp_hdrs.get("retry-after", "")
        is_rl = exc.code in (429, 503)
        return Result(req_id=req_id, phase=phase, status=exc.code, latency=latency,
                      success=False, rate_limited=is_rl,
                      error_msg=err_body, retry_after=retry_after, rl_headers=rl_hdrs)

    except Exception as exc:
        latency = time.perf_counter() - start
        return Result(req_id=req_id, phase=phase, status=None, latency=latency,
                      success=False, rate_limited=False,
                      error_msg=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Burst
# ---------------------------------------------------------------------------

async def run_burst(
    phase: str,
    concurrency: int,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
) -> list[Result]:
    """Fire `concurrency` requests simultaneously."""
    print(f"\n  Firing {concurrency} concurrent requests ...", flush=True)

    tasks = [fire(i + 1, phase, api_key, base_url, model, prompt) for i in range(concurrency)]
    results: list[Result] = await asyncio.gather(*tasks)

    col = [5, 7, 10, 10, 13, 32]
    sep = "  " + "-" * (sum(col) + len(col))
    hdr = (
        f"  {'#':<{col[0]}} {'Status':<{col[1]}} {'Latency':<{col[2]}}"
        f" {'Result':<{col[3]}} {'Retry-After':<{col[4]}} {'Detail':<{col[5]}}"
    )
    print(sep)
    print(hdr)
    print(sep)
    for r in sorted(results, key=lambda x: x.req_id):
        status_s = str(r.status) if r.status is not None else "ERR"
        result_s = "[RL]" if r.rate_limited else ("OK" if r.success else "FAIL")
        detail = r.reply[:30] if r.success else r.error_msg[:30]
        retry = r.retry_after or "-"
        print(
            f"  {r.req_id:<{col[0]}} {status_s:<{col[1]}} {r.latency:<{col[2]}.2f}"
            f" {result_s:<{col[3]}} {retry:<{col[4]}} {detail:<{col[5]}}"
        )
    print(sep)
    return results


# ---------------------------------------------------------------------------
# Phase summary
# ---------------------------------------------------------------------------

def print_summary(phase: str, concurrency: int, results: list[Result]) -> None:
    total = len(results)
    ok    = sum(1 for r in results if r.success)
    rl    = sum(1 for r in results if r.rate_limited)
    errs  = total - ok - rl
    lats  = [r.latency for r in results if r.success]

    print(f"\n  Phase {phase} summary  ({concurrency} requests sent)")
    print(f"  [OK]  Success      : {ok}/{total}")
    print(f"  [RL]  Rate limited : {rl}/{total}  (HTTP 429/503)")
    print(f"  [XX]  Other errors : {errs}/{total}")
    if lats:
        print(f"  [TM]  Latency      : avg={sum(lats)/len(lats):.2f}s  "
              f"min={min(lats):.2f}s  max={max(lats):.2f}s")

    # Collect unique RL header values
    all_rl: dict[str, set[str]] = {}
    for r in results:
        for k, v in r.rl_headers.items():
            all_rl.setdefault(k, set()).add(v)
    if all_rl:
        print("  [RL]  Rate-limit headers seen:")
        for k, vals in sorted(all_rl.items()):
            print(f"         {k}: {', '.join(sorted(vals))}")


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

async def warmup(api_key: str, base_url: str, model: str) -> bool:
    """Probe the endpoint. Returns True on success, False only on auth failure (401/403).

    Quota errors (429) and server errors (500) return True with a warning so
    the burst phases still run and capture real rate-limit data.
    """
    model_id = model.removeprefix("models/")
    full_url = f"{base_url.rstrip('/')}/models/{model_id}:generateContent"
    print(f"  Endpoint : {full_url}?key=...")
    print(f"  Model    : {model_id}")
    print(f"  API Key  : {_mask(api_key)}")
    print("\n  Warm-up call ... ", end="", flush=True)
    r = await fire(0, "warmup", api_key, base_url, model, "Reply with exactly one word: READY.")
    if r.success:
        print(f"OK  ({r.latency:.2f}s)")
        print(f"  Reply    : {r.reply!r}")
        return True
    print(f"HTTP {r.status}")
    print(f"  Error    : {r.error_msg[:300]}")
    # 401 / 403 = bad key -> abort.  429/500/503 = quota/overload -> still run burst.
    if r.status in (429, 500, 503):
        print("  [NOTE] Quota/overload detected - running burst phases to measure RL behaviour.")
        return True
    return False



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    env = _load_env(repo_root / ".env")

    api_key  = env.get("GOOGLE_API_KEY", "")
    base_url = env.get("JOBAPPLY_LLM_BASE_URL",
                       "https://generativelanguage.googleapis.com/v1beta")
    # Strip /openai suffix if present (legacy .env values)
    base_url = base_url.removesuffix("/openai")
    model = env.get("JOBAPPLY_LLM_MODEL", "gemma-4-31b-it")
    if model != "gemma-4-31b-it":
        print("ERROR: JOBAPPLY_LLM_MODEL must be gemma-4-31b-it.", file=sys.stderr)
        sys.exit(1)
    prompt   = args.prompt

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env - aborting.", file=sys.stderr)
        sys.exit(1)

    print(_banner("Gemma 4 31B - Rate Limit Tester"))

    ok = await warmup(api_key, base_url, model)
    if not ok:
        print("\nERROR: Warm-up failed - check your GOOGLE_API_KEY and model name.", file=sys.stderr)
        sys.exit(1)

    # Define phases
    phase_map = {"A": 5, "B": 15, "C": 30}

    if args.concurrency:
        phases_to_run = {"X": args.concurrency}
    elif args.phase:
        phases_to_run = {args.phase: phase_map[args.phase]}
    else:
        phases_to_run = phase_map

    all_results: dict[str, list[Result]] = {}

    for phase, concurrency in phases_to_run.items():
        print(_banner(f"Phase {phase} - {concurrency} concurrent requests"))
        results = await run_burst(phase, concurrency, api_key, base_url, model, prompt)
        all_results[phase] = results
        print_summary(phase, concurrency, results)

        if list(phases_to_run)[-1] != phase:
            print(f"\n  Cooling down 5s before next phase ...", flush=True)
            await asyncio.sleep(5)

    # Overall summary (only when multiple phases ran)
    if len(all_results) > 1:
        print(_banner("Overall Summary"))
        total_req = sum(len(v) for v in all_results.values())
        total_ok  = sum(r.success for v in all_results.values() for r in v)
        total_rl  = sum(r.rate_limited for v in all_results.values() for r in v)
        print(f"  Total requests        : {total_req}")
        print(f"  Succeeded             : {total_ok}")
        print(f"  Rate limited (429/503): {total_rl}")
        print("\n  Rate-limit threshold observation:")
        for phase, results in all_results.items():
            concurrency = phases_to_run[phase]
            rl = sum(1 for r in results if r.rate_limited)
            pct = rl / len(results) * 100
            filled = int(pct / 5)
            bar = "#" * filled + "-" * (20 - filled)
            print(f"    Phase {phase} ({concurrency:>3} reqs): [{bar}] {pct:5.1f}% rate limited")

    print("\n  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 4 31B rate limit tester")
    parser.add_argument(
        "--phase", choices=["A", "B", "C"],
        help="Run a single phase: A=5, B=15, C=30 concurrent. Default: all three.",
    )
    parser.add_argument(
        "--concurrency", type=int, metavar="N",
        help="Custom burst size (overrides --phase).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly one word: Pong.",
        help="Prompt sent in every request.",
    )
    asyncio.run(main(parser.parse_args()))
