"""Probe OpenRouter rate limiting for a specific chat-completion model.

The API key is read only from ``OPENROUTER_API_KEY``. It is never accepted as
a command-line argument, printed, or written to the JSON report.

Conservative smoke test (five sequential requests):

    $env:OPENROUTER_API_KEY = "<your-key>"
    python scripts/openrouter_rate_limit_test.py

Explicit burst test (may consume a substantial part of a free daily quota):

    python scripts/openrouter_rate_limit_test.py --requests 25 --concurrency 25

Add ``--output outputs/openrouter-rate-limit.json`` to save the measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_INFO_URL = "https://openrouter.ai/api/v1/key"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
RATE_HEADER_MARKERS = ("rate", "retry-after", "quota", "limit", "remaining", "reset")


@dataclass
class Result:
    request_id: int
    status: int | None
    latency_seconds: float
    started_seconds: float
    outcome: str
    retry_after: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    error: str = ""
    provider: str = ""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    if parsed > 100:
        raise argparse.ArgumentTypeError("must not exceed the safety cap of 100")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def selected_headers(headers: httpx.Headers) -> dict[str, str]:
    """Return potentially useful throttling headers without auth material."""
    return {
        name.lower(): value
        for name, value in headers.items()
        if any(marker in name.lower() for marker in RATE_HEADER_MARKERS)
    }


def error_message(response: httpx.Response) -> str:
    """Extract a short API error without dumping an entire response."""
    try:
        payload = response.json()
        error = payload.get("error", payload) if isinstance(payload, dict) else payload
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)[:300]
        return str(error)[:300]
    except (ValueError, TypeError):
        return response.text[:300]


async def get_key_info(client: httpx.AsyncClient) -> dict[str, Any] | None:
    """Fetch non-secret account/quota metadata when the endpoint is available."""
    try:
        response = await client.get(KEY_INFO_URL)
        if response.status_code != 200:
            print(f"Key metadata unavailable (HTTP {response.status_code}).")
            return None
        data = response.json().get("data", {})
        return {
            "is_free_tier": data.get("is_free_tier"),
            "usage": data.get("usage"),
            "usage_daily": data.get("usage_daily"),
            "limit": data.get("limit"),
            "limit_remaining": data.get("limit_remaining"),
            "limit_reset": data.get("limit_reset"),
            # OpenRouter marks this field deprecated; show it only as metadata.
            "deprecated_rate_limit": data.get("rate_limit"),
        }
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Key metadata unavailable ({type(exc).__name__}).")
        return None


async def send_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    request_id: int,
    epoch: float,
    model: str,
    prompt: str,
    max_tokens: int,
    start_delay: float,
) -> Result:
    """Send one measured completion request."""
    if start_delay:
        await asyncio.sleep(start_delay)

    async with semaphore:
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        try:
            response = await client.post(API_URL, json=payload)
            latency = time.perf_counter() - started
            headers = selected_headers(response.headers)
            provider = ""
            if response.status_code == 200:
                try:
                    provider = str(response.json().get("provider") or "")
                except ValueError:
                    pass
                outcome = "ok"
                error = ""
            elif response.status_code == 429:
                outcome = "rate_limited"
                error = error_message(response)
            elif response.status_code in {401, 403}:
                outcome = "auth_error"
                error = error_message(response)
            elif response.status_code >= 500:
                outcome = "upstream_error"
                error = error_message(response)
            else:
                outcome = "http_error"
                error = error_message(response)

            return Result(
                request_id=request_id,
                status=response.status_code,
                latency_seconds=latency,
                started_seconds=started - epoch,
                outcome=outcome,
                retry_after=response.headers.get("retry-after", ""),
                headers=headers,
                error=error,
                provider=provider,
            )
        except httpx.TimeoutException as exc:
            return Result(
                request_id=request_id,
                status=None,
                latency_seconds=time.perf_counter() - started,
                started_seconds=started - epoch,
                outcome="timeout",
                error=type(exc).__name__,
            )
        except httpx.HTTPError as exc:
            return Result(
                request_id=request_id,
                status=None,
                latency_seconds=time.perf_counter() - started,
                started_seconds=started - epoch,
                outcome="network_error",
                error=f"{type(exc).__name__}: {exc}"[:300],
            )


def print_results(results: list[Result]) -> None:
    print("\n #   start    status  latency   outcome")
    print("---  -------  ------  --------  --------------")
    for result in results:
        status = str(result.status) if result.status is not None else "-"
        print(
            f"{result.request_id:>3}  {result.started_seconds:>6.2f}s  "
            f"{status:>6}  {result.latency_seconds:>7.2f}s  {result.outcome}"
        )
        if result.retry_after:
            print(f"     Retry-After: {result.retry_after}")
        if result.error:
            print(f"     {result.error}")


def build_summary(results: list[Result], elapsed: float) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1

    successful_latencies = [r.latency_seconds for r in results if r.outcome == "ok"]
    first_429 = next((r.request_id for r in results if r.status == 429), None)
    rate_headers: dict[str, set[str]] = {}
    for result in results:
        for name, value in result.headers.items():
            rate_headers.setdefault(name, set()).add(value)

    return {
        "total_requests": len(results),
        "elapsed_seconds": round(elapsed, 3),
        "request_start_rate_per_minute": (
            round(len(results) / max(r.started_seconds for r in results) * 60, 2)
            if len(results) > 1 and max(r.started_seconds for r in results) > 0
            else None
        ),
        "outcomes": counts,
        "first_http_429_request": first_429,
        "successful_latency_seconds": {
            "min": round(min(successful_latencies), 3) if successful_latencies else None,
            "median": round(statistics.median(successful_latencies), 3)
            if successful_latencies
            else None,
            "max": round(max(successful_latencies), 3) if successful_latencies else None,
        },
        "observed_rate_headers": {
            name: sorted(values) for name, values in sorted(rate_headers.items())
        },
    }


async def run(args: argparse.Namespace) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY before running this script.", file=sys.stderr)
        return 2

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/jobapply-rate-limit-test",
        "X-Title": "JobApply rate-limit test",
    }
    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )

    print(f"Model:       {args.model}")
    print(f"Requests:    {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Start gap:   {args.interval:.3f}s")

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        key_info = None if args.skip_key_info else await get_key_info(client)
        if key_info:
            print(f"Key metadata: {json.dumps(key_info, sort_keys=True)}")

        semaphore = asyncio.Semaphore(args.concurrency)
        epoch = time.perf_counter()
        tasks = [
            send_one(
                client=client,
                semaphore=semaphore,
                request_id=index + 1,
                epoch=epoch,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                start_delay=index * args.interval,
            )
            for index in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - epoch

    results.sort(key=lambda item: item.request_id)
    print_results(results)
    summary = build_summary(results, elapsed)
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model": args.model,
            "configuration": {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "interval_seconds": args.interval,
                "max_tokens": args.max_tokens,
                "timeout_seconds": args.timeout,
            },
            "key_metadata": key_info,
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {output}")

    if any(result.outcome == "auth_error" for result in results):
        return 2
    if any(result.outcome == "rate_limited" for result in results):
        return 3
    if any(result.outcome != "ok" for result in results):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure OpenRouter throttling without storing or printing the API key."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", type=positive_int, default=5)
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument(
        "--interval",
        type=nonnegative_float,
        default=0.0,
        help="seconds between request starts (default: 0)",
    )
    parser.add_argument("--max-tokens", type=positive_int, default=1)
    parser.add_argument("--timeout", type=positive_int, default=180)
    parser.add_argument("--prompt", default="Reply with only: OK")
    parser.add_argument("--output", help="optional JSON report path")
    parser.add_argument("--skip-key-info", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.concurrency > args.requests:
        args.concurrency = args.requests
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
