"""Gemma rate-limit test entry point.

The old version made one request through ``google-genai``, which is not a
project dependency and cannot reveal a rate limit. The actual tester uses the
same native Google REST endpoint and .env configuration as the application.

Examples:
    python scripts/stress_test.py
    python scripts/stress_test.py --phase A
    python scripts/stress_test.py --concurrency 12
"""

import argparse
import asyncio

try:
    from gemma_rate_limit_test import main
except ModuleNotFoundError:
    from scripts.gemma_rate_limit_test import main


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    if parsed > 100:
        raise argparse.ArgumentTypeError("must not exceed the safety cap of 100")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemma rate-limit tester")
    parser.add_argument(
        "--phase",
        choices=["A", "B", "C"],
        help="A=5, B=15, or C=30 concurrent requests (default: all phases)",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        metavar="N",
        help="custom concurrent burst size; overrides --phase (maximum: 100)",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly one word: Pong.",
        help="prompt sent in every request",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(build_parser().parse_args())))
