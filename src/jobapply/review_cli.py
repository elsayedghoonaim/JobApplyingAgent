"""`jobapply review` — bounded, redacted operator commands for the manual-review queue.

Mutations use exact atomic state transitions and clearly report no-op,
conflict, and not-found outcomes. Resolving a ``submission_unknown`` item never
marks it submitted and requires an explicit safe resolution label; no browser
retry is ever triggered. Every displayed field is redacted and bounded, storage
failures produce a deterministic nonzero exit with bounded output (never a
traceback), and the shared Mongo client is always closed.
"""

from typing import Any, Optional

from jobapply.models.manual_review import ManualReviewState, ResolutionLabel
from jobapply.utils.dedup import bound_string
from jobapply.utils.manual_review import ManualReviewRepository
from jobapply.utils.mongo import MongoClientManager
from jobapply.utils.redaction import redact_string

EXIT_OK = 0
EXIT_CONFLICT = 3
EXIT_NOT_FOUND = 4
EXIT_STORAGE_ERROR = 5

MAX_DISPLAY_FIELD_LENGTH = 80


def _display(value: Any, max_length: int = MAX_DISPLAY_FIELD_LENGTH) -> str:
    """Redact and bound any operator-facing field."""
    if value is None:
        return "-"
    return bound_string(redact_string(str(value)), max_length=max_length) or "-"


def _print_item(index: int, item) -> None:
    ambiguous = " [AMBIGUOUS-SUBMISSION]" if item.ambiguous_submission else ""
    label = f" label={_display(item.resolution_label)}" if item.resolution_label else ""
    note = f" note={_display(item.operator_note)}" if item.operator_note else ""
    print(
        f"{index}. {_display(item.item_id, 64)}\n"
        f"   state={_display(item.state.value)}{ambiguous}{label}{note}\n"
        f"   run={_display(item.run_id, 32)} job={_display(item.job_id, 40)}\n"
        f"   title={_display(item.title)} | company={_display(item.company)}\n"
        f"   reason={_display(item.reason_category, 60)}"
        + (f": {_display(item.reason_detail, 100)}" if item.reason_detail else "")
        + f"\n   url={_display(item.url, 120)}"
    )


async def run_review_cli(args) -> int:
    """Execute one `jobapply review` subcommand and return its exit code."""
    try:
        repo = ManualReviewRepository()

        if args.review_command == "list":
            state = None if args.state == "all" else ManualReviewState(args.state)
            items = await repo.list_items(limit=args.limit, state=state)
            if not items:
                print("Manual-review queue is empty (no matching items).")
                return EXIT_OK
            for index, item in enumerate(items, 1):
                _print_item(index, item)
            print(f"\n{len(items)} item(s) listed.")
            return EXIT_OK

        if args.review_command == "acknowledge":
            result = await repo.acknowledge(args.item_id, note=args.note)
            if result.changed:
                print(f"Acknowledged {_display(result.item_id, 64)}.")
                return EXIT_OK
            if result.not_found:
                print(f"Not found: {_display(args.item_id, 64)}")
                return EXIT_NOT_FOUND
            if result.no_op:
                state_text = result.state.value if result.state else "transitioned"
                print(f"No-op: item is already {_display(state_text, 32)}.")
                return EXIT_OK
            print(f"Conflict: cannot acknowledge ({_display(result.reason, 80)}).")
            return EXIT_CONFLICT

        if args.review_command == "resolve":
            label = ResolutionLabel(args.resolution_label) if args.resolution_label else None
            result = await repo.resolve(args.item_id, note=args.note, resolution_label=label)
            if result.changed:
                print(
                    f"Resolved {_display(result.item_id, 64)} (label={label.value if label else 'unspecified'})."
                )
                return EXIT_OK
            if result.not_found:
                print(f"Not found: {_display(args.item_id, 64)}")
                return EXIT_NOT_FOUND
            if result.no_op and not result.conflict:
                print("No-op: item is already resolved.")
                return EXIT_OK
            if (
                result.conflict
                and result.reason == "submission_unknown_requires_explicit_resolution_label"
            ):
                print(
                    "Conflict: this item has an ambiguous submission (submission_unknown). "
                    "Provide --label confirmed_not_submitted|confirmed_submitted|requires_followup. "
                    "Resolving never marks it submitted and triggers no retry."
                )
                return EXIT_CONFLICT
            print(f"Conflict: cannot resolve ({_display(result.reason, 80)}).")
            return EXIT_CONFLICT

        return EXIT_CONFLICT
    except Exception as exc:
        # Bounded, class-level diagnosis only: raw storage errors can carry
        # credential-bearing connection strings.
        print(f"Storage error: manual-review queue unavailable ({type(exc).__name__}).")
        return EXIT_STORAGE_ERROR
    finally:
        try:
            await MongoClientManager.close()
        except Exception:
            pass


def summarize_pending_for_notification(pending: Optional[list]) -> str | None:
    """One bounded summary line about un-flushed queue items (used by tests)."""
    count = len(pending or [])
    if not count:
        return None
    return f"{count} manual-review queue item(s) pending durable enqueue"
