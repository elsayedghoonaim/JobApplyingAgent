"""Guardrail: workflow code must never print directly.

All operator-facing output belongs in the CLI layer (main.py, doctor.py,
review_cli.py) or structured `log_event` calls. This test walks the AST of
every workflow module and fails when a bare print() call is introduced.
"""

import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "jobapply"

# Workflow modules where direct prints are forbidden. The operator CLI modules
# (main.py, doctor.py, review_cli.py, live.py) intentionally print for humans.
WORKFLOW_DIRECTORIES = ("nodes", "execution", "models", "utils")
WORKFLOW_FILES = ("graph.py", "state.py", "settings.py")


def _iter_workflow_files():
    for directory in WORKFLOW_DIRECTORIES:
        yield from sorted((SRC_ROOT / directory).rglob("*.py"))
    for name in WORKFLOW_FILES:
        path = SRC_ROOT / name
        if path.exists():
            yield path


def _print_call_locations(path: Path) -> list[int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - syntax errors fail other gates
        return []
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_print_name = isinstance(func, ast.Name) and func.id == "print"
        is_print_attr = (
            isinstance(func, ast.Attribute)
            and func.attr == "print"
            and isinstance(func.value, ast.Name)
            and func.value.id in {"sys", "console"}
        )
        if is_print_name or is_print_attr:
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("path", list(_iter_workflow_files()), ids=lambda p: p.name)
def test_no_direct_prints_in_workflow_code(path: Path):
    locations = _print_call_locations(path)
    assert locations == [], f"Direct print() found in {path} at lines {locations}"


def test_operator_cli_modules_are_the_only_expected_print_users():
    """Sanity-check that the guardrail actually scans real workflow files."""
    scanned = list(_iter_workflow_files())
    assert len(scanned) > 30
    names = {p.name for p in scanned}
    assert "search.py" in names
    assert "telegram.py" in names
