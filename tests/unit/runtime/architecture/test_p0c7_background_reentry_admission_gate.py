# © Artur Czarnecki. All rights reserved.

"""P0C-7 — production workers must use canonical re-entry admission."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_PRODUCTION_WORKER_PATHS = (
    _REPO_ROOT / "intergrax" / "background_tasks" / "worker_runtime.py",
    _REPO_ROOT / "intergrax" / "queueing" / "providers" / "broker_worker_base.py",
    _REPO_ROOT / "intergrax" / "queueing" / "worker" / "dispatcher.py",
    _REPO_ROOT / "intergrax" / "queueing" / "providers" / "document_store" / "colocated_worker.py",
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_bootstrap_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) == "bootstrap_background_execution":
            violations.append(f"{rel}:{node.lineno}: bootstrap_background_execution()")
    return violations


def test_production_workers_do_not_call_bootstrap_background_execution_directly() -> None:
    violations: list[str] = []
    for path in _PRODUCTION_WORKER_PATHS:
        violations.extend(_collect_bootstrap_calls(path))
    assert violations == [], (
        "production background workers must delegate to admit_background_execution_reentry: "
        + ", ".join(violations)
    )


def _collect_reentry_calls(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) == "admit_background_execution_reentry":
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("path", _PRODUCTION_WORKER_PATHS, ids=lambda p: p.name)
def test_production_workers_invoke_canonical_reentry_admission(path: Path) -> None:
    assert _collect_reentry_calls(path), f"{path.name} must call admit_background_execution_reentry"
