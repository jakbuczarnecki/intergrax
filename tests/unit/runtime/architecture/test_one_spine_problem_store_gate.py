# © Artur Czarnecki. All rights reserved.

"""Architecture gate — production code must not construct side Problem stores (DIAG-FOUNDATION-5)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "wire_problem_persistence",
        "InMemoryProblemPersistence",
        "DocumentStoreProblemPersistence",
        "ProblemLifecycleEngine",
    }
)

_ALLOWED_RELATIVE_PREFIXES = (
    "intergrax/runtime/diagnostics/",
    "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
    "intergrax/applications/_shared/diagnostic_read_wiring.py",
)

_SCAN_ROOTS = (
    "intergrax/applications",
    "intergrax/runtime",
    "agents",
    "applications",
    "scripts",
)

_EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        "tests",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_allowed_path(path: Path) -> bool:
    rel = path.relative_to(_repo_root()).as_posix()
    return any(rel.startswith(prefix) for prefix in _ALLOWED_RELATIVE_PREFIXES)


def _production_python_files() -> list[Path]:
    root = _repo_root()
    files: list[Path] = []
    for scan_root in _SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            if _is_allowed_path(path):
                continue
            files.append(path)
    return files


def _collect_forbidden_symbol_references(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
            violations.append(f"{path.relative_to(_repo_root())}:{node.lineno} references {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            violations.append(
                f"{path.relative_to(_repo_root())}:{node.lineno} references .{node.attr}",
            )
    return violations


def test_production_code_cannot_construct_side_problem_stores() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        violations.extend(_collect_forbidden_symbol_references(path))
    assert violations == []
