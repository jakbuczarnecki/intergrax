# © Artur Czarnecki. All rights reserved.

"""UE-10R4 — graph authority fail-closed and canonical path quality gate."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GRAPH_EXECUTOR_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py"
)
_NEXUS_LOOP_PATH = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_EXECUTION_DIR = _REPO_ROOT / "intergrax" / "runtime" / "execution"

_FORBIDDEN_GRAPH_AUTHORITY_CALLS = frozenset(
    {
        "bind_active_execution_authority",
        "reset_active_execution_authority",
        "peek_active_execution_authority",
        "resolve_root_parent_execution_authority",
    }
)

_FORBIDDEN_QUALITY_PATTERNS = (
    re.compile(r"^\s*import\s+inspect\b"),
    re.compile(r"^\s*from\s+typing\s+import\b.*\bAny\b"),
    re.compile(r"\btyping\.Any\b"),
    re.compile(r"\bgetattr\("),
    re.compile(r"\bsetattr\("),
    re.compile(r"\bhasattr\("),
    re.compile(r"#\s*type:\s*ignore\b"),
)

_PRIVATE_GRAPH_EXECUTOR_WRITES = (
    "_graph_executor._validation_engine",
    "_graph_executor._critic_graph_hooks",
)


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_calls(path: Path, forbidden: frozenset[str]) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in forbidden:
            violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _quality_violations(path: Path) -> list[str]:
    rel = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        for pattern in _FORBIDDEN_QUALITY_PATTERNS:
            if pattern.search(line):
                violations.append(f"{rel}:{lineno}: {line.strip()}")
    return violations


def test_graph_executor_has_no_root_authority_fallback() -> None:
    violations = _collect_forbidden_calls(
        _GRAPH_EXECUTOR_PATH,
        _FORBIDDEN_GRAPH_AUTHORITY_CALLS,
    )
    assert violations == [], (
        "GraphExecutor must not bind or derive root authority: " + ", ".join(violations)
    )


def test_graph_executor_requires_active_execution_authority() -> None:
    source = _GRAPH_EXECUTOR_PATH.read_text(encoding="utf-8-sig")
    assert "require_active_execution_authority()" in source


@pytest.mark.parametrize(
    "path",
    [
        _GRAPH_EXECUTOR_PATH,
        _NEXUS_LOOP_PATH,
    ],
)
def test_canonical_files_have_no_forbidden_quality_constructions(path: Path) -> None:
    violations = _quality_violations(path)
    assert violations == [], (
        f"Forbidden quality construction in {path.name}: " + ", ".join(violations)
    )


def test_execution_package_has_no_forbidden_quality_constructions() -> None:
    violations: list[str] = []
    for path in sorted(_EXECUTION_DIR.rglob("*.py")):
        violations.extend(_quality_violations(path))
    assert violations == [], (
        "Forbidden quality construction in runtime/execution: " + ", ".join(violations)
    )


def test_nexus_loop_does_not_mutate_graph_executor_private_fields() -> None:
    source = _NEXUS_LOOP_PATH.read_text(encoding="utf-8-sig")
    violations = [token for token in _PRIVATE_GRAPH_EXECUTOR_WRITES if token in source]
    assert violations == [], (
        "NexusLoop must not mutate GraphExecutor private fields: " + ", ".join(violations)
    )
