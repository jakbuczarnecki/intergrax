# © Artur Czarnecki. All rights reserved.

"""GR-2-R3 mandatory MODEL C1 architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.runtime.architecture.gr2_r3_model_c1_gate_policy import (
    AUTHORITY_RESOLUTION_ALLOWLIST,
    INTERNAL_ROOT_ENGINE_ALLOWLIST,
    PRODUCTION_SCAN_ROOTS,
    REPO_ROOT,
    TEST_TREE_PREFIXES,
)

pytestmark = pytest.mark.unit


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _iter_production_python_files() -> list[Path]:
    paths: list[Path] = []
    for root in PRODUCTION_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "docker" in path.parts or "runtime-context" in path.parts:
                continue
            rel = _rel(path)
            if rel.startswith(TEST_TREE_PREFIXES):
                continue
            if "/tests/" in f"/{rel}/":
                continue
            if path.name.startswith("test_"):
                continue
            paths.append(path)
    return paths


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_forbidden_root_construction(path: Path) -> list[str]:
    rel = _rel(path)
    if rel in INTERNAL_ROOT_ENGINE_ALLOWLIST:
        return []
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in {"RootExecutionOptions", "CanonicalExecutionIntakeRequest"}:
            violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


def _collect_forbidden_root_authority_resolution(path: Path) -> list[str]:
    rel = _rel(path)
    if rel in INTERNAL_ROOT_ENGINE_ALLOWLIST or rel in AUTHORITY_RESOLUTION_ALLOWLIST:
        return []
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "resolve_root_parent_execution_authority":
            continue
        if rel == "intergrax/runtime/execution/host_task.py":
            continue
        violations.append(f"{rel}:{node.lineno}: resolve_root_parent_execution_authority()")
    return violations


def _collect_forbidden_facade_root_execute(path: Path) -> list[str]:
    rel = _rel(path)
    if rel in INTERNAL_ROOT_ENGINE_ALLOWLIST:
        return []
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "execute":
            continue
        if _call_name(node.func.value) != "Execution":
            continue
        violations.append(f"{rel}:{node.lineno}: Execution.execute()")
    return violations


def test_production_has_no_unauthorized_root_construction() -> None:
    violations: list[str] = []
    for path in _iter_production_python_files():
        violations.extend(_collect_forbidden_root_construction(path))
    assert violations == [], "forbidden root construction: " + ", ".join(violations)


def test_production_has_no_unauthorized_root_authority_resolution() -> None:
    violations: list[str] = []
    for path in _iter_production_python_files():
        violations.extend(_collect_forbidden_root_authority_resolution(path))
    assert violations == [], "forbidden root authority resolution: " + ", ".join(violations)


def test_production_has_no_unauthorized_execution_facade_root_calls() -> None:
    violations: list[str] = []
    for path in _iter_production_python_files():
        violations.extend(_collect_forbidden_facade_root_execute(path))
    assert violations == [], "forbidden Execution.execute: " + ", ".join(violations)


def test_gate_detects_synthetic_root_bypass_fixture() -> None:
    fixture = Path(__file__).with_name("gr2_r3_synthetic_bypass_fixture.py")
    fixture.write_text(
        "from intergrax.runtime.execution.facade import Execution\n"
        "from intergrax.runtime.execution.runtime import RootExecutionOptions\n"
        "from intergrax.contracts.execution_intake import CanonicalExecutionIntakeRequest\n"
        "from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority\n"
        "def bypass():\n"
        "    RootExecutionOptions(authority=resolve_root_parent_execution_authority(None))\n"
        "    CanonicalExecutionIntakeRequest(payload=object(), trusted_parent_execution_authority=object(), tenant_id='t')\n"
        "    Execution(object()).execute(object(), options=RootExecutionOptions(authority=object()))\n",
        encoding="utf-8",
    )
    violations = (
        _collect_forbidden_root_construction(fixture)
        + _collect_forbidden_root_authority_resolution(fixture)
        + _collect_forbidden_facade_root_execute(fixture)
    )
    fixture.unlink(missing_ok=True)
    assert len(violations) >= 3
