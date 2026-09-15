# © Artur Czarnecki. All rights reserved.

"""GR-3 mandatory inner enforcement architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.unit.runtime.architecture.gr3_inner_enforcement_ast import (
    InnerEnforcementViolation,
    collect_forbidden_concrete_inner_guard_imports,
    collect_unauthorized_authorize_and_execute_calls,
)
from tests.unit.runtime.architecture.gr3_inner_enforcement_gate_policy import (
    AUTHORIZE_AND_EXECUTE_CALL_ALLOWLIST,
    MEANINGFUL_SIDE_EFFECT_POLICY_BOUNDARY_REL,
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


def _parse_fixture(source: str, name: str = "fixture.py") -> tuple[Path, ast.AST, str]:
    fixture = Path(__file__).with_name(name)
    rel = _rel(fixture)
    tree = ast.parse(source, filename=str(fixture))
    return fixture, tree, rel


def test_production_authorize_and_execute_calls_are_allowlisted() -> None:
    violations: list[InnerEnforcementViolation] = []
    for path in _iter_production_python_files():
        rel = _rel(path)
        if rel in AUTHORIZE_AND_EXECUTE_CALL_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        violations.extend(
            collect_unauthorized_authorize_and_execute_calls(tree, rel_path=rel),
        )
    messages = [v.as_message() for v in violations]
    assert violations == [], messages


def test_gate_detects_synthetic_unauthorized_authorize_and_execute() -> None:
    _, tree, rel = _parse_fixture(
        """
class Boundary:
    def authorize_and_execute(self, request, execute):
        return execute()

def rogue(boundary, request):
    boundary.authorize_and_execute(request, lambda: 1)
""",
        name="_gr3_violating_fixture.py",
    )
    violations = collect_unauthorized_authorize_and_execute_calls(tree, rel_path=rel)
    assert len(violations) == 1


def test_meaningful_side_effect_policy_boundary_has_no_concrete_guard_import() -> None:
    path = REPO_ROOT / MEANINGFUL_SIDE_EFFECT_POLICY_BOUNDARY_REL
    rel = _rel(path)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    violations = collect_forbidden_concrete_inner_guard_imports(tree, rel_path=rel)
    messages = [v.as_message() for v in violations]
    assert violations == [], messages


def test_gate_detects_synthetic_concrete_guard_import_in_policy_boundary() -> None:
    _, tree, rel = _parse_fixture(
        """
from intergrax.runtime.governance.canonical_inner_execution_guard import (
    DefaultCanonicalInnerExecutionGuard,
)
from intergrax.contracts.canonical_inner_governance import CanonicalInnerExecutionGuardPort
""",
        name="_gr3_policy_boundary_guard_import_fixture.py",
    )
    violations = collect_forbidden_concrete_inner_guard_imports(tree, rel_path=rel)
    assert len(violations) == 1


def test_gate_negative_control_contract_import_in_policy_boundary() -> None:
    _, tree, rel = _parse_fixture(
        """
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
    CanonicalInnerGovernanceViolation,
)
""",
        name="_gr3_policy_boundary_contract_import_fixture.py",
    )
    violations = collect_forbidden_concrete_inner_guard_imports(tree, rel_path=rel)
    assert violations == []


def test_gate_negative_control_allowlisted_module_shape() -> None:
    _, tree, rel = _parse_fixture(
        """
def helper(boundary, request, execute):
    return boundary.authorize(request)
""",
        name="_gr3_negative_fixture.py",
    )
    violations = collect_unauthorized_authorize_and_execute_calls(tree, rel_path=rel)
    assert violations == []
