# © Artur Czarnecki. All rights reserved.

"""NPSC-4 architecture gates — agent runtime governance ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GOVERNANCE_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "agent_governance"
_CONTRACTS = _REPO_ROOT / "intergrax" / "contracts" / "agent_runtime_governance.py"
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_IDENTITY_AUTHORITY = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "identity_authority.py"
_EXECUTION_RUNTIME = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "runtime.py"

_FORBIDDEN_IDENTITY_MINT_CALLS = (
    "mint_run_id",
    "mint_attempt_id",
    "mint_execution_id",
    "mint_task_id",
    "mint_root_execution_identity",
)


def _python_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def _source_calls(path: Path, names: tuple[str, ...]) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in names:
                hits.append(f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}")
    return hits


@pytest.mark.gate
def test_npsc4_governance_cannot_mint_execution_identity() -> None:
    violations: list[str] = []
    for path in _python_files(_GOVERNANCE_ROOT):
        violations.extend(_source_calls(path, _FORBIDDEN_IDENTITY_MINT_CALLS))
    assert violations == [], (
        "agent_governance must not mint execution identity:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc4_governance_contracts_do_not_mint_execution_identity() -> None:
    violations = _source_calls(_CONTRACTS, _FORBIDDEN_IDENTITY_MINT_CALLS)
    assert violations == [], (
        "agent_runtime_governance contracts must not mint execution identity:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc4_tool_invoker_integrates_governance_before_execution() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    assert "agent_runtime_governance" in source
    assert "_require_agent_runtime_governance" in source
    assert "AgentRuntimeGovernancePort" in source
    auth_pos = source.index("def _require_current_attempt_authorization")
    governance_pos = source.index("self._require_agent_runtime_governance")
    scope_pos = source.index("if self._scope_policy is not None", auth_pos)
    assert governance_pos < scope_pos, (
        "governance must be evaluated before scope policy in authorization chain"
    )


@pytest.mark.gate
def test_npsc4_nexus_does_not_own_governance_contracts() -> None:
    nexus_root = _REPO_ROOT / "intergrax" / "runtime" / "nexus"
    forbidden_defs = {"AgentIdentity", "CapabilityGrant", "ToolAuthorizationRequest"}
    violations: list[str] = []
    for path in _python_files(nexus_root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in forbidden_defs:
                rel = path.relative_to(_REPO_ROOT).as_posix()
                violations.append(f"{rel}:{node.lineno}: class {node.name}")
    assert violations == [], (
        "Nexus must not define NPSC-4 governance contracts:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc4_execution_runtime_unchanged_ownership() -> None:
    """Governance layer must not modify ExecutionRuntime lifecycle ownership."""
    runtime_source = _EXECUTION_RUNTIME.read_text(encoding="utf-8")
    assert "class ExecutionRuntime" in runtime_source
    assert "agent_governance" not in runtime_source
    assert "AgentRuntimeGovernance" not in runtime_source


@pytest.mark.gate
def test_npsc4_identity_authority_unchanged() -> None:
    authority_source = _IDENTITY_AUTHORITY.read_text(encoding="utf-8")
    assert "agent_governance" not in authority_source
