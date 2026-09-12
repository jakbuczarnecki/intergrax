# © Artur Czarnecki. All rights reserved.

"""NPSC-4.2-H1 — Execution ↔ Governance dependency boundary freeze certification."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOUNDARY_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_GOVERNANCE_BOUNDARY_MODEL.md"
)
_P0_INVENTORY = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "PLATFORM_EXECUTION_UNIFICATION_P0_BYPASS_INVENTORY.md"
)
_EXECUTION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "execution"
_GOVERNANCE_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "governance"
_AGENT_GOV_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "agent_governance"
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_PARTIAL_RECOVERY = _EXECUTION_ROOT / "fan_out_partial_recovery.py"
_RUNTIME_PATH = _EXECUTION_ROOT / "runtime.py"

_REQUIRED_DOC_SECTIONS = (
    "## Global Freeze Statement",
    "## Canonical decision flow",
    "## Ownership matrix",
    "## Identity integration",
    "## Evidence integration",
    "## Recovery integration",
    "## HITL integration",
)

_FROZEN_GATES = (
    "tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py",
    "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py",
)

_FORBIDDEN_EXECUTION_IMPORT_PREFIXES = (
    "intergrax.runtime.agent_governance.policy_engine",
    "intergrax.runtime.policy.policy_engine",
    "intergrax.runtime.policy.runtime_policy_engine",
    "intergrax.runtime.governance.execution_guard",
)

_FORBIDDEN_EXECUTION_CALLS = frozenset(
    {
        "evaluate_replay",
        "authorize_tool",
        "compose_policy_decisions",
    },
)

_IDENTITY_MINT_CALLS = frozenset(
    {
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
        "mint_root_execution_identity",
        "mint_child_execution_id",
        "mint_retry_attempt_id",
    },
)

_EVIDENCE_OWNERSHIP_TOKENS = (
    "RuntimeEventPersistence",
    "RuntimeEventBus",
)


def _iter_python_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and "tests" not in path.parts
    ]


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_import_violations(root: Path, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden_prefixes:
                    if node.module == prefix or node.module.startswith(prefix + "."):
                        violations.append(f"{rel}:{node.lineno}: import from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden_prefixes:
                        if alias.name == prefix or alias.name.startswith(prefix + "."):
                            violations.append(f"{rel}:{node.lineno}: import {alias.name}")
    return violations


def _collect_call_violations(root: Path, forbidden: frozenset[str]) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(root):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name in forbidden:
                violations.append(f"{rel}:{node.lineno}: {name}()")
    return violations


@pytest.mark.gate
def test_npsc42_h1_boundary_document_present_and_structured() -> None:
    assert _BOUNDARY_DOC.is_file(), "EXECUTION_GOVERNANCE_BOUNDARY_MODEL.md missing"
    text = _BOUNDARY_DOC.read_text(encoding="utf-8")
    for heading in _REQUIRED_DOC_SECTIONS:
        assert heading in text, f"missing section {heading!r}"
    assert "ExecutionRuntime" in text
    assert "Governance Evaluation Port" in text


@pytest.mark.gate
def test_npsc42_h1_frozen_gates_exist() -> None:
    missing = [rel for rel in _FROZEN_GATES if not (_REPO_ROOT / rel).is_file()]
    assert missing == [], f"frozen governance gates missing: {missing}"


@pytest.mark.gate
def test_npsc42_h1_p0_inventory_reports_zero_bypass() -> None:
    text = _P0_INVENTORY.read_text(encoding="utf-8")
    match = re.search(r"^\| BYPASS \| (\d+) \|", text, flags=re.MULTILINE)
    assert match is not None, "P0 inventory missing BYPASS metric row"
    assert int(match.group(1)) == 0


@pytest.mark.gate
def test_npsc42_h1_execution_does_not_host_policy_engine() -> None:
    import_hits = _collect_import_violations(
        _EXECUTION_ROOT,
        _FORBIDDEN_EXECUTION_IMPORT_PREFIXES,
    )
    call_hits = _collect_call_violations(_EXECUTION_ROOT, _FORBIDDEN_EXECUTION_CALLS)
    violations = import_hits + call_hits
    assert violations == [], "execution must not own governance evaluation:\n" + "\n".join(
        violations,
    )


@pytest.mark.gate
def test_npsc42_h1_execution_runtime_has_no_governance_coupling() -> None:
    source = _RUNTIME_PATH.read_text(encoding="utf-8")
    assert "agent_governance" not in source
    assert "PolicyEngine" not in source
    assert "evaluate_replay" not in source


@pytest.mark.gate
def test_npsc42_h1_governance_planes_do_not_mint_execution_identity() -> None:
    violations: list[str] = []
    for root in (_GOVERNANCE_ROOT, _AGENT_GOV_ROOT):
        violations.extend(_collect_call_violations(root, _IDENTITY_MINT_CALLS))
    assert violations == [], (
        "governance planes must not mint execution identity:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc42_h1_governance_does_not_own_evidence_plane() -> None:
    violations: list[str] = []
    for root in (_GOVERNANCE_ROOT, _AGENT_GOV_ROOT):
        for path in _iter_python_files(root):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            source = path.read_text(encoding="utf-8")
            for token in _EVIDENCE_OWNERSHIP_TOKENS:
                if token in source:
                    violations.append(f"{rel}: references {token}")
    assert violations == [], (
        "governance must not own evidence persistence/bus:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc42_h1_tool_invoker_governance_before_scope() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    assert "_require_agent_runtime_governance" in source
    auth_pos = source.index("def _require_current_attempt_authorization")
    governance_pos = source.index("self._require_agent_runtime_governance")
    scope_pos = source.index("if self._scope_policy is not None", auth_pos)
    assert governance_pos < scope_pos


@pytest.mark.gate
def test_npsc42_h1_partial_recovery_consumes_governance_deny_only() -> None:
    source = _PARTIAL_RECOVERY.read_text(encoding="utf-8")
    assert "PolicyDecision" in source
    assert "PolicyAction.DENY" in source
    assert "evaluate_replay" not in source
    assert "PolicyEngine" not in source
    allow_pos = source.find("PolicyAction.ALLOW")
    assert allow_pos == -1, "partial recovery must not mint ALLOW governance decisions"


@pytest.mark.gate
def test_npsc42_h1_no_bypass_policy_antipattern_in_runtime() -> None:
    runtime_root = _REPO_ROOT / "intergrax" / "runtime"
    forbidden_tokens = ("bypass_policy", "hidden_allow", "local_policy(", "internal_decision_engine")
    violations: list[str] = []
    for path in _iter_python_files(runtime_root):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                violations.append(f"{rel}: forbidden token {token!r}")
    assert violations == [], "\n".join(violations)
