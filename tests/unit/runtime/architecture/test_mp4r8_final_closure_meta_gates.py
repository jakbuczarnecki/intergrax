# © Artur Czarnecki. All rights reserved.

"""MP-4R8 — final closure meta-gates (aggregate invariants beyond R0–R7 slices)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"
_FORBIDDEN_RETIRED_IMPORTS = (
    "intergrax.contracts.approval",
    "intergrax.approval",
    "intergrax.runtime.human.legacy_human_input_compatibility",
)
_FORBIDDEN_GOD_ORCHESTRATORS = (
    "MultiplayerDecisionManager",
    "MultiplayerDecisionExecutionManager",
    "UnifiedDecisionExecutionService",
    "EnterpriseDecisionCoordinator",
)
_MINT_AUTH_FN = "mint_validated_execution_authorization"
_MINT_AUTH_ALLOWED_CALLERS = frozenset(
    {
        _REPO_ROOT / "intergrax" / "runtime" / "decision_flow.py",
        _REPO_ROOT / "intergrax" / "runtime" / "decision_authorization.py",
    },
)


def _production_py_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def test_mp4r8_no_retired_authority_imports_in_intergrax() -> None:
    violations: list[str] = []
    for module in _production_py_files(_INTERGRAX_ROOT):
        rel = module.relative_to(_REPO_ROOT)
        for lineno, imported in _collect_imports(module):
            for forbidden in _FORBIDDEN_RETIRED_IMPORTS:
                if imported == forbidden or imported.startswith(forbidden + "."):
                    violations.append(f"{rel}:{lineno} imports {imported}")
    assert not violations, "\n".join(violations)


def test_mp4r8_no_god_orchestrator_types_in_multiplayer_or_decision_runtime() -> None:
    roots = (
        _INTERGRAX_ROOT / "collaborative_work",
        _INTERGRAX_ROOT / "runtime" / "decision_flow.py",
        _INTERGRAX_ROOT / "runtime" / "decision_authorization.py",
        _INTERGRAX_ROOT / "runtime" / "decision_integration_composition.py",
        _INTERGRAX_ROOT / "runtime" / "decision_plugin_composition.py",
    )
    violations: list[str] = []
    for root in roots:
        if root.is_file():
            paths = [root]
        else:
            paths = _production_py_files(root)
        for module in paths:
            text = module.read_text(encoding="utf-8-sig")
            rel = module.relative_to(_REPO_ROOT)
            for name in _FORBIDDEN_GOD_ORCHESTRATORS:
                if re.search(rf"\bclass\s+{name}\b", text):
                    violations.append(f"{rel} defines forbidden orchestrator {name}")
    assert not violations, "\n".join(violations)


def test_mp4r8_execution_authorization_minted_only_via_decision_flow_in_runtime() -> None:
    runtime_root = _INTERGRAX_ROOT / "runtime"
    violations: list[str] = []
    for module in _production_py_files(runtime_root):
        if module.name == "decision_authorization.py":
            continue
        text = module.read_text(encoding="utf-8-sig")
        if _MINT_AUTH_FN in text:
            rel = module.relative_to(_REPO_ROOT)
            if module.resolve() not in _MINT_AUTH_ALLOWED_CALLERS:
                violations.append(f"{rel} references {_MINT_AUTH_FN}")
    assert not violations, "\n".join(violations)


def test_mp4r8_task_control_hitl_resume_does_not_synthesize_local_dev_approver() -> None:
    path = _INTERGRAX_ROOT / "applications" / "_shared" / "task_control.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    target: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_materialize_hitl_resume_input":
            target = node
            break
    assert target is not None, "_materialize_hitl_resume_input missing from task_control.py"
    for child in ast.walk(target):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id == "local_development_approver_evidence":
                pytest.fail(
                    "_materialize_hitl_resume_input must not call local_development_approver_evidence"
                )
            if isinstance(func, ast.Attribute) and func.attr == "local_development_approver_evidence":
                pytest.fail(
                    "_materialize_hitl_resume_input must not call local_development_approver_evidence"
                )


def test_mp4r8_collaborative_work_does_not_mint_governance_or_execution_authorization() -> None:
    cw_root = _INTERGRAX_ROOT / "collaborative_work"
    forbidden_tokens = (
        _MINT_AUTH_FN,
        "DecisionGovernanceDecision(",
        "decision_execution_authorization(",
    )
    violations: list[str] = []
    for module in _production_py_files(cw_root):
        text = module.read_text(encoding="utf-8-sig")
        rel = module.relative_to(_REPO_ROOT)
        for token in forbidden_tokens:
            if token in text:
                violations.append(f"{rel} contains forbidden authority token {token!r}")
    assert not violations, "\n".join(violations)
