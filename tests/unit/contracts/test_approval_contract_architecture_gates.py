# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for MP-4C Approval contract surface."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_PATH = _REPO_ROOT / "intergrax" / "contracts" / "approval.py"
_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime.nexus",
    "intergrax.collaborative_work.repository",
    "intergrax.collaborative_work.service",
    "intergrax.collaborative_work.in_memory_repository",
    "intergrax.collaborative_work.artifact_service",
)
_FORBIDDEN_IMPORT_SUBSTRINGS = (
    "GraphExecutor",
    "TaskState",
    "ApprovalService",
    "Repository",
    "Database",
    "ProofReceipt",
)
_FORBIDDEN_AST_NAMES = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "vars",
        "Any",
    },
)
_FORBIDDEN_ATTRIBUTE_NAMES = frozenset({"__dict__"})
_REQUIRED_SYMBOLS = (
    "class ApprovalRequest",
    "class ApprovalOutcome",
    "class ApprovalReferences",
    "class ApprovalLifecycleState",
    "class HumanApprovalAction",
    "class ApprovalContractInvariantError",
    "def validate_approval_scope",
    "def validate_approval_identity",
    "def validate_approval_transition",
    "def validate_approval_references",
    "SCHEMA_APPROVAL_REQUEST_V1",
    "SCHEMA_APPROVAL_OUTCOME_V1",
    "SCHEMA_HUMAN_APPROVAL_ACTION_V1",
)


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_forbidden_ast_usage(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_AST_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRIBUTE_NAMES:
            violations.append(f"{path.name}:{node.lineno} references forbidden name {node.attr}")
    return violations


def test_mp4c_contract_module_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for lineno, module in _collect_imports(_CONTRACT_PATH):
        if any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in _FORBIDDEN_IMPORT_PREFIXES
        ):
            violations.append(f"{_CONTRACT_PATH.name}:{lineno} imports {module}")
        if any(token in module for token in _FORBIDDEN_IMPORT_SUBSTRINGS):
            violations.append(f"{_CONTRACT_PATH.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp4c_contract_module_has_no_forbidden_dynamic_patterns() -> None:
    violations = _collect_forbidden_ast_usage(_CONTRACT_PATH)
    assert not violations, "\n".join(violations)


def test_mp4c_contract_module_defines_approval_surface() -> None:
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    for symbol in _REQUIRED_SYMBOLS:
        assert symbol in source


def test_mp4c_contract_module_has_no_forbidden_approval_fields() -> None:
    forbidden_fields = (
        "artifact_status",
        "approval_status",
        "execution_status",
        "governance_status",
        "task_state",
        "runtime_state",
        "metadata: dict",
        "payload: Any",
        "ExecutionPausedForApproval",
        "ExecutionWaitingForApproval",
        "ExecutionApprovalGate",
        "ProofReceipt",
    )
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    violations = [field for field in forbidden_fields if field in source]
    assert not violations, "\n".join(violations)
