# © Artur Czarnecki. All rights reserved.

"""GR-10-R3-R1: author-facing agent_step contract must stay implementation-neutral."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENT_STEP_PATH = _REPO_ROOT / "intergrax" / "contracts" / "agent_step.py"

_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.contracts.step_execution",
        "intergrax.runtime.kernel",
        "intergrax.runtime.kernel.step_kernel",
    }
)

_FORBIDDEN_IMPORT_NAMES = frozenset({"StepExecutionRecord", "HarnessKernel"})


@dataclass(frozen=True, slots=True)
class AgentStepBoundaryViolation:
    line: int
    rule: str


def _module_level_import_from_nodes(tree: ast.AST) -> list[ast.ImportFrom]:
    if not isinstance(tree, ast.Module):
        return []
    imports: list[ast.ImportFrom] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Name) and stmt.test.id == "TYPE_CHECKING":
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.ImportFrom):
                imports.append(node)
    return imports


def collect_agent_step_boundary_violations(source: str) -> list[AgentStepBoundaryViolation]:
    tree = ast.parse(source)
    violations: list[AgentStepBoundaryViolation] = []
    for node in _module_level_import_from_nodes(tree):
        module = node.module
        if module in _FORBIDDEN_IMPORT_MODULES:
            violations.append(
                AgentStepBoundaryViolation(
                    line=node.lineno,
                    rule=f"forbidden import from harness-owned module: {module}",
                )
            )
        for alias in node.names:
            if alias.name in _FORBIDDEN_IMPORT_NAMES:
                violations.append(
                    AgentStepBoundaryViolation(
                        line=node.lineno,
                        rule=f"forbidden harness-owned symbol import: {alias.name}",
                    )
                )
    return violations


def test_agent_step_contract_has_no_harness_owned_imports() -> None:
    source = _AGENT_STEP_PATH.read_text(encoding="utf-8")
    violations = collect_agent_step_boundary_violations(source)
    assert not violations, "; ".join(f"L{v.line}: {v.rule}" for v in violations)


def test_step_execution_result_fields_exclude_kernel_record() -> None:
    from intergrax.contracts.agent_step import StepExecutionResult

    assert "kernel_step_record" not in StepExecutionResult.model_fields
