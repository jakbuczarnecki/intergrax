# © Artur Czarnecki. All rights reserved.

"""AST collectors for GR-3 inner enforcement architecture gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InnerEnforcementViolation:
    path: str
    line: int
    rule: str

    def as_message(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}"


def collect_unauthorized_authorize_and_execute_calls(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[InnerEnforcementViolation]:
    """Flag direct ``authorize_and_execute`` calls outside certified inner enforcement surface."""
    violations: list[InnerEnforcementViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "authorize_and_execute":
            violations.append(
                InnerEnforcementViolation(
                    path=rel_path,
                    line=node.lineno,
                    rule="unauthorized_authorize_and_execute_call",
                )
            )
    return violations


POLICY_BOUNDARY_FORBIDDEN_GUARD_IMPORT_PREFIXES: frozenset[str] = frozenset(
    {
        "intergrax.runtime.governance.canonical_inner_execution_guard",
    }
)

POLICY_BOUNDARY_ALLOWED_CONTRACT_IMPORT_PREFIXES: frozenset[str] = frozenset(
    {
        "intergrax.contracts.canonical_inner_governance",
    }
)


def collect_forbidden_concrete_inner_guard_imports(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[InnerEnforcementViolation]:
    """Policy boundary must not import concrete inner guard implementations."""
    violations: list[InnerEnforcementViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None:
            continue
        module = node.module
        if module in POLICY_BOUNDARY_FORBIDDEN_GUARD_IMPORT_PREFIXES:
            violations.append(
                InnerEnforcementViolation(
                    path=rel_path,
                    line=node.lineno,
                    rule="policy_boundary_imports_concrete_inner_guard",
                )
            )
            continue
        for alias in node.names:
            if alias.name == "DefaultCanonicalInnerExecutionGuard":
                violations.append(
                    InnerEnforcementViolation(
                        path=rel_path,
                        line=node.lineno,
                        rule="policy_boundary_imports_default_inner_guard_class",
                    )
                )
    return violations
