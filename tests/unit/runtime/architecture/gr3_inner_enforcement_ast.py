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

DEFAULT_INNER_GUARD_FORBIDDEN_IMPORT_PREFIXES: frozenset[str] = frozenset(
    {
        "intergrax.runtime.task.active_task_registry",
    }
)

DEFAULT_INNER_GUARD_ALLOWED_CONTRACT_IMPORT_PREFIXES: frozenset[str] = frozenset(
    {
        "intergrax.contracts.active_execution_task_scope",
        "intergrax.contracts.canonical_inner_governance",
        "intergrax.contracts.meaningful_side_effect",
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


def collect_forbidden_concrete_task_scope_imports_in_default_guard(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[InnerEnforcementViolation]:
    """Default inner guard must depend only on ActiveExecutionTaskScopePort (contracts)."""
    violations: list[InnerEnforcementViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None:
            continue
        module = node.module
        if module in DEFAULT_INNER_GUARD_FORBIDDEN_IMPORT_PREFIXES:
            violations.append(
                InnerEnforcementViolation(
                    path=rel_path,
                    line=node.lineno,
                    rule="default_inner_guard_imports_concrete_task_scope_resolver",
                )
            )
            continue
        for alias in node.names:
            if alias.name == "ActiveTaskRegistryTaskScopeResolver":
                violations.append(
                    InnerEnforcementViolation(
                        path=rel_path,
                        line=node.lineno,
                        rule="default_inner_guard_imports_active_task_registry_resolver",
                    )
                )
    return violations


def _self_method_calls_in_function(func_def: ast.FunctionDef) -> list[str]:
    calls: list[str] = []
    for stmt in func_def.body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                calls.append(func.attr)
    return calls


def prepare_invocation_inner_guard_before_authorization_indices(
    tree: ast.AST,
) -> tuple[int | None, int | None]:
    """Return statement-order indices of inner guard vs current-attempt auth in ``_prepare_invocation``."""
    guard_idx: int | None = None
    auth_idx: int | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_prepare_invocation":
            continue
        for index, name in enumerate(_self_method_calls_in_function(node)):
            if name == "_require_canonical_inner_execution_guard":
                guard_idx = index
            elif name == "_require_current_attempt_authorization":
                auth_idx = index
        break
    return guard_idx, auth_idx


def prepare_invocation_mse_before_idempotency_indices(
    tree: ast.AST,
) -> tuple[int | None, int | None]:
    """Return indices of MSE authorization vs idempotency claim in ``invoke``."""
    mse_idx: int | None = None
    idempotency_idx: int | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "invoke":
            continue
        for index, name in enumerate(_self_method_calls_in_function(node)):
            if name == "_prepare_invocation":
                mse_idx = index
            elif name == "before_external_effect":
                idempotency_idx = index
        break
    return mse_idx, idempotency_idx
