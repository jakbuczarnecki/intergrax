# © Artur Czarnecki. All rights reserved.

"""Shared AST primitives for canonical AgentRegistry lifecycle bypass detection."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum

CANONICAL_AGENT_REGISTRY_MODULE = "intergrax.runtime.registry.agent_registry"
CANONICAL_AGENT_REGISTRY_SYMBOL = "AgentRegistry"


class LifecycleAstViolationKind(StrEnum):
    AGENT_REGISTRY_CONSTRUCTION = "agent_registry_construction"
    AGENT_REGISTRY_FROM_AGENTS = "agent_registry_from_agents"
    LOCAL_REGISTER = "local_register"


@dataclass(frozen=True, slots=True)
class LifecycleAstViolation:
    line: int
    symbol: str
    kind: LifecycleAstViolationKind
    message: str


@dataclass(frozen=True, slots=True)
class AgentRegistryImportBindings:
    """Explicit import bindings for canonical AgentRegistry lifecycle AST checks."""

    class_aliases: frozenset[str]
    module_aliases: frozenset[str]
    qualified_module_roots: frozenset[tuple[str, str]]


def collect_agent_registry_import_bindings(tree: ast.AST) -> AgentRegistryImportBindings:
    class_aliases: set[str] = set()
    module_aliases: set[str] = set()
    qualified_module_roots: set[tuple[str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == CANONICAL_AGENT_REGISTRY_MODULE:
            for alias in node.names:
                if alias.name == CANONICAL_AGENT_REGISTRY_SYMBOL:
                    class_aliases.add(alias.asname or alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == CANONICAL_AGENT_REGISTRY_MODULE:
                    if alias.asname is not None:
                        module_aliases.add(alias.asname)
                    else:
                        qualified_module_roots.add(
                            (alias.name.split(".")[0], alias.name)
                        )
                elif alias.name.endswith(f".{CANONICAL_AGENT_REGISTRY_SYMBOL}"):
                    class_aliases.add(alias.asname or CANONICAL_AGENT_REGISTRY_SYMBOL)

    return AgentRegistryImportBindings(
        class_aliases=frozenset(class_aliases),
        module_aliases=frozenset(module_aliases),
        qualified_module_roots=frozenset(qualified_module_roots),
    )


def _attribute_chain_parts(expr: ast.expr) -> tuple[str, tuple[str, ...]] | None:
    attrs: list[str] = []
    current = expr
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        return current.id, tuple(reversed(attrs))
    return None


def _resolves_to_agent_registry_module(
    expr: ast.expr,
    bindings: AgentRegistryImportBindings,
) -> bool:
    parts = _attribute_chain_parts(expr)
    if parts is None:
        return False
    root, attrs = parts
    if root in bindings.module_aliases and not attrs:
        return True
    for imported_root, module_path in bindings.qualified_module_roots:
        if root != imported_root or module_path != CANONICAL_AGENT_REGISTRY_MODULE:
            continue
        suffix = module_path[len(imported_root) + 1 :]
        if suffix and attrs == tuple(suffix.split(".")):
            return True
    return False


def _is_agent_registry_call(func: ast.expr, bindings: AgentRegistryImportBindings) -> bool:
    if isinstance(func, ast.Name) and func.id in bindings.class_aliases:
        return True
    if not isinstance(func, ast.Attribute):
        return False

    if func.attr == "from_agents":
        if isinstance(func.value, ast.Name) and func.value.id in bindings.class_aliases:
            return True
        return (
            isinstance(func.value, ast.Attribute)
            and func.value.attr == CANONICAL_AGENT_REGISTRY_SYMBOL
            and _resolves_to_agent_registry_module(func.value.value, bindings)
        )

    if func.attr == CANONICAL_AGENT_REGISTRY_SYMBOL:
        return _resolves_to_agent_registry_module(func.value, bindings)

    return False


def _agent_registry_call_symbol(
    func: ast.expr,
    bindings: AgentRegistryImportBindings,
) -> str:
    if isinstance(func, ast.Attribute) and func.attr == "from_agents":
        return "AgentRegistry.from_agents"
    return CANONICAL_AGENT_REGISTRY_SYMBOL


def _default_message(kind: LifecycleAstViolationKind, symbol: str) -> str:
    if kind is LifecycleAstViolationKind.LOCAL_REGISTER:
        return (
            "must not mutate a locally constructed AgentRegistry; "
            "use canonical agent lifecycle"
        )
    return (
        "must not construct or mutate AgentRegistry; "
        "use canonical agent lifecycle"
    )


def collect_agent_registry_lifecycle_violations(
    tree: ast.AST,
    *,
    bindings: AgentRegistryImportBindings | None = None,
) -> tuple[LifecycleAstViolation, ...]:
    """Return canonical AgentRegistry lifecycle bypass violations in deterministic order."""
    resolved_bindings = bindings or collect_agent_registry_import_bindings(tree)
    violations: list[LifecycleAstViolation] = []
    local_registry_names: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Call) and _is_agent_registry_call(
            node.value.func,
            resolved_bindings,
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local_registry_names.add(target.id)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if _is_agent_registry_call(node.func, resolved_bindings):
            symbol = _agent_registry_call_symbol(node.func, resolved_bindings)
            kind = (
                LifecycleAstViolationKind.AGENT_REGISTRY_FROM_AGENTS
                if symbol == "AgentRegistry.from_agents"
                else LifecycleAstViolationKind.AGENT_REGISTRY_CONSTRUCTION
            )
            violations.append(
                LifecycleAstViolation(
                    line=node.lineno,
                    symbol=symbol,
                    kind=kind,
                    message=_default_message(kind, symbol),
                )
            )
            continue

        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in local_registry_names
        ):
            violations.append(
                LifecycleAstViolation(
                    line=node.lineno,
                    symbol="register",
                    kind=LifecycleAstViolationKind.LOCAL_REGISTER,
                    message=_default_message(
                        LifecycleAstViolationKind.LOCAL_REGISTER,
                        "register",
                    ),
                )
            )

    return tuple(
        sorted(violations, key=lambda violation: (violation.line, violation.kind, violation.symbol))
    )
