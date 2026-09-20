# © Artur Czarnecki. All rights reserved.

"""Conservative higher-layer Nexus import and private-access detection (HARNESS-01-R2)."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NexusImportedSymbols:
    """Names bound to Nexus modules or types via import statements."""

    type_names: frozenset[str]
    module_aliases: frozenset[str]


def nexus_imported_symbols(tree: ast.Module) -> NexusImportedSymbols:
    type_names: set[str] = set()
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.runtime.nexus"
        ):
            for alias in node.names:
                type_names.add(alias.asname or alias.name.split(".")[-1])
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod == "intergrax.runtime.nexus" or mod.startswith("intergrax.runtime.nexus."):
                    module_aliases.add(alias.asname or mod.split(".")[-1])
    return NexusImportedSymbols(
        type_names=frozenset(type_names),
        module_aliases=frozenset(module_aliases),
    )


def _nexus_module_string(value: str) -> bool:
    return value == "intergrax.runtime.nexus" or value.startswith("intergrax.runtime.nexus.")


def _collect_nexus_module_string_bindings(tree: ast.Module) -> set[str]:
    bindings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                if _nexus_module_string(node.value.value):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            bindings.add(target.id)
        if (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and _nexus_module_string(node.value.value)
            and isinstance(node.target, ast.Name)
        ):
            bindings.add(node.target.id)
    return bindings


def _expr_is_nexus_module_reference(
    node: ast.expr,
    *,
    nexus_string_bindings: set[str],
) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _nexus_module_string(node.value)
    if isinstance(node, ast.Name) and node.id in nexus_string_bindings:
        return True
    return False


def file_imports_nexus_module(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    symbols = nexus_imported_symbols(tree)
    return bool(symbols.type_names or symbols.module_aliases)


def file_has_dynamic_nexus_import(source: str) -> bool:
    """Detect ``importlib.import_module`` / ``__import__`` targeting Nexus modules."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    nexus_string_bindings = _collect_nexus_module_string_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_import_module = (
            isinstance(func, ast.Attribute)
            and func.attr == "import_module"
            and isinstance(func.value, ast.Name)
            and func.value.id == "importlib"
        )
        is_dunder_import = isinstance(func, ast.Name) and func.id == "__import__"
        if not is_import_module and not is_dunder_import:
            continue
        for arg in node.args[:1]:
            if _expr_is_nexus_module_reference(arg, nexus_string_bindings=nexus_string_bindings):
                return True
    return False


def _is_nexus_type_reference(
    node: ast.expr,
    *,
    type_names: frozenset[str],
    module_aliases: frozenset[str],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in type_names
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in module_aliases:
            return True
        if isinstance(node.value, ast.Name) and node.value.id in type_names:
            return True
    return False


def _is_nexus_origin_expression(
    node: ast.expr,
    *,
    type_names: frozenset[str],
    module_aliases: frozenset[str],
) -> bool:
    if isinstance(node, ast.Call):
        return _is_nexus_type_reference(
            node.func,
            type_names=type_names,
            module_aliases=module_aliases,
        )
    if isinstance(node, ast.Name) and node.id in type_names:
        return True
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in module_aliases:
            return True
    return _is_nexus_type_reference(
        node,
        type_names=type_names,
        module_aliases=module_aliases,
    )


def _register_assignment_target(
    target: ast.expr,
    *,
    local_bindings: set[str],
    instance_attrs: set[str],
) -> None:
    if isinstance(target, ast.Name):
        local_bindings.add(target.id)
    elif (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        instance_attrs.add(target.attr)


def _collect_nexus_value_bindings(
    tree: ast.Module,
    symbols: NexusImportedSymbols,
) -> tuple[set[str], set[str]]:
    local_bindings: set[str] = set(symbols.type_names)
    instance_attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if _is_nexus_origin_expression(
                node.value,
                type_names=symbols.type_names,
                module_aliases=symbols.module_aliases,
            ):
                for target in node.targets:
                    _register_assignment_target(
                        target,
                        local_bindings=local_bindings,
                        instance_attrs=instance_attrs,
                    )
        if (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and _is_nexus_origin_expression(
                node.value,
                type_names=symbols.type_names,
                module_aliases=symbols.module_aliases,
            )
        ):
            _register_assignment_target(
                node.target,
                local_bindings=local_bindings,
                instance_attrs=instance_attrs,
            )
    return local_bindings, instance_attrs


def _receiver_is_nexus_bound(
    node: ast.expr,
    *,
    local_bindings: set[str],
    instance_attrs: set[str],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in local_bindings
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in instance_attrs
    ):
        return True
    if isinstance(node, ast.Attribute) and _receiver_is_nexus_bound(
        node.value,
        local_bindings=local_bindings,
        instance_attrs=instance_attrs,
    ):
        return True
    return False


def collect_nexus_private_member_access_violations(
    source: str,
    *,
    filename: str = "<memory>",
) -> list[str]:
    """
    Detect private member access on Nexus-imported symbols or derived bindings.

    Does not flag ``self._private`` unless ``self._private`` holds a Nexus binding.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return []
    symbols = nexus_imported_symbols(tree)
    if not symbols.type_names and not symbols.module_aliases:
        return []
    local_bindings, instance_attrs = _collect_nexus_value_bindings(tree, symbols)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not node.attr.startswith("_") or node.attr.startswith("__"):
            continue
        if _receiver_is_nexus_bound(
            node.value,
            local_bindings=local_bindings,
            instance_attrs=instance_attrs,
        ):
            violations.append(f"{filename}:{node.lineno}:{node.attr}")
    return violations
