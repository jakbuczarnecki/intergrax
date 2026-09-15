# © Artur Czarnecki. All rights reserved.

"""AST guards for memory contract boundary purity."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemoryContractBoundaryViolation:
    path: str
    line: int
    rule: str

    def as_message(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}"


_CONTRACTS_PACKAGE_PREFIX = "intergrax.memory.contracts"
_ALLOWED_MEMORY_IMPORT_PREFIXES = (f"{_CONTRACTS_PACKAGE_PREFIX}.",)


def _module_name(node: ast.ImportFrom) -> str | None:
    if node.level and node.level > 0:
        return None
    return node.module


def _is_type_checking_if(node: ast.If) -> bool:
    test = node.test
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _module_level_import_from_nodes(tree: ast.AST) -> list[ast.ImportFrom]:
    if not isinstance(tree, ast.Module):
        return []
    imports: list[ast.ImportFrom] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.If) and _is_type_checking_if(stmt):
            continue
        for node in ast.walk(stmt):
            if isinstance(node, ast.ImportFrom):
                imports.append(node)
    return imports


def collect_forbidden_memory_implementation_imports_in_contracts(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[MemoryContractBoundaryViolation]:
    """Contracts must not import non-contract memory implementation modules."""
    violations: list[MemoryContractBoundaryViolation] = []
    for node in _module_level_import_from_nodes(tree):
        module = _module_name(node)
        if module is None:
            continue
        if not module.startswith("intergrax.memory."):
            continue
        if module.startswith(_ALLOWED_MEMORY_IMPORT_PREFIXES):
            continue
        violations.append(
            MemoryContractBoundaryViolation(
                path=rel_path,
                line=node.lineno,
                rule=f"contracts_imports_memory_implementation:{module}",
            )
        )
    return violations


def collect_memory_lifecycle_user_profile_memory_imports(
    tree: ast.AST,
    *,
    rel_path: str,
) -> list[MemoryContractBoundaryViolation]:
    violations: list[MemoryContractBoundaryViolation] = []
    for node in _module_level_import_from_nodes(tree):
        module = _module_name(node)
        if module == "intergrax.memory.user_profile_memory":
            violations.append(
                MemoryContractBoundaryViolation(
                    path=rel_path,
                    line=node.lineno,
                    rule="memory_lifecycle_imports_user_profile_memory",
                )
            )
    return violations


def scan_memory_contracts_tree(
    contracts_root: Path,
) -> list[MemoryContractBoundaryViolation]:
    violations: list[MemoryContractBoundaryViolation] = []
    for path in sorted(contracts_root.glob("*.py")):
        rel_path = path.as_posix()
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=rel_path)
        violations.extend(
            collect_forbidden_memory_implementation_imports_in_contracts(
                tree,
                rel_path=rel_path,
            )
        )
        if path.name == "memory_lifecycle.py":
            violations.extend(
                collect_memory_lifecycle_user_profile_memory_imports(
                    tree,
                    rel_path=rel_path,
                )
            )
    return violations
