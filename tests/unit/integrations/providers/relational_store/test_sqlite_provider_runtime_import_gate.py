# © Artur Czarnecki. All rights reserved.

"""R6-AUDIT-MAJOR-01 — SQLite provider must not import runtime/Nexus concrete stores."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[5]
SQLITE_PROVIDER = (
    REPO / "intergrax" / "integrations" / "providers" / "relational_store" / "sqlite"
)
SQLITE_INIT = SQLITE_PROVIDER / "__init__.py"
SQLITE_RUNTIME_COMPOSITION = (
    REPO / "intergrax" / "runtime" / "persistence" / "sqlite_composition.py"
)

_ALLOWED_RUNTIME_PREFIXES = ("intergrax.runtime.integrations.",)

_FORBIDDEN_PREFIXES = (
    "intergrax.runtime.nexus.",
    "intergrax.runtime.events.",
    "intergrax.runtime.human.",
    "intergrax.runtime.long_running.",
    "intergrax.runtime.task_memory.",
    "intergrax.runtime.organization.",
    "intergrax.runtime.tools.",
    "intergrax.runtime.notifications.",
    "intergrax.experiments.",
    "intergrax.memory.stores.",
)


def _is_forbidden_runtime_import(module: str) -> bool:
    if any(module.startswith(p) for p in _ALLOWED_RUNTIME_PREFIXES):
        return False
    if module.startswith("intergrax.runtime."):
        return True
    return any(module.startswith(p) for p in _FORBIDDEN_PREFIXES)


def _forbidden_imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if _is_forbidden_runtime_import(name):
                    violations.append(f"{path.name}:{node.lineno} import {name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            mod = node.module
            if _is_forbidden_runtime_import(mod):
                violations.append(f"{path.name}:{node.lineno} from {mod}")
    return violations


def test_sqlite_provider_package_has_no_runtime_concrete_imports() -> None:
    violations: list[str] = []
    for path in sorted(SQLITE_PROVIDER.glob("*.py")):
        violations.extend(_forbidden_imports_in_file(path))
    assert not violations, "SQLite provider runtime import violations:\n" + "\n".join(
        violations
    )


def _module_has_getattr(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__getattr__":
                return True
    return False


def _imports_lazy_export(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "lazy_export" in alias.name:
                    violations.append(f"{path.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and "lazy_export" in node.module:
                violations.append(f"{path.name}:{node.lineno} from {node.module}")
    return violations


def _private_sqlite_provider_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prefix = "intergrax.integrations.providers.relational_store.sqlite"
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.startswith(prefix):
            continue
        for alias in node.names:
            name = alias.name
            if name.startswith("_"):
                violations.append(
                    f"{path.name}:{node.lineno} from {node.module} import {name}"
                )
    return violations


def _relational_store_field_uses_relational_store_contract(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "relational_store":
            continue
        ann = node.annotation
        if isinstance(ann, ast.Name) and ann.id == "RelationalStore":
            return True
        if isinstance(ann, ast.Attribute) and ann.attr == "RelationalStore":
            return True
    return False


def test_sqlite_provider_init_has_no_module_getattr() -> None:
    tree = ast.parse(SQLITE_INIT.read_text(encoding="utf-8"), filename=str(SQLITE_INIT))
    assert not _module_has_getattr(tree), (
        "SQLite provider __init__.py must not define module-level __getattr__"
    )


def test_sqlite_provider_init_has_no_lazy_export() -> None:
    violations = _imports_lazy_export(SQLITE_INIT)
    assert not violations, (
        "SQLite provider __init__.py lazy_export violations:\n" + "\n".join(violations)
    )


def test_runtime_sqlite_composition_has_no_private_provider_imports() -> None:
    violations = _private_sqlite_provider_imports(SQLITE_RUNTIME_COMPOSITION)
    assert not violations, (
        "runtime sqlite_composition private provider import violations:\n"
        + "\n".join(violations)
    )


def test_runtime_bundle_relational_store_typed_as_relational_store_contract() -> None:
    assert _relational_store_field_uses_relational_store_contract(
        SQLITE_RUNTIME_COMPOSITION
    ), (
        "SQLiteRuntimePersistenceBundle.relational_store must be annotated as "
        "RelationalStore"
    )
