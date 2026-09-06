# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 2–5 architecture boundary regression gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_CORE_MODULE = "intergrax.capability_catalog"
_ADAPTERS_MODULE = "intergrax.capability_catalog.adapters"

_FORBIDDEN_CORE_IMPORT_PREFIXES = (
    "applications",
    "intergrax.applications",
    "intergrax.agent_distribution",
    "intergrax.autonomous_work",
    "intergrax.runtime",
    "intergrax.skills",
    "intergrax.tools",
)

_FORBIDDEN_ADAPTER_RUNTIME_IMPORTS = (
    "intergrax.skills.registry.runtime",
    "intergrax.tools.registry.runtime",
)

_FORBIDDEN_REGISTRY_CLASS_NAMES = frozenset(
    {
        "AgentRegistry",
        "SkillRegistry",
        "ToolRegistry",
        "UniversalCapabilityEngine",
        "UniversalRegistry",
        "CapabilityRegistry",
    },
)

_FORBIDDEN_RUNTIME_MUTATION_FUNCTION_NAMES = frozenset(
    {
        "install",
        "enable",
        "activate",
        "materialize",
        "register",
        "execute",
        "select",
        "authorize",
        "grant_permission",
    },
)

_RANKING_MODULE_ALLOWLIST = frozenset({"ranking.py"})


def _package_root(module_name: str) -> Path:
    package = importlib.import_module(module_name)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _iter_package_py_files(module_name: str) -> list[Path]:
    return sorted(path for path in _package_root(module_name).rglob("*.py") if path.is_file())


def _iter_core_py_files() -> list[Path]:
    root = _package_root(_CORE_MODULE)
    adapters_dir = root / "adapters"
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and adapters_dir not in path.parents
    )


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def _collect_private_cross_module_imports(path: Path, tree: ast.AST) -> list[str]:
    current_module = (
        _CORE_MODULE
        if path.stem == "__init__"
        else f"{_CORE_MODULE}.{path.stem}"
    )
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not (
            node.module == _CORE_MODULE
            or node.module.startswith(f"{_CORE_MODULE}.")
        ):
            continue
        if node.module == current_module:
            continue
        for alias in node.names:
            if alias.name.startswith("_"):
                violations.append(
                    f"{path.name}:{node.lineno} imports private {alias.name} "
                    f"from {node.module}",
                )
    return violations


def _collect_forbidden_registry_class_defs(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in _FORBIDDEN_REGISTRY_CLASS_NAMES:
            violations.append(f"class {node.name} at line {node.lineno}")
    return violations


def _collect_forbidden_runtime_mutation_defs(
    tree: ast.AST,
    *,
    path_name: str,
) -> list[str]:
    violations: list[str] = []
    allow_rank = path_name in _RANKING_MODULE_ALLOWLIST
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _FORBIDDEN_RUNTIME_MUTATION_FUNCTION_NAMES:
                violations.append(f"function {node.name} at line {node.lineno}")
            elif node.name == "rank" and not allow_rank:
                violations.append(f"function {node.name} at line {node.lineno}")
    return violations


def test_capability_catalog_core_has_no_forbidden_imports() -> None:
    root = _package_root(_CORE_MODULE)
    for path in _iter_core_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_CORE_IMPORT_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports forbidden dependency: {imported}",
                    )


def test_capability_catalog_core_has_no_registry_unification_classes() -> None:
    root = _package_root(_CORE_MODULE)
    for path in _iter_core_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations = _collect_forbidden_registry_class_defs(tree)
        assert not violations, (
            f"{path.relative_to(root)} defines forbidden registry symbols: "
            + ", ".join(violations)
        )


def test_capability_catalog_core_has_no_runtime_mutation_api() -> None:
    root = _package_root(_CORE_MODULE)
    for path in _iter_core_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations = _collect_forbidden_runtime_mutation_defs(
            tree,
            path_name=path.name,
        )
        assert not violations, (
            f"{path.relative_to(root)} exposes forbidden API: "
            + ", ".join(violations)
        )


def test_capability_catalog_adapters_do_not_import_runtime_registries() -> None:
    root = _package_root(_ADAPTERS_MODULE)
    for path in _iter_package_py_files(_ADAPTERS_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for forbidden in _FORBIDDEN_ADAPTER_RUNTIME_IMPORTS:
                if imported == forbidden or imported.startswith(f"{forbidden}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports runtime registry: {imported}",
                    )


def test_capability_catalog_core_does_not_import_adapters() -> None:
    root = _package_root(_CORE_MODULE)
    for path in _iter_core_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            if imported == _ADAPTERS_MODULE or imported.startswith(f"{_ADAPTERS_MODULE}."):
                raise AssertionError(
                    f"{path.relative_to(root)} imports domain adapters: {imported}",
                )


def test_capability_catalog_core_has_no_private_cross_module_imports() -> None:
    root = _package_root(_CORE_MODULE)
    for path in _iter_core_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations = _collect_private_cross_module_imports(path, tree)
        assert not violations, (
            f"{path.relative_to(root)} has forbidden private cross-module imports: "
            + ", ".join(violations)
        )
