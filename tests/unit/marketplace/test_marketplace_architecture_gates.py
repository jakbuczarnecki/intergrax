# © Artur Czarnecki. All rights reserved.

"""Stage 11 marketplace architecture boundary regression gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MARKETPLACE_MODULE = "intergrax.marketplace"
_CONTRACTS_MODULE = "intergrax.contracts.marketplace"

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.tools.registry.runtime",
    "intergrax.skills.registry.runtime",
    "intergrax.agent_distribution",
    "intergrax.runtime",
    "applications",
)

_FORBIDDEN_RUNTIME_MUTATION_NAMES = frozenset(
    {
        "install",
        "activate",
        "enable",
        "register",
        "purchase",
        "checkout",
        "grant",
        "execute",
    },
)

_FORBIDDEN_INSTALL_MODULES = frozenset({"pip", "subprocess"})

_CATALOG_AUTHORITY_MODULES = (
    "intergrax.capability_catalog.ranking",
    "intergrax.capability_catalog.governance",
    "intergrax.capability_catalog.ranking_validation",
    "intergrax.capability_catalog.governance_validation",
)


def _package_root(module_name: str) -> Path:
    package = importlib.import_module(module_name)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _iter_package_py_files(module_name: str) -> list[Path]:
    return sorted(path for path in _package_root(module_name).rglob("*.py") if path.is_file())


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def _collect_call_names(tree: ast.AST) -> list[str]:
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    return calls


def test_marketplace_package_has_no_forbidden_runtime_imports() -> None:
    root = _package_root(_MARKETPLACE_MODULE)
    for path in _iter_package_py_files(_MARKETPLACE_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports forbidden dependency: {imported}",
                    )
            if imported in _FORBIDDEN_INSTALL_MODULES:
                raise AssertionError(
                    f"{path.relative_to(root)} imports forbidden install module: {imported}",
                )


def test_marketplace_package_has_no_runtime_mutation_api() -> None:
    root = _package_root(_MARKETPLACE_MODULE)
    for path in _iter_package_py_files(_MARKETPLACE_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in _FORBIDDEN_RUNTIME_MUTATION_NAMES:
                    raise AssertionError(
                        f"{path.relative_to(root)} exposes forbidden API: {node.name}",
                    )


def test_marketplace_package_does_not_call_agent_registry_register() -> None:
    root = _package_root(_MARKETPLACE_MODULE)
    for path in _iter_package_py_files(_MARKETPLACE_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call_name in _collect_call_names(tree):
            if call_name == "register":
                raise AssertionError(
                    f"{path.relative_to(root)} invokes forbidden register() call",
                )


def test_forbidden_flow_1_no_pip_or_tool_registry_mutation_imports() -> None:
    root = _package_root(_MARKETPLACE_MODULE)
    forbidden = (
        "intergrax.tools.registry.runtime",
        "intergrax.tools.registry.catalog",
        "pip",
        "subprocess",
    )
    for path in _iter_package_py_files(_MARKETPLACE_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for item in forbidden:
                if imported == item or imported.startswith(f"{item}."):
                    raise AssertionError(
                        f"Forbidden flow 1: {path.relative_to(root)} imports {imported}",
                    )


def test_forbidden_flow_3_no_agent_registry_register_path() -> None:
    root = _package_root(_MARKETPLACE_MODULE)
    for path in _iter_package_py_files(_MARKETPLACE_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            if imported == "intergrax.agent_distribution" or imported.startswith(
                "intergrax.agent_distribution.",
            ):
                raise AssertionError(
                    f"Forbidden flow 3: {path.relative_to(root)} imports {imported}",
                )


def test_catalog_authority_modules_do_not_import_marketplace_contracts() -> None:
    for module_name in _CATALOG_AUTHORITY_MODULES:
        module = importlib.import_module(module_name)
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            if imported == _CONTRACTS_MODULE or imported.startswith(f"{_CONTRACTS_MODULE}."):
                raise AssertionError(f"{module_name} imports marketplace contracts: {imported}")
            if imported == _MARKETPLACE_MODULE or imported.startswith(f"{_MARKETPLACE_MODULE}."):
                raise AssertionError(f"{module_name} imports marketplace package: {imported}")


def test_marketplace_service_does_not_import_ranking_with_commercial_metadata() -> None:
    service_module = importlib.import_module("intergrax.marketplace.service")
    path = Path(service_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = _collect_imports(tree)
    assert "intergrax.capability_catalog.ranking" not in imported
    assert "intergrax.contracts.marketplace.commercial" not in imported
