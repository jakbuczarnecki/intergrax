# © Artur Czarnecki. All rights reserved.

"""ME-RB2 — marketplace contracts must not import implementation packages."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE_MODULE = "intergrax.contracts.marketplace"

_FORBIDDEN_IMPLEMENTATION_PREFIXES = (
    "intergrax.marketplace",
    "intergrax.capability_catalog",
    "intergrax.nexus",
    "intergrax.runtime.nexus",
    "intergrax.tools.providers.billing",
    "intergrax.skills.providers.billing",
)


def _package_root() -> Path:
    package = importlib.import_module(_PACKAGE_MODULE)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _iter_package_py_files() -> list[Path]:
    return sorted(path for path in _package_root().rglob("*.py") if path.is_file())


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_marketplace_contracts_package_has_no_implementation_imports() -> None:
    root = _package_root()
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_IMPLEMENTATION_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports forbidden dependency: {imported}",
                    )


def test_marketplace_contracts_do_not_expose_default_implementation_types() -> None:
    init_path = _package_root() / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    forbidden_symbols = (
        "DefaultMarketplaceListingProjection",
        "MarketplaceCapabilityCatalogSource",
        "InMemoryMarketplaceMetadataSource",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_symbols:
            raise AssertionError(f"contracts __init__ must not export implementation: {node.id}")
