# © Artur Czarnecki. All rights reserved.

"""ME-RB1 — common marketplace engine boundary gates (vertical vs engine)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MARKETPLACE_MODULE = "intergrax.marketplace"
_CATALOG_ADAPTERS = "intergrax.capability_catalog.adapters"

_DOMAIN_PACKAGES_MUST_NOT_IMPORT_MARKETPLACE = (
    "intergrax.agent_distribution",
    "intergrax.tools",
    "intergrax.skills",
    _CATALOG_ADAPTERS,
)


def _package_root(module_name: str) -> Path:
    package = importlib.import_module(module_name)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _iter_py_files(module_name: str) -> list[Path]:
    return sorted(path for path in _package_root(module_name).rglob("*.py") if path.is_file())


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


@pytest.mark.parametrize("module_name", _DOMAIN_PACKAGES_MUST_NOT_IMPORT_MARKETPLACE)
def test_domain_vertical_packages_do_not_import_marketplace_engine(module_name: str) -> None:
    """Verticals and domain trees must not embed the marketplace product engine."""
    root = _package_root(module_name)
    for path in _iter_py_files(module_name):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            if imported == _MARKETPLACE_MODULE or imported.startswith(f"{_MARKETPLACE_MODULE}."):
                raise AssertionError(
                    f"{path.relative_to(root)} imports marketplace engine: {imported}",
                )


def test_integrations_marketplace_catalog_is_not_capability_engine_import() -> None:
    """Integration marketplace catalog stays in integrations bounded context."""
    module = importlib.import_module("intergrax.integrations.registry.marketplace_catalog")
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for imported in _collect_imports(tree):
        forbidden = (
            _MARKETPLACE_MODULE,
            "intergrax.capability_catalog",
            "intergrax.capability_metering",
        )
        for prefix in forbidden:
            if imported == prefix or imported.startswith(f"{prefix}."):
                raise AssertionError(
                    f"IntegrationMarketplaceCatalog must not depend on capability engine: {imported}",
                )
