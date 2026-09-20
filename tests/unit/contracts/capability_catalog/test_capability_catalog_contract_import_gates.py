# © Artur Czarnecki. All rights reserved.

"""ME-6 / ME-RB2 — capability catalog contracts must not import implementation packages."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE_MODULE = "intergrax.contracts.capability_catalog"

_FORBIDDEN_IMPLEMENTATION_PREFIXES = (
    "intergrax.capability_catalog",
    "intergrax.marketplace",
    "intergrax.agent_distribution",
    "intergrax.autonomous_work",
    "intergrax.tools",
    "intergrax.skills",
    "intergrax.nexus",
    "intergrax.runtime",
    "intergrax.codecraft",
    "intergrax.runtime.codecraft",
    "intergrax.runtime.sandbox",
    "intergrax.runtime.execution",
    "applications",
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


def test_capability_catalog_contracts_package_has_no_implementation_imports() -> None:
    root = _package_root()
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_IMPLEMENTATION_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports forbidden dependency: {imported}",
                    )
