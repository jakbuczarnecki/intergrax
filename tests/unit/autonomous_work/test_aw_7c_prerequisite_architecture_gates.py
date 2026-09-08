# © Artur Czarnecki. All rights reserved.

"""AW-7C prerequisite — Nexus public-boundary architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_NEXUS_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.nexus",
    "intergrax.runtime.nexus",
)

_NEXUS_FORBIDDEN_SYMBOLS = (
    "NexusTask",
    "NexusRun",
    "NexusPlanner",
    "NexusExecution",
    "NexusLoop",
    "NexusState",
)


def _python_sources_under(package_name: str) -> list[Path]:
    package = importlib.import_module(package_name)
    assert package.__file__ is not None
    root = Path(package.__file__).parent
    return sorted(root.rglob("*.py"))


def _collect_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


@pytest.mark.parametrize(
    "package_name",
    (
        "intergrax.autonomous_work",
        "intergrax.contracts.autonomous_work",
    ),
)
def test_aw_7c_modules_forbidden_nexus_imports(package_name: str) -> None:
    for path in _python_sources_under(package_name):
        source = path.read_text(encoding="utf-8")
        imported = _collect_imports(source)
        joined = "\n".join(imported).lower()
        for prefix in _NEXUS_FORBIDDEN_IMPORT_PREFIXES:
            assert prefix.lower() not in joined, f"{path} imports forbidden Nexus surface {prefix}"


@pytest.mark.parametrize(
    "package_name",
    (
        "intergrax.autonomous_work",
        "intergrax.contracts.autonomous_work",
    ),
)
def test_aw_7c_modules_forbidden_nexus_symbols(package_name: str) -> None:
    for path in _python_sources_under(package_name):
        source = path.read_text(encoding="utf-8")
        for symbol in _NEXUS_FORBIDDEN_SYMBOLS:
            assert symbol not in source, f"{path} references forbidden Nexus symbol {symbol}"
