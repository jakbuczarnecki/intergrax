# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 program-level architecture boundary gates (Stages 1–14)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PROGRAM_PACKAGES = (
    "intergrax.capability_catalog",
    "intergrax.contracts.capability_catalog",
    "intergrax.capability_metering",
    "intergrax.contracts.capability_metering",
    "intergrax.marketplace",
    "intergrax.contracts.marketplace",
)

_AW_STAGE14_MODULE = "intergrax.autonomous_work.work_stage_capability_loop"

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.runtime",
    "intergrax.applications",
    "applications",
)

_FORBIDDEN_SYMBOLS = frozenset(
    {
        "UniversalCapabilityEngine",
        "UniversalCapabilityRuntime",
        "UniversalCapabilityExecutor",
        "UniversalRegistry",
        "UniversalWorkerOrchestrator",
        "CapabilityDiscoveryPort",
        "CapabilityLifecycle",
    },
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


def test_program_packages_do_not_import_runtime_or_applications() -> None:
    for module_name in _PROGRAM_PACKAGES:
        root = _package_root(module_name)
        for path in _iter_py_files(module_name):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for imported in _collect_imports(tree):
                for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                    if imported == prefix or imported.startswith(f"{prefix}."):
                        raise AssertionError(
                            f"{path.relative_to(root)} imports forbidden dependency: {imported}",
                        )


def test_program_packages_do_not_define_forbidden_universal_symbols() -> None:
    for module_name in _PROGRAM_PACKAGES:
        root = _package_root(module_name)
        for path in _iter_py_files(module_name):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name in _FORBIDDEN_SYMBOLS:
                    raise AssertionError(
                        f"{path.relative_to(root)} defines forbidden symbol {node.name}",
                    )


def test_aw_stage14_core_does_not_import_runtime_or_applications() -> None:
    module = importlib.import_module(_AW_STAGE14_MODULE)
    assert module.__file__ is not None
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for imported in _collect_imports(tree):
        for prefix in _FORBIDDEN_IMPORT_PREFIXES:
            if imported == prefix or imported.startswith(f"{prefix}."):
                raise AssertionError(
                    f"{path.name} imports forbidden dependency: {imported}",
                )


def test_aw_stage14_core_has_no_forbidden_universal_symbol_references() -> None:
    module = importlib.import_module(_AW_STAGE14_MODULE)
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    for symbol in _FORBIDDEN_SYMBOLS:
        assert symbol not in source, f"{module.__name__} must not reference {symbol}"
