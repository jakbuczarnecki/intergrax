# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 1 architecture boundary regression gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE_MODULE = "intergrax.contracts.capability_catalog"

_FORBIDDEN_IMPORT_PREFIXES = (
    "applications",
    "intergrax.applications",
    "intergrax.agent_distribution",
    "intergrax.autonomous_work",
    "intergrax.runtime",
    "intergrax.skills",
    "intergrax.tools",
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
        "mutate",
        "select",
        "rank",
        "authorize",
        "grant_permission",
    },
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


def _collect_forbidden_registry_class_defs(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in _FORBIDDEN_REGISTRY_CLASS_NAMES:
            violations.append(f"class {node.name} at line {node.lineno}")
    return violations


def _collect_forbidden_runtime_mutation_defs(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _FORBIDDEN_RUNTIME_MUTATION_FUNCTION_NAMES:
                violations.append(f"function {node.name} at line {node.lineno}")
    return violations


def test_ast_gate_ignores_forbidden_tokens_in_comments_and_docstrings() -> None:
    source = '''
"""AgentRegistry is documented but not defined."""

# def install(): pass

x = "UniversalRegistry"
'''
    tree = ast.parse(source)
    assert _collect_forbidden_registry_class_defs(tree) == []
    assert _collect_forbidden_runtime_mutation_defs(tree) == []


def test_capability_catalog_package_has_no_forbidden_imports() -> None:
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.relative_to(_package_root())} imports forbidden dependency: {imported}",
                    )


def test_capability_catalog_package_has_no_registry_unification_classes() -> None:
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations = _collect_forbidden_registry_class_defs(tree)
        assert not violations, (
            f"{path.relative_to(_package_root())} defines forbidden registry symbols: "
            + ", ".join(violations)
        )


def test_capability_catalog_package_has_no_runtime_mutation_api() -> None:
    for path in _iter_package_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        violations = _collect_forbidden_runtime_mutation_defs(tree)
        assert not violations, (
            f"{path.relative_to(_package_root())} exposes forbidden API: "
            + ", ".join(violations)
        )


def test_capability_catalog_import_smoke_subprocess() -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[4]
    statement = """
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityStageVocabulary,
)
assert CapabilityKind.AGENT.value == "agent"
assert CapabilityStageVocabulary.DISCOVERED.value == "discovered"
print("capability catalog import smoke OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
