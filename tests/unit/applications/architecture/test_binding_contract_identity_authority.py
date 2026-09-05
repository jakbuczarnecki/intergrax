# © Artur Czarnecki. All rights reserved.

"""Architecture gate: single canonical AgentBinding contract identity authority."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PRODUCTION_ROOTS = (REPO_ROOT / "intergrax" / "applications",)
CATALOG_MODULE = REPO_ROOT / "intergrax" / "applications" / "_shared" / "capability_graph_catalog.py"
MANIFEST_MODULE = REPO_ROOT / "intergrax" / "applications" / "contracts" / "manifest.py"
FORBIDDEN_SYMBOL = "resolve_binding_agent_contract_id"
REFLECTION_HELPERS = frozenset({"getattr", "hasattr", "setattr"})
REFLECTION_MODULES = frozenset({"inspect"})

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _iter_python_files(root: Path) -> list[Path]:
    skip_parts = frozenset({"__pycache__", "build"})
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and skip_parts.isdisjoint(path.parts)
    )


def _import_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    rel = path.relative_to(REPO_ROOT)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name == FORBIDDEN_SYMBOL:
                module = node.module or ""
                violations.append(f"{rel}: from {module} import {FORBIDDEN_SYMBOL}")
    return violations


def test_production_modules_do_not_import_stale_binding_contract_resolver() -> None:
    violations: list[str] = []
    for root in PRODUCTION_ROOTS:
        if not root.is_dir():
            continue
        for path in _iter_python_files(root):
            violations.extend(_import_violations(path))
    assert violations == [], "\n".join(violations)


def test_capability_graph_catalog_does_not_own_binding_identity_resolution() -> None:
    tree = ast.parse(CATALOG_MODULE.read_text(encoding="utf-8-sig"), filename=str(CATALOG_MODULE))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == FORBIDDEN_SYMBOL:
            violations.append(f"{CATALOG_MODULE.name}: defines forbidden identity helper {FORBIDDEN_SYMBOL!r}")
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name == FORBIDDEN_SYMBOL:
                    violations.append(
                        f"{CATALOG_MODULE.name}: imports stale identity helper {FORBIDDEN_SYMBOL!r}",
                    )
    assert violations == [], "\n".join(violations)


def _normalize_typed_fields_identity_reflection_violations(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_normalize_typed_fields":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if child.func.id in REFLECTION_HELPERS:
                    violations.append(
                        f"{MANIFEST_MODULE.name}: _normalize_typed_fields uses forbidden {child.func.id}()",
                    )
            if isinstance(child, ast.Import) or isinstance(child, ast.ImportFrom):
                module_names = []
                if isinstance(child, ast.Import):
                    module_names.extend(alias.name for alias in child.names)
                elif child.module:
                    module_names.append(child.module)
                for module_name in module_names:
                    base = module_name.split(".", 1)[0]
                    if base in REFLECTION_MODULES:
                        violations.append(
                            f"{MANIFEST_MODULE.name}: _normalize_typed_fields imports forbidden module {module_name!r}",
                        )
    return violations


def test_agent_binding_normalize_typed_fields_has_no_identity_reflection() -> None:
    tree = ast.parse(
        MANIFEST_MODULE.read_text(encoding="utf-8-sig"),
        filename=str(MANIFEST_MODULE),
    )
    violations = _normalize_typed_fields_identity_reflection_violations(tree)
    assert violations == [], "\n".join(violations)
