# © Artur Czarnecki. All rights reserved.

"""CAPABILITY-CATALOG-1 Stage 13 architecture boundary regression gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_METERING_MODULE = "intergrax.capability_metering"
_CATALOG_MODULE = "intergrax.capability_catalog"

_FORBIDDEN_METERING_IMPORT_PREFIXES = (
    "intergrax.marketplace",
    "intergrax.runtime.nexus.tools.invoker",
    "intergrax.tools.registry",
    "intergrax.skills.registry",
)

_FORBIDDEN_PRICE_TOKENS = frozenset(
    {
        "price",
        "billing",
        "rate",
        "cost_per",
        "subscription",
    },
)

_REGISTRY_CLASS_PATHS = (
    ("intergrax.tools.registry.runtime", "ToolRegistry"),
    ("intergrax.skills.registry.runtime", "SkillRegistry"),
    ("intergrax.runtime.registry.agent_registry", "AgentRegistry"),
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


def _collect_class_member_names(tree: ast.AST, class_name: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                names.append(item.target.id)
            elif isinstance(item, ast.FunctionDef):
                names.append(item.name)
    return names


def test_capability_metering_has_no_forbidden_runtime_or_marketplace_imports() -> None:
    root = _package_root(_METERING_MODULE)
    for path in _iter_package_py_files(_METERING_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_METERING_IMPORT_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"{path.relative_to(root)} imports forbidden dependency: {imported}",
                    )


def test_capability_catalog_does_not_import_metering() -> None:
    root = _package_root(_CATALOG_MODULE)
    for path in _iter_package_py_files(_CATALOG_MODULE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            if imported == _METERING_MODULE or imported.startswith(f"{_METERING_MODULE}."):
                raise AssertionError(
                    f"{path.relative_to(root)} imports metering package: {imported}",
                )


def test_domain_registries_have_no_stage13_price_or_billing_fields() -> None:
    for module_name, class_name in _REGISTRY_CLASS_PATHS:
        module = importlib.import_module(module_name)
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        member_names = _collect_class_member_names(tree, class_name)
        violations = [
            name
            for name in member_names
            if any(token in name.lower() for token in _FORBIDDEN_PRICE_TOKENS)
        ]
        assert not violations, (
            f"{module_name}.{class_name} exposes forbidden pricing/billing members: "
            + ", ".join(violations)
        )


def test_capability_usage_event_contract_has_no_monetary_fields() -> None:
    from intergrax.contracts.capability_metering import CapabilityUsageEvent

    assert "price" not in CapabilityUsageEvent.model_fields
    assert "cost" not in CapabilityUsageEvent.model_fields
    assert "billing" not in CapabilityUsageEvent.model_fields
