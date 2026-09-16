# © Artur Czarnecki. All rights reserved.

"""ME-5-C1 recommendation architecture boundary gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_RECOMMENDATION_MODULES = (
    "intergrax.capability_catalog.recommendation",
    "intergrax.capability_catalog.recommendation_validation",
    "intergrax.capability_catalog.recommended_capability",
)

_FORBIDDEN_GOVERNANCE_IMPLEMENTATION = (
    "intergrax.capability_catalog.governance",
)


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_recommendation_modules_do_not_import_governance_implementation() -> None:
    for module_name in _RECOMMENDATION_MODULES:
        module = importlib.import_module(module_name)
        path = Path(module.__file__)
        for imported in _collect_imports(path):
            for forbidden in _FORBIDDEN_GOVERNANCE_IMPLEMENTATION:
                if imported == forbidden or imported.startswith(f"{forbidden}."):
                    raise AssertionError(
                        f"{module_name} must not import governance implementation: {imported}",
                    )


def test_marketplace_discovery_does_not_invoke_recommendation_on_ranked_pipeline() -> None:
    module = importlib.import_module("intergrax.marketplace.discovery")
    path = Path(module.__file__)
    text = path.read_text(encoding="utf-8")
    assert "recommend_capability_candidates" not in text
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "recommend_capability_candidates":
                raise AssertionError("discovery must not call recommend_capability_candidates")
