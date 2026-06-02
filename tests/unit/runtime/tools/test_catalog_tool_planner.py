# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner


pytestmark = pytest.mark.gate


def test_catalog_tool_planner_module_does_not_import_tools_agent() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    source_path = repo_root / "intergrax" / "runtime" / "nexus" / "tools" / "catalog_tool_planner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "tools_agent" not in alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "tools_agent" not in node.module


def test_catalog_tool_planner_exposes_llm_from_service() -> None:
    assert hasattr(CatalogToolPlanner, "from_registry")
    assert hasattr(CatalogToolPlanner, "plan_tools")
