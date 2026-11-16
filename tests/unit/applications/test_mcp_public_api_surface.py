# © Artur Czarnecki. All rights reserved.

"""MCP-P6-01 — supported public import surface and Tier-3 facade import policy."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_FACADE_ROOT = _REPO / "applications"
_FORBIDDEN_PREFIXES = (
    "intergrax.applications._shared.mcp_catalog_invocation_bridge",
    "intergrax.applications._shared.mcp_catalog_side_effect_enforcement",
    "intergrax.applications._shared.mcp_server_engine",
    "intergrax.applications._shared.mcp_nexus_server",
)
_ALLOWED_SHARED_PREFIXES = (
    "intergrax.applications._shared.mcp_application_server",
    "intergrax.applications._shared.mcp_developer_api",
    "intergrax.applications._shared.mcp_host_mount",
    "intergrax.applications._shared.mcp_import_guard",
    "intergrax.applications._shared.fastapi_mcp",
)


def test_mcp_developer_api_exports_contract_and_compose() -> None:
    from intergrax.applications._shared import mcp_developer_api

    assert "compose_application_mcp_server" in mcp_developer_api.__all__
    assert "McpServerDefinition" in mcp_developer_api.__all__
    assert "execute_mcp_agent_task" in mcp_developer_api.__all__
    from intergrax.applications._shared.mcp_developer_api import (
        build_mcp_server_definition,
        compose_application_mcp_server,
    )

    assert callable(build_mcp_server_definition)
    assert callable(compose_application_mcp_server)


def _collect_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


@pytest.mark.parametrize(
    "rel_path",
    sorted(_FACADE_ROOT.glob("*/mcp/server.py")),
)
def test_tier3_mcp_facades_avoid_engine_internals(rel_path: Path) -> None:
    modules = _collect_import_modules(rel_path)
    for mod in modules:
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not mod.startswith(forbidden), f"{rel_path}: forbidden import {mod}"
        if mod.startswith("intergrax.applications._shared.mcp_"):
            assert any(mod.startswith(prefix) for prefix in _ALLOWED_SHARED_PREFIXES), (
                f"{rel_path}: non-public shared import {mod}"
            )
