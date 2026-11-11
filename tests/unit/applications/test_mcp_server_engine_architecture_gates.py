# © Artur Czarnecki. All rights reserved.

"""MCP-P2-06 — engine must not host a shadow tool_id → handler registry."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE = REPO_ROOT / "intergrax" / "applications" / "_shared" / "mcp_server_engine.py"

_FORBIDDEN_NAME_PATTERNS = (
    re.compile(r".*tool_handlers.*", re.IGNORECASE),
    re.compile(r".*_handler_map.*", re.IGNORECASE),
    re.compile(r"^SHADOW_.*", re.IGNORECASE),
)


def test_engine_module_has_no_shadow_tool_handler_map() -> None:
    source = ENGINE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if any(pattern.match(name) for pattern in _FORBIDDEN_NAME_PATTERNS):
                        violations.append(name)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if any(pattern.match(name) for pattern in _FORBIDDEN_NAME_PATTERNS):
                violations.append(name)
    assert not violations, f"forbidden handler-map identifiers: {sorted(violations)}"


def test_engine_delegates_catalog_mount_to_existing_bridge() -> None:
    source = ENGINE.read_text(encoding="utf-8")
    assert "mount_catalog_tools_on_mcp" in source
    assert "execute_mcp_catalog_tool_invocation" not in source or "mount_catalog_tools_on_mcp" in source
