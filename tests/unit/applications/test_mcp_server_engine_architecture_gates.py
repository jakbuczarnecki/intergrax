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
    """The engine mounts catalog tools only through the bridge and never calls the invocation function itself."""
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "mount_catalog_tools_on_mcp" in called
    assert "execute_mcp_catalog_tool_invocation" not in called


def test_engine_has_no_second_identity_or_deadline_authority() -> None:
    """MCP-RC: the engine resolves identity through the single helper and defines no own authenticator/deadline."""
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    forbidden = {
        name
        for name in defined
        if any(part in name.lower() for part in ("authenticate", "verify_token", "deadline_authority"))
    }
    assert not forbidden, sorted(forbidden)
