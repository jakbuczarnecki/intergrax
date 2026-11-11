# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from intergrax.contracts.mcp_server_definition import (
    MCP_SERVER_DEFINITION_CONTRACT_VERSION,
    McpCatalogToolBinding,
    McpExposurePolicy,
    McpHostBuiltinBinding,
    McpHostBuiltinId,
    McpServerDefinition,
    McpServerDefinitionErrorCode,
    McpServerDefinitionValidationError,
    build_mcp_server_definition,
    resolve_effective_bindings,
    validate_mcp_server_definition,
)

pytestmark = [pytest.mark.unit]


def test_contract_module_has_no_fastmcp_import() -> None:
    path = Path(importlib.import_module("intergrax.contracts.mcp_server_definition").__file__ or "")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "fastmcp" not in alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert "fastmcp" not in node.module


def test_required_fields_reject_incomplete_definition() -> None:
    with pytest.raises(ValueError):
        McpServerDefinition(
            server_id="",
            display_name="x",
            default_capability="cap",
        )


def test_unknown_allowlist_binding_fails_closed() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap.a",
        bindings=(McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("missing",)),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        validate_mcp_server_definition(definition)
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN


def test_unknown_catalog_tool_id_fails_closed() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap.a",
        bindings=(
            McpHostBuiltinBinding(
                binding_id="list_catalog_tools",
                builtin=McpHostBuiltinId.LIST_CATALOG_TOOLS,
            ),
            McpCatalogToolBinding(binding_id="t1", tool_id="does.not.exist"),
        ),
    )
    with pytest.raises(McpServerDefinitionValidationError) as exc:
        validate_mcp_server_definition(definition, known_catalog_tool_ids=frozenset({"other.tool"}))
    assert exc.value.code == McpServerDefinitionErrorCode.MCP_BINDING_UNKNOWN_TOOL


def test_exposure_policy_filters_discovery_bindings() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap.a",
        bindings=(
            McpHostBuiltinBinding(binding_id="list_agents", builtin=McpHostBuiltinId.LIST_AGENTS),
            McpHostBuiltinBinding(binding_id="run_agent", builtin=McpHostBuiltinId.RUN_AGENT),
        ),
        exposure_policy=McpExposurePolicy(enabled=True, allowed_binding_ids=("list_agents",)),
    )
    exposed = resolve_effective_bindings(definition)
    assert len(exposed) == 1
    assert exposed[0].binding_id == "list_agents"


def test_contract_version_constant() -> None:
    definition = build_mcp_server_definition(
        server_id="srv",
        display_name="Srv",
        default_capability="cap.a",
        bindings=(),
    )
    assert definition.contract_version == MCP_SERVER_DEFINITION_CONTRACT_VERSION
