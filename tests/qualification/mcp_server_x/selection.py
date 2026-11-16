# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P5 regression selection (MCP-P5-01..07)."""

from __future__ import annotations

MCP_SERVER_X_P5_SELECTION_BATCH: tuple[str, ...] = (
    "tests/unit/applications/test_mcp_trust_a_principal_binding.py",
    "tests/unit/applications/test_mcp_canonical_execution.py",
    "tests/unit/applications/test_mcp_cancel_a_transport_governed_cancel.py",
    "tests/unit/applications/test_mcp_catalog_gate_b_cancel_c_qualification.py",
    "tests/unit/applications/test_mcp_catalog_child_execution_qualification.py",
    "tests/unit/applications/test_mcp_catalog_discovery_qualification.py",
    "tests/unit/applications/test_mcp_engine_concurrency.py",
    "tests/unit/applications/test_mcp_host_mount.py",
    "tests/unit/applications/test_mcp_server_engine.py",
    "tests/unit/applications/test_mcp_server_p4_lifecycle.py",
    "tests/unit/applications/test_reference_apps_mcp.py",
    "tests/unit/contracts/test_mcp_server_definition.py",
)
