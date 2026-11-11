# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P6 regression selection (MCP-P6-01..04 + P4/P5 subset)."""

from __future__ import annotations

MCP_SERVER_X_P6_SELECTION_BATCH: tuple[str, ...] = (
    "tests/unit/applications/test_mcp_public_api_surface.py",
    "tests/unit/contracts/test_mcp_server_definition.py",
    "tests/unit/applications/test_mcp_server_engine.py",
    "tests/unit/applications/test_mcp_server_p4_lifecycle.py",
    "tests/unit/applications/test_mcp_host_mount.py",
    "tests/unit/applications/test_reference_apps_mcp.py",
    "tests/qualification/mcp_server_x/test_mcp_server_x_p6_reference_l3_qualification.py",
    "tests/qualification/mcp_server_x/test_mcp_server_x_p5_qualification_batch.py",
)
