# © Artur Czarnecki. All rights reserved.

"""MCP-RC (MCP-SERVER-X corrective campaign) mandatory test contract.

Supplements ``p7_selection`` (historical, unchanged): the P7 qualification and regression selections plus the
files repaired by MRC-5 (R-P7-02, formerly ``MCP_SERVER_X_P7_KNOWN_BASELINE_RED``) plus the MRC-1..4 suites.
Run with one pytest process, ``PYTHONPATH`` set to the checkout under test (subprocess fixtures import the
project from ``PYTHONPATH``, not from an editable install of another checkout).
"""

from __future__ import annotations

from tests.qualification.mcp_server_x.p7_selection import (
    MCP_SERVER_X_P7_KNOWN_BASELINE_RED,
    MCP_SERVER_X_P7_NEW_QUALIFICATION,
    MCP_SERVER_X_P7_REGRESSION_SELECTION,
)

_UA = "tests/unit/applications"
_QM = "tests/qualification/mcp_server_x"

MCP_SERVER_X_RC_NEW_SUITES: tuple[str, ...] = (
    f"{_UA}/test_mcp_identity_mode_fail_closed.py",
    f"{_UA}/test_mcp_result_bounds_and_error_contract.py",
    f"{_UA}/test_mcp_deadline_and_resilience.py",
    f"{_UA}/test_mcp_invocation_id.py",
    f"{_UA}/test_mcp_rc_audit_corrections.py",
)

# Existing suites outside the P7 selections that build MCP servers or call MCP tools (found by the RC impact sweep).
MCP_SERVER_X_RC_EXTRA_REGRESSION: tuple[str, ...] = (
    "tests/unit/tools/providers/test_tool_life_x_l3_r2_mcp_host_lifecycle_qualification.py",
    "tests/unit/runtime/external_operations/test_tool_life_x_r4_b2_mcp_class_c_fail_closed.py",
    "tests/qualification/host_01/test_host_01_gates.py",
)

MCP_SERVER_X_RC_SELECTION: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *MCP_SERVER_X_P7_NEW_QUALIFICATION,
            f"{_QM}/test_mcp_server_x_p7_matrix_consistency.py",
            *MCP_SERVER_X_P7_REGRESSION_SELECTION,
            *MCP_SERVER_X_P7_KNOWN_BASELINE_RED,
            *MCP_SERVER_X_RC_NEW_SUITES,
            *MCP_SERVER_X_RC_EXTRA_REGRESSION,
        )
    )
)

__all__ = ["MCP_SERVER_X_RC_EXTRA_REGRESSION", "MCP_SERVER_X_RC_NEW_SUITES", "MCP_SERVER_X_RC_SELECTION"]
