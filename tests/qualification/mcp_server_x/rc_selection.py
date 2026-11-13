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
)

MCP_SERVER_X_RC_SELECTION: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *MCP_SERVER_X_P7_NEW_QUALIFICATION,
            f"{_QM}/test_mcp_server_x_p7_matrix_consistency.py",
            *MCP_SERVER_X_P7_REGRESSION_SELECTION,
            *MCP_SERVER_X_P7_KNOWN_BASELINE_RED,
            *MCP_SERVER_X_RC_NEW_SUITES,
        )
    )
)

__all__ = ["MCP_SERVER_X_RC_NEW_SUITES", "MCP_SERVER_X_RC_SELECTION"]
