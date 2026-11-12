# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 selections (MCP-P7-01..09): new qualification files and the regression set."""

from __future__ import annotations

from tests.qualification.mcp_server_x.selection import MCP_SERVER_X_P5_SELECTION_BATCH

_QM = "tests/qualification/mcp_server_x"
_UA = "tests/unit/applications"

# New P7 files that carry the MCP-ACC-01..18 evidence (each test is named test_acc_<NN>_...).
MCP_SERVER_X_P7_NEW_QUALIFICATION: tuple[str, ...] = (
    f"{_QM}/test_mcp_server_x_p7_l1_contract.py",
    f"{_QM}/test_mcp_server_x_p7_l2_composition_exposure.py",
    f"{_QM}/test_mcp_server_x_p7_l2_security_trace.py",
    f"{_QM}/test_mcp_server_x_p7_l3_publication_discovery.py",
    f"{_QM}/test_mcp_server_x_p7_l3_invoke_security.py",
    f"{_QM}/test_mcp_server_x_p7_l3_protocol_subset.py",
    f"{_QM}/test_mcp_server_x_p7_failure_injection_l2.py",
    f"{_QM}/test_mcp_server_x_p7_failure_injection_l3.py",
)

# Existing MCP tests re-run as regression (MCP-P7-06), in addition to the P5 selection.
MCP_SERVER_X_P7_REGRESSION_EXTRA: tuple[str, ...] = (
    f"{_UA}/test_mcp_public_api_surface.py",
    f"{_UA}/test_fastapi_mcp.py",
    f"{_UA}/test_mcp_server_engine_architecture_gates.py",
    f"{_UA}/test_mcp_catalog_cross_loop_ingress_qualification.py",
    f"{_UA}/test_mcp_catalog_f3_streamable_http_physical_l3.py",
    f"{_UA}/test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py",
    f"{_QM}/test_mcp_server_x_p6_reference_l3_qualification.py",
)

# Pre-existing failures on this machine, NOT introduced by P6/P7 and NOT part of any earlier qualified
# selection: the same 11 tests fail identically at c59892829 (P5 publication) and d5c1383fc (P7 cut)
# and at the P7 head (logs: .tmp/session/mcp-p7p8-run/baseline_p7-base-*.log). Causes seen in the logs:
# "LLM provider not explicitly configured", STRICT agent contract needs revision-bound authority, and the
# TL11 catalog subprocess startup timeout. They are recorded as residual R-P7-02 and are excluded from the
# P7 regression selection; they must be investigated outside P7 (do not treat them as passing).
MCP_SERVER_X_P7_KNOWN_BASELINE_RED: tuple[str, ...] = (
    f"{_UA}/test_mcp_surface_opt_in.py",
    f"{_UA}/test_mcp_surface_opt_in_runtime.py",
    f"{_UA}/test_mcp_catalog_e2e_hung_server_qualification.py",
)

MCP_SERVER_X_P7_REGRESSION_SELECTION: tuple[str, ...] = tuple(
    dict.fromkeys((*MCP_SERVER_X_P5_SELECTION_BATCH, *MCP_SERVER_X_P7_REGRESSION_EXTRA))
)

# MCP-ACC rows without a test_acc_<NN>_ function: they are proven by records, not by one test.
MCP_SERVER_X_P7_RECORD_ONLY_ACC: dict[int, str] = {
    19: "regression selection (MCP_SERVER_X_P7_REGRESSION_SELECTION) executed with per-file exit codes",
    20: "exact-SHA package record MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md",
}
