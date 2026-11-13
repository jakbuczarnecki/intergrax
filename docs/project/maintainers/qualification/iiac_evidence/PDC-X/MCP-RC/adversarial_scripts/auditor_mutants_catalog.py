"""Auditor-own mutants A12/A13: default-mode principal gate removed from catalog discovery tools."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import scripts.qualification.mcp_rc_mutation as h  # noqa: E402

S = h._SHARED
UA = "tests/unit/applications"
CAT = (
    f"{UA}/test_mcp_identity_mode_fail_closed.py",
    f"{UA}/test_mcp_result_bounds_and_error_contract.py",
    f"{UA}/test_mcp_catalog_discovery_qualification.py",
    f"{UA}/test_mcp_trust_a_principal_binding.py",
    "tests/qualification/mcp_server_x/test_mcp_server_x_p7_l2_security_trace.py",
)
GATE = "            require_mcp_caller_principal(identity_mode=identity_mode)\n"
h.MUTANTS = (
    h.Mutant(
        "A12-list-catalog-tools-principal-gate-removed",
        "list_catalog_tools requires a principal in the default mode",
        f"{S}/mcp_catalog_tools.py",
        '            """List enabled catalog tools (tool_id, description, input schema)."""\n' + GATE,
        '            """List enabled catalog tools (tool_id, description, input schema)."""\n',
        CAT,
    ),
    h.Mutant(
        "A13-describe-catalog-tool-principal-gate-removed",
        "describe_catalog_tool requires a principal in the default mode",
        f"{S}/mcp_catalog_tools.py",
        '            """Return MCP schema for one catalog tool_id."""\n' + GATE,
        '            """Return MCP schema for one catalog tool_id."""\n',
        CAT,
    ),
)

if __name__ == "__main__":
    raise SystemExit(h.main(sys.argv[1:]))
