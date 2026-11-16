"""Auditor-own mutants (not used by the implementer); reuses the harness runner on a throw-away worktree."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import scripts.qualification.mcp_rc_mutation as h  # noqa: E402

S = h._SHARED
UA = "tests/unit/applications"
ID = (f"{UA}/test_mcp_identity_mode_fail_closed.py",)
BO = (f"{UA}/test_mcp_result_bounds_and_error_contract.py",)
XL = BO + (f"{UA}/test_mcp_catalog_cross_loop_ingress_qualification.py", f"{UA}/test_mcp_catalog_f3_cross_loop_cancel_c_physical_l3.py")
DL = (f"{UA}/test_mcp_deadline_and_resilience.py",)
IV = (f"{UA}/test_mcp_invocation_id.py",)

h.MUTANTS = (
    h.Mutant("A1-local-dev-loopback-guard-removed", "LOCAL_DEV refuses non-loopback clients", f"{S}/mcp_harness_principal.py",
             "    if not is_loopback_client(request):\n        raise mcp_tool_error(", "    if False:\n        raise mcp_tool_error(", ID),
    h.Mutant("A2-catalog-invoke-unfiltered-registry", "catalog invoke composes the allowlist-filtered registry", f"{S}/mcp_server_engine.py",
             "                registry=registry_view,\n                tool_enablement=deps.tool_enablement,", "                registry=deps.tool_registry,\n                tool_enablement=deps.tool_enablement,", ID),
    h.Mutant("A3-cross-loop-payload-bound-removed", "cross-loop catalog payload is bounded at the MCP boundary", f"{S}/mcp_catalog_invocation_bridge.py",
             "            return bound_cross_loop_catalog_payload(terminal.payload)\n", "            return terminal.payload\n", XL),
    h.Mutant("A3b-cross-loop-success-payload-bound-removed", "cross-loop success payload is bounded", f"{S}/mcp_catalog_invocation_bridge.py",
             "            **bound_cross_loop_catalog_payload(terminal.payload),\n", "            **terminal.payload,\n", XL),
    h.Mutant("A4-cancelled-outcome-returned-as-success", "a non-completed outcome after the call bound is never success", f"{S}/mcp_host_execution_transport_cancel.py",
             "        if isinstance(outcome, TaskResult) and outcome.state is TaskState.COMPLETED:\n", "        if True:\n", DL),
    h.Mutant("A5-list-agents-principal-gate-removed", "list_agents requires a principal in the default mode", f"{S}/mcp_server_engine.py",
             "            async with optional_mcp_call_admission(call_admission):\n                require_mcp_caller_principal(identity_mode=identity_mode)\n                agents:",
             "            async with optional_mcp_call_admission(call_admission):\n                agents:", ID),
    h.Mutant("A6-request-time-auth-resolver-disabled", "identity wired after coupling still reaches MCP", f"{S}/fastapi_mcp.py",
             "    mcp_app.state.harness_auth_resolver = lambda: resolve_harness_auth_state_from_app(wrapper) or inner_auth",
             "    mcp_app.state.harness_auth_resolver = lambda: None", ID),
    h.Mutant("A7-custom-tool-exception-text-echoed", "custom tool exception text is never echoed", f"{S}/mcp_server_engine.py",
             'f"tool execution failed ({type(exc).__name__})"', 'f"tool execution failed ({exc})"', BO),
    h.Mutant("A8-unknown-tool-echo-uncapped", "unknown-tool echo is capped", f"{S}/mcp_catalog_tools.py",
             'f"unknown_tool:{tool_id[:_MAX_ECHOED_TOOL_ID_CHARS]}"', 'f"unknown_tool:{tool_id}"', BO),
    h.Mutant("A9-invocation-id-constant", "mcp_invocation_id is unique per call", "intergrax/applications/_shared/mcp_invocation_id.py",
             'return f"{_PREFIX}{uuid.uuid4().hex}"', 'return f"{_PREFIX}{\'0\' * 32}"', IV),
    h.Mutant("A10-unverified-label-dropped", "assertion-only identity is labelled unverified", f"{S}/mcp_nexus_server.py",
             "    if not identity_verified:\n", "    if False:\n", ID),
    h.Mutant("A11-local-dev-allowed-with-configured-auth", "LOCAL_DEV never downgrades configured auth", f"{S}/mcp_harness_principal.py",
             "    if _harness_auth_requires_principal(effective_harness_auth) and not _local_dev_harness_bypass(\n        effective_harness_auth,\n    ):",
             "    if False and not _local_dev_harness_bypass(\n        effective_harness_auth,\n    ):", ID),
)

if __name__ == "__main__":
    raise SystemExit(h.main(sys.argv[1:]))
