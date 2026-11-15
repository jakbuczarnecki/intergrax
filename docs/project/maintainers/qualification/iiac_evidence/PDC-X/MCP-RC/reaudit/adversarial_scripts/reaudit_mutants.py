"""Re-audit own mutants R1-R14 for the NEW code of the MCP-RC corrective loop (throw-away worktree only).

Usage: python reaudit_mutants.py --repo <checkout> --python <py> --scratch <new path> --out <dir>
Reuses the implementer's runner (git worktree add --detach, one mutation at a time, one pytest process).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("AUDIT_ROOT", Path.cwd())).resolve()  # the checkout under audit
sys.path.insert(0, str(REPO))
import scripts.qualification.mcp_rc_mutation as h  # noqa: E402

S = h._SHARED
UA = "tests/unit/applications"
RC = (f"{UA}/test_mcp_rc_audit_corrections.py",)
RCW = RC + (f"{UA}/test_mcp_identity_mode_fail_closed.py", f"{UA}/test_mcp_result_bounds_and_error_contract.py",
            f"{UA}/test_mcp_deadline_and_resilience.py")
TC = f"{S}/mcp_host_execution_transport_cancel.py"
EN = f"{S}/mcp_server_engine.py"
CT = f"{S}/mcp_catalog_tools.py"
WE = f"{S}/mcp_wire_errors.py"

h.MUTANTS = (
    h.Mutant("R1-sanitizer-passes-non-toolerror-through", "sanitized_tool_errors converts every non-wire exception", WE,
             '    except Exception as exc:\n        _LOG.exception("MCP tool %s failed", tool_name)\n',
             '    except ZeroDivisionError as exc:\n        _LOG.exception("MCP tool %s failed", tool_name)\n', RC),
    h.Mutant("R2-custom-gate-bypassed-for-every-tool", "custom host tools are gated unless explicitly public", EN,
             "                if binding.tool_name in deps.public_custom_tool_names\n",
             "                if True\n", RC),
    h.Mutant("R3-outcome-unknown-reported-as-success", "outcome_unknown after the bound is never a success", TC,
             '    raise McpHostExecutionDeadlineExceeded("outcome_unknown", max_call_seconds)\n',
             '    return TaskResult(task_id="mutant", state=TaskState.COMPLETED, answer="ok")\n', RC + (f"{UA}/test_mcp_deadline_and_resilience.py",)),
    h.Mutant("R4-public-opt-out-inverted", "only listed custom tools are public", EN,
             "                if binding.tool_name in deps.public_custom_tool_names\n",
             "                if binding.tool_name not in deps.public_custom_tool_names\n", RC),
    h.Mutant("R5-toolerror-not-passed-through", "wire ToolErrors keep their MCP code", WE,
             "    except (ToolError, asyncio.CancelledError, McpConcurrentToolCallLimitExceeded):\n",
             "    except (asyncio.CancelledError, McpConcurrentToolCallLimitExceeded):\n", RC),
    h.Mutant("R6-host-cancelled-after-bound-propagates", "host ending cancelled after the bound is MCP_DEADLINE_EXCEEDED", TC,
             '            raise McpHostExecutionDeadlineExceeded("cancelled", max_call_seconds) from None\n',
             "            raise\n", RC),
    h.Mutant("R7-custom-gate-evaluated-after-handler", "the principal gate runs before the custom handler", EN,
             "        if gate is not None:\n            gate()\n        result = await handler(**kwargs)\n",
             "        result = await handler(**kwargs)\n        if gate is not None:\n            gate()\n", RC),
    h.Mutant("R8-invoke-catalog-tool-unsanitized", "invoke_catalog_tool never echoes raw exception text", CT,
             'async with optional_mcp_call_admission(call_admission), sanitized_tool_errors("invoke_catalog_tool"):',
             "async with optional_mcp_call_admission(call_admission):", RCW),
    h.Mutant("R9-run-agent-unsanitized", "run_agent never echoes raw exception text", EN,
             'async with optional_mcp_call_admission(call_admission), sanitized_tool_errors("run_agent"):',
             "async with optional_mcp_call_admission(call_admission):", RC),
    h.Mutant("R10-list-agents-unsanitized", "list_agents never echoes raw exception text", EN,
             'async with optional_mcp_call_admission(call_admission), sanitized_tool_errors("list_agents"):',
             "async with optional_mcp_call_admission(call_admission):", RC),
    h.Mutant("R11-custom-handler-unsanitized", "custom handler exceptions are never echoed", EN,
             '    async with sanitized_tool_errors(getattr(handler, "__name__", "custom_tool")):\n',
             "    if True:\n", RCW),
    h.Mutant("R12-custom-gate-evaluated-in-local-dev-mode", "the custom gate uses the server identity mode", EN,
             "                else lambda: require_mcp_caller_principal(identity_mode=identity_mode)\n",
             "                else lambda: require_mcp_caller_principal(identity_mode=McpIdentityMode.LOCAL_DEV_EXPLICIT)\n", RC),
    h.Mutant("R13-failed-host-mislabelled-cancelled", "a failed host after the bound reports outcome=failed", TC,
             '            raise McpHostExecutionDeadlineExceeded("failed", max_call_seconds) from exc\n',
             '            raise McpHostExecutionDeadlineExceeded("cancelled", max_call_seconds) from exc\n', RC),
    h.Mutant("R14-describe-catalog-unsanitized", "describe_catalog_tool never echoes raw exception text", CT,
             '            async with sanitized_tool_errors("describe_catalog_tool"):\n',
             "            if True:\n", RC),
    h.Mutant("R15-list-catalog-unsanitized", "list_catalog_tools never echoes raw exception text", CT,
             '            async with sanitized_tool_errors("list_catalog_tools"):\n',
             "            if True:\n", RC),
    h.Mutant("R16-bounded-observe-swallow-widened-to-base", "observing the host never hides a non-host failure (control: expected equivalent/benign)", TC,
             "    except Exception:  # noqa: BLE001 - host outcome is retrieved separately\n        pass\n    return host_coro.done()",
             "    except BaseException:  # noqa: BLE001\n        pass\n    return host_coro.done()", RC + (f"{UA}/test_mcp_deadline_and_resilience.py",)),
)

if __name__ == "__main__":
    raise SystemExit(h.main(sys.argv[1:]))
