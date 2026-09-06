# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.sandbox.contracts import SandboxExecInput, SandboxExecOutput
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_TOOL_NAME
from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.runtime.sandbox.enforcement import resolve_tool_execution_environment
from intergrax.runtime.sandbox.execution_environment import ExecutionEnvironmentRequirement
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.core.contracts import ToolIsolationRequirement

SANDBOX_EXEC_TOOL_ID = SANDBOX_TOOL_NAME


def sandbox_exec(ctx: ToolWiringContext, params: SandboxExecInput) -> SandboxExecOutput:
    _, resolution_error = resolve_tool_execution_environment(
        ctx,
        requirement=ExecutionEnvironmentRequirement.from_tool_isolation(
            ToolIsolationRequirement.SANDBOX,
        ),
    )
    if resolution_error is not None:
        return resolution_error

    raw_session = ctx.sandbox_session or ctx.extras.get("sandbox_session")
    if raw_session is None:
        return SandboxExecOutput(success=False, error="sandbox_session_not_configured")
    if isinstance(raw_session, SandboxSession):
        session: SandboxExecCapable = raw_session
    elif isinstance(raw_session, SandboxExecCapable):
        session = raw_session
    else:
        return SandboxExecOutput(success=False, error="sandbox_session_invalid_type")

    result = session.execute(params.operation, dict(params.payload))
    output = dict(result.output or {})
    if result.audit_entry is not None:
        output["audit_entry_id"] = result.audit_entry.entry_id

    return SandboxExecOutput(
        success=bool(result.success),
        output=output,
        error=result.error or "",
        session_id=session.session_id,
    )
