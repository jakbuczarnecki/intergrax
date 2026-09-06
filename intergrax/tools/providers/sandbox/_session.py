# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.runtime.sandbox.enforcement import resolve_tool_execution_environment
from intergrax.runtime.sandbox.execution_environment import ExecutionEnvironmentRequirement
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.core.contracts import ToolIsolationRequirement
from intergrax.tools.providers.sandbox.contracts import SandboxExecOutput
from intergrax.tools.registry.wiring import ToolWiringContext


def resolve_sandbox_session(ctx: ToolWiringContext) -> SandboxExecCapable | None:
    raw_session = ctx.sandbox_session or ctx.extras.get("sandbox_session")
    if raw_session is None:
        return None
    if isinstance(raw_session, (SandboxSession, SandboxExecCapable)):
        return raw_session
    return None


def run_sandbox_operation(
    ctx: ToolWiringContext,
    operation: str,
    payload: dict,
) -> SandboxExecOutput:
    _, resolution_error = resolve_tool_execution_environment(
        ctx,
        requirement=ExecutionEnvironmentRequirement.from_tool_isolation(
            ToolIsolationRequirement.SANDBOX,
        ),
    )
    if resolution_error is not None:
        return resolution_error

    session = resolve_sandbox_session(ctx)
    if session is None:
        return SandboxExecOutput(success=False, error="sandbox_session_not_configured")
    result = session.execute(operation, dict(payload))
    output = dict(result.output or {})
    if result.audit_entry is not None:
        output["audit_entry_id"] = result.audit_entry.entry_id
    return SandboxExecOutput(
        success=bool(result.success),
        output=output,
        error=result.error or "",
        session_id=session.session_id,
    )
