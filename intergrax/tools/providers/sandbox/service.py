# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.sandbox.contracts import SandboxExecInput, SandboxExecOutput
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_TOOL_NAME
from intergrax.runtime.sandbox.session import SandboxSession

SANDBOX_EXEC_TOOL_ID = SANDBOX_TOOL_NAME


def sandbox_exec(ctx: ToolWiringContext, params: SandboxExecInput) -> SandboxExecOutput:
    raw_session = ctx.sandbox_session or ctx.extras.get("sandbox_session")
    if raw_session is None:
        return SandboxExecOutput(success=False, error="sandbox_session_not_configured")
    if not isinstance(raw_session, SandboxSession):
        return SandboxExecOutput(success=False, error="sandbox_session_invalid_type")
    session: SandboxSession = raw_session

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
