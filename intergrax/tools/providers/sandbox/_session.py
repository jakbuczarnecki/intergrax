# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.sandbox.contracts import SandboxExecCapable
from intergrax.runtime.sandbox.session import SandboxSession
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
