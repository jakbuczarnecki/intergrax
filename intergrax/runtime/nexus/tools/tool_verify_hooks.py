# © Artur Czarnecki. All rights reserved.

"""Post-tool verification hooks (TOOL-ENG-7)."""

from __future__ import annotations

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tracing.tools.tool_invocation import ToolVerifyRequiredDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.tools.core.contracts import ToolRiskLevel


def emit_high_risk_tool_verify_signal(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    trace: ToolCallTrace,
) -> bool:
    """
    Emit trace when ``risk_level >= HIGH``.

    Returns ``True`` when verification signal was emitted (approval path deferred to CVL).
    """
    if not trace.success or not trace.tool_name:
        return False
    try:
        contract = invoker.registry.get(trace.tool_name).contract
    except KeyError:
        return False
    if contract.risk_level not in (ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL):
        return False
    state.trace_event(
        component=TraceComponent.TOOLS,
        step="tool_verify_required",
        message="High-risk tool invocation requires verification.",
        level=TraceLevel.WARNING,
        payload=ToolVerifyRequiredDiagV1(
            tool_id=contract.tool_id,
            risk_level=contract.risk_level.value,
        ),
    )
    return True
