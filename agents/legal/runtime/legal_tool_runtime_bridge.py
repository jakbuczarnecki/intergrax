# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification or distribution without written permission is prohibited.

"""
Invokes Tier-1 capabilities through the canonical tool gateway (§42.12).

Agents MUST NOT import Nexus runtime steps directly — use ``ToolRequest`` only.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from legal.domain.legal_agent_state import LegalAgentState
from legal.domain.legal_tool_plan import LegalToolPlan
from intergrax.contracts.tool_request import ToolRequest, ToolResponseStatus
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.tool_gateway import NEXUS_CAPABILITY_PLAN, RuntimeToolGateway
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


def _allowed_tools_from_state(state: RuntimeState) -> Optional[Sequence[str]]:
    raw = state.request.metadata.get("allowed_tools")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        return list(raw)
    return None


async def run_legal_tool_runtime_bridge(
    *,
    state: RuntimeState,
    plan: LegalToolPlan,
    agent_id: str = "legal",
    step_id: str = "legal_tool_bridge",
) -> None:
    """Order: RAG → websearch → tools via ``RuntimeToolGateway`` (capability plan)."""
    gateway = RuntimeToolGateway.for_state(
        state,
        allowed_tools=_allowed_tools_from_state(state),
        trace_step="LegalToolBridge",
    )
    request = ToolRequest(
        tool_name=NEXUS_CAPABILITY_PLAN,
        agent_id=agent_id,
        step_id=step_id,
        input={
            "tool_ids": plan.resolved_tool_ids(),
            "use_tools": plan.use_tools,
        },
    )
    response = await gateway.invoke(request)
    if response.status != ToolResponseStatus.SUCCESS:
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalToolBridge",
            message=f"tool gateway {response.status.value}: {response.error or 'unknown'}",
            level=TraceLevel.WARNING,
        )


def sync_legal_tool_runtime_feedback(
    agent_state: LegalAgentState,
    state: RuntimeState,
) -> None:
    """Persist compact runtime tool/RAG/web flags for legal routing metrics."""
    fb = {
        "used_rag": state.used_rag,
        "used_websearch": state.used_websearch,
        "used_tools": state.used_tools,
        "tool_trace_count": len(state.tool_traces or []),
        "tool_names": [t.tool_name for t in (state.tool_traces or [])],
    }
    agent_state.legal_tool_runtime_feedback_json = json.dumps(
        fb,
        ensure_ascii=False,
        sort_keys=True,
    )
