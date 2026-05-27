# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Invokes Tier-1 Nexus capability steps via :class:`~intergrax.runtime.nexus.tools.tool_runtime.ToolRuntime`.
"""

from __future__ import annotations

import json

from legal.domain.legal_agent_state import LegalAgentState
from legal.domain.legal_tool_plan import LegalToolPlan
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.tool_runtime import ToolRuntime


async def run_legal_tool_runtime_bridge(
    *,
    state: RuntimeState,
    plan: LegalToolPlan,
) -> None:
    """Order: RAG → websearch → tools (matches typical context layering)."""
    await ToolRuntime.invoke(
        state=state,
        plan=ToolRuntime.plan_from_like(plan),
        trace_step="LegalToolBridge",
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
