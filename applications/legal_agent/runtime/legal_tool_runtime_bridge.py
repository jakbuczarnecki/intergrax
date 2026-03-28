# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Invokes Tier-1 Nexus steps only (RAG, websearch, tools) — no custom executor.
"""

from __future__ import annotations

import json

from legal_agent.domain.legal_agent_state import LegalAgentState
from legal_agent.domain.legal_tool_plan import LegalToolPlan
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.runtime_steps.rag_step import RagStep
from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.nexus.runtime_steps.websearch_step import WebsearchStep
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


async def run_legal_tool_runtime_bridge(
    *,
    state: RuntimeState,
    plan: LegalToolPlan,
) -> None:
    """
    Order: RAG → websearch → tools (matches typical context layering).
    """
    cfg = state.context.config

    if plan.use_rag:
        if cfg.enable_rag:
            await RagStep().run(state)
        else:
            state.trace_event(
                component=TraceComponent.PIPELINE,
                step="LegalToolBridge",
                message="Plan requested RAG but enable_rag is false; skipping RagStep.",
                level=TraceLevel.WARNING,
            )

    if plan.use_websearch:
        if cfg.enable_websearch:
            await WebsearchStep().run(state)
        else:
            state.trace_event(
                component=TraceComponent.PIPELINE,
                step="LegalToolBridge",
                message="Plan requested websearch but enable_websearch is false; skipping.",
                level=TraceLevel.WARNING,
            )

    if plan.use_tools:
        if cfg.tools_agent and cfg.tool_invoker and cfg.tools_mode != "off":
            await ToolsStep().run(state)
        else:
            state.trace_event(
                component=TraceComponent.PIPELINE,
                step="LegalToolBridge",
                message="Plan requested tools but tools are off or not configured; skipping.",
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
