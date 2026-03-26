# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Tier-2 component: chooses RAG / websearch / tools intent before legal stages.

Execution is delegated to Nexus :class:`~intergrax.runtime.nexus.runtime_steps.rag_step.RagStep`,
:class:`~intergrax.runtime.nexus.runtime_steps.websearch_step.WebsearchStep`, and
:class:`~intergrax.runtime.nexus.runtime_steps.tools_step.ToolsStep` (no new invoker).
"""

from __future__ import annotations

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_llm_prompts import (
    LEGAL_TOOL_DECISION_SYSTEM,
    legal_tool_decision_user,
)
from intergrax.agents_packages.legal_agent.legal_tool_plan import (
    LegalToolPlan,
    compute_legal_tool_intent_from_layers,
)
from intergrax.agents_packages.legal_agent.legal_memory_policy import build_legal_conversation_snippet
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


def _runtime_capability_flags(state: RuntimeState) -> tuple[bool, bool, bool]:
    ctx = state.context
    cfg = ctx.config
    rag_ok = bool(cfg.enable_rag and ctx.context_builder is not None)
    web_ok = bool(
        cfg.enable_websearch
        and ctx.websearch_executor is not None
        and ctx.websearch_prompt_builder is not None
    )
    tools_ok = bool(
        cfg.tools_agent is not None
        and cfg.tool_invoker is not None
        and cfg.tools_mode != "off"
    )
    return rag_ok, web_ok, tools_ok


def _clamp_plan_to_capabilities(
    plan: LegalToolPlan,
    *,
    rag_ok: bool,
    web_ok: bool,
    tools_ok: bool,
) -> LegalToolPlan:
    return plan.model_copy(
        update={
            "use_rag": bool(plan.use_rag and rag_ok),
            "use_websearch": bool(plan.use_websearch and web_ok),
            "use_tools": bool(plan.use_tools and tools_ok),
        }
    )


async def decide_legal_tool_plan(
    *,
    state: RuntimeState,
    legal_config: LegalAgentConfig,
) -> LegalToolPlan:
    """
    LLM structured decision, clamped to actual runtime capabilities.
    """
    if not legal_config.use_legal_tool_decision:
        return LegalToolPlan.default_llm_only()

    rag_ok, web_ok, tools_ok = _runtime_capability_flags(state)
    req = state.request
    has_attachments = bool(req.attachments)
    snippet = build_legal_conversation_snippet(state, policy=legal_config.memory_policy)

    llm = legal_config.llm_adapter
    user = legal_tool_decision_user(
        user_message=(req.message or "").strip(),
        has_attachments=has_attachments,
        conversation_snippet=snippet,
        rag_available=rag_ok,
        websearch_available=web_ok,
        tools_available=tools_ok,
    )
    messages = [
        ChatMessage(role="system", content=LEGAL_TOOL_DECISION_SYSTEM),
        ChatMessage(role="user", content=user),
    ]

    try:
        raw = llm.generate_structured(
            messages,
            LegalToolPlan,
            run_id=state.run_id,
        )
        plan = raw if isinstance(raw, LegalToolPlan) else LegalToolPlan.default_llm_only()
    except Exception:
        plan = LegalToolPlan.default_llm_only()

    plan = _clamp_plan_to_capabilities(
        plan,
        rag_ok=rag_ok,
        web_ok=web_ok,
        tools_ok=tools_ok,
    )
    # Structured models sometimes emit intent labels that disagree with layer flags; flags win.
    plan = plan.model_copy(
        update={
            "intent": compute_legal_tool_intent_from_layers(
                use_rag=plan.use_rag,
                use_tools=plan.use_tools,
                use_websearch=plan.use_websearch,
            )
        }
    )

    state.trace_event(
        component=TraceComponent.PIPELINE,
        step="LegalToolDecision",
        message=(
            f"intent={plan.intent} conf={plan.confidence:.2f} "
            f"rag={plan.use_rag} web={plan.use_websearch} tools={plan.use_tools}"
        ),
        level=TraceLevel.INFO,
    )

    return plan
