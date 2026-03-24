# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Organization governance clamp for :class:`~intergrax.agents_packages.legal_agent.legal_tool_plan.LegalToolPlan`.

Runs **after** Tier-2 tool decision LLM and **before** :mod:`legal_tool_runtime_bridge`.
Degrades requested Nexus layers (RAG / websearch / tools) when the legal agent config
disallows them for the tenant/organization — no exceptions, trace per clamp.
"""

from __future__ import annotations

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_tool_plan import (
    LegalToolPlan,
    compute_legal_tool_intent_from_layers,
)
from intergrax.agents_packages.legal_agent.tracing.legal_tool_plan_governance_clamp_diag_v1 import (
    LegalToolPlanGovernanceClampDiagV1,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


def enforce_legal_tool_plan_governance(
    *,
    plan: LegalToolPlan,
    state: RuntimeState,
    legal_config: LegalAgentConfig,
) -> LegalToolPlan:
    """
    Apply organization-level allow flags from :class:`LegalAgentConfig`.

    Returns a possibly updated plan (``model_copy``) with reconciled ``intent``.
    Emits one WARNING trace per clamped layer.
    """
    use_rag = plan.use_rag
    use_tools = plan.use_tools
    use_websearch = plan.use_websearch
    changed = False

    if use_rag and not legal_config.organization_allow_rag:
        use_rag = False
        changed = True
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalToolPlanGovernance",
            message="rag disabled by organization governance",
            level=TraceLevel.WARNING,
            payload=LegalToolPlanGovernanceClampDiagV1(
                layer="rag",
                reason_code="organization_disallows_nexus_rag",
            ),
        )

    if use_websearch and not legal_config.organization_allow_websearch:
        use_websearch = False
        changed = True
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalToolPlanGovernance",
            message="websearch disabled by organization governance",
            level=TraceLevel.WARNING,
            payload=LegalToolPlanGovernanceClampDiagV1(
                layer="websearch",
                reason_code="organization_disallows_nexus_websearch",
            ),
        )

    if use_tools and not legal_config.organization_allow_tools:
        use_tools = False
        changed = True
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalToolPlanGovernance",
            message="tools disabled by organization governance",
            level=TraceLevel.WARNING,
            payload=LegalToolPlanGovernanceClampDiagV1(
                layer="tools",
                reason_code="organization_disallows_nexus_tools",
            ),
        )

    if not changed:
        return plan

    new_intent = compute_legal_tool_intent_from_layers(
        use_rag=use_rag,
        use_tools=use_tools,
        use_websearch=use_websearch,
    )
    summary = (plan.reasoning_summary or "").strip()
    gov_note = "governance: organization clamp applied to Nexus layers"
    new_summary = f"{summary} | {gov_note}" if summary else gov_note

    return plan.model_copy(
        update={
            "use_rag": use_rag,
            "use_tools": use_tools,
            "use_websearch": use_websearch,
            "intent": new_intent,
            "reasoning_summary": new_summary,
        }
    )
