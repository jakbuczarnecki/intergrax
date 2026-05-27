# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Platform-side dynamic governance for Legal Agent (SaaS / multi-tenant factory).

Provides:

* :class:`LegalExecutionPolicyPort` — resolve per-request caps for Nexus layers
  (implement with DB, billing, feature flags; read :class:`~intergrax.runtime.nexus.engine.runtime_state.RuntimeState`.request).
* :class:`ResolvingLegalToolPlanGovernance` — :class:`LegalToolPlanGovernancePort` that applies those caps before the runtime bridge.
* :class:`DualLegalGovernanceService` — same instance for ``governance_service`` and ``legal_tool_plan_governance``
  (post-run :meth:`~intergrax.runtime.governance.service.GovernanceService.evaluate` + pre-bridge adjust).

Static org clamp (:func:`~legal.governance.legal_tool_plan_governance.enforce_legal_tool_plan_governance`)
still runs first; this layer applies **additional** caps (AND with the plan).

See also :mod:`legal_agent.governance.legal_execution_policy_sources` (tenant registry,
request metadata, policy chains) and :mod:`legal_agent.governance.legal_agent_governance_wiring`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from legal.config.legal_agent_config import LegalAgentConfig
from legal.domain.legal_tool_plan import (
    LegalToolPlan,
    compute_legal_tool_intent_from_layers,
)
from legal.governance.legal_tool_plan_governance_port import (
    LegalToolPlanGovernancePort,
)
from legal.tracing.legal_tool_plan_governance_clamp_diag_v1 import (
    LegalToolPlanGovernanceClampDiagV1,
)
from intergrax.runtime.governance.execution_guard import ExecutionGuard
from intergrax.runtime.governance.service import GovernanceService
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


@dataclass(frozen=True, slots=True)
class LegalNexusLayerCaps:
    """
    Upper bounds for Nexus layers after Tier-2 tool decision (AND with :class:`LegalToolPlan` flags).
    """

    allow_rag: bool = True
    allow_websearch: bool = True
    allow_tools: bool = True


@runtime_checkable
class LegalExecutionPolicyPort(Protocol):
    """
    Platform contract: resolve caps from tenant / request / subscription context.

    Implementations should stay fast (cache, in-memory policy snapshot); no full replay here.
    """

    def resolve_nexus_layer_caps(
        self,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalNexusLayerCaps:
        ...


def apply_legal_nexus_layer_caps_to_plan(
    *,
    plan: LegalToolPlan,
    state: RuntimeState,
    caps: LegalNexusLayerCaps,
) -> LegalToolPlan:
    """
    AND plan layer flags with ``caps``; reconcile ``intent`` and emit clamp diagnostics when a layer drops.
    """
    use_rag = plan.use_rag and caps.allow_rag
    use_tools = plan.use_tools and caps.allow_tools
    use_websearch = plan.use_websearch and caps.allow_websearch
    changed = False

    if plan.use_rag and not use_rag:
        changed = True
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalToolPlanGovernance",
            message="rag disabled by execution policy",
            level=TraceLevel.WARNING,
            payload=LegalToolPlanGovernanceClampDiagV1(
                layer="rag",
                reason_code="execution_policy_disallows_nexus_rag",
            ),
        )

    if plan.use_websearch and not use_websearch:
        changed = True
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalToolPlanGovernance",
            message="websearch disabled by execution policy",
            level=TraceLevel.WARNING,
            payload=LegalToolPlanGovernanceClampDiagV1(
                layer="websearch",
                reason_code="execution_policy_disallows_nexus_websearch",
            ),
        )

    if plan.use_tools and not use_tools:
        changed = True
        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalToolPlanGovernance",
            message="tools disabled by execution policy",
            level=TraceLevel.WARNING,
            payload=LegalToolPlanGovernanceClampDiagV1(
                layer="tools",
                reason_code="execution_policy_disallows_nexus_tools",
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
    note = "governance: execution policy clamp applied to Nexus layers"
    new_summary = f"{summary} | {note}" if summary else note

    return plan.model_copy(
        update={
            "use_rag": use_rag,
            "use_tools": use_tools,
            "use_websearch": use_websearch,
            "intent": new_intent,
            "reasoning_summary": new_summary,
        }
    )


@dataclass(slots=True)
class StaticLegalExecutionPolicy:
    """
    Fixed caps (tests, bootstrap, or config snapshot from factory at agent build time).
    """

    caps: LegalNexusLayerCaps = field(default_factory=LegalNexusLayerCaps)

    def resolve_nexus_layer_caps(
        self,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalNexusLayerCaps:
        return self.caps


@dataclass(slots=True)
class ResolvingLegalToolPlanGovernance(LegalToolPlanGovernancePort):
    """
    :class:`LegalToolPlanGovernancePort` backed by :class:`LegalExecutionPolicyPort`.
    """

    policy: LegalExecutionPolicyPort

    def adjust_legal_tool_plan(
        self,
        plan: LegalToolPlan,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalToolPlan:
        caps = self.policy.resolve_nexus_layer_caps(state=state, legal_config=legal_config)
        return apply_legal_nexus_layer_caps_to_plan(plan=plan, state=state, caps=caps)


class DualLegalGovernanceService(GovernanceService, LegalToolPlanGovernancePort):
    """
    Single object for ``LegalAgentConfig.governance_service`` and ``legal_tool_plan_governance``.
    """

    def __init__(self, guard: ExecutionGuard, policy: LegalExecutionPolicyPort) -> None:
        super().__init__(guard=guard)
        self._policy = policy

    def adjust_legal_tool_plan(
        self,
        plan: LegalToolPlan,
        *,
        state: RuntimeState,
        legal_config: LegalAgentConfig,
    ) -> LegalToolPlan:
        caps = self._policy.resolve_nexus_layer_caps(state=state, legal_config=legal_config)
        return apply_legal_nexus_layer_caps_to_plan(plan=plan, state=state, caps=caps)
