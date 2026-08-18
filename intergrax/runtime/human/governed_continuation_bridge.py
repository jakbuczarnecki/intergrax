# © Artur Czarnecki. All rights reserved.

"""Bridge governed continuation requests into canonical Nexus HITL pause lifecycle."""

from __future__ import annotations

from uuid import uuid4

from intergrax.contracts.agent_decision import AgentDecision, HumanRequest
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
    compose_continuation_agent_decision,
    compose_continuation_interrupt,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.task.task import Task

__all__ = [
    "apply_governed_continuation_pause",
    "bridge_governed_continuation_to_execution_result",
    "bridge_governed_continuation_to_governance",
    "compose_governed_continuation_from_enforcement",
]


def compose_governed_continuation_from_enforcement(
    request: CollaborativeWorkEnforcementRequest,
    *,
    decision: PolicyDecision,
    enforcement_operation_id: str,
    enforcement_authority_scope: str | None,
    requires_governed_continuation: bool,
    source_agent_id: str,
    source_step_id: str | None = None,
    reason: ContinuationReason | None = None,
) -> GovernedContinuationRequest | None:
    """Build a typed continuation request from enforcement evaluation — no pause."""
    if not requires_governed_continuation:
        return None

    side_effect = request.meaningful_side_effect_request
    if side_effect is None:
        return None

    resolved_reason = reason or ContinuationReason.COMPLIANCE
    resource_scope = request.resource_scope or enforcement_authority_scope

    return GovernedContinuationRequest(
        reason=resolved_reason,
        task_id=side_effect.task_id,
        run_id=side_effect.run_id,
        source_agent_id=source_agent_id,
        source_step_id=source_step_id,
        prompt=(
            f"Governed continuation required for operation {enforcement_operation_id}"
            f" ({decision.reason or decision.action.value})"
        ),
        operation_id=enforcement_operation_id,
        policy_rule_id=decision.policy_rule_id,
        resource_scope=resource_scope,
        policy_action=decision.action,
        side_effect_scope_id=side_effect.side_effect_scope_id,
        side_effect_scope_digest=side_effect.side_effect_scope_digest,
    )


def compose_continuation_human_request(
    request: GovernedContinuationRequest,
    *,
    request_id: str | None = None,
) -> HumanRequest:
    """Canonical HumanRequest with typed continuation correlation."""
    return HumanRequest(
        request_id=request_id or f"hr_{uuid4().hex[:12]}",
        prompt=request.prompt,
        options=[
            HumanResponseVerdict.APPROVE.value,
            HumanResponseVerdict.REJECT.value,
            HumanResponseVerdict.ESCALATE.value,
        ],
        governed_continuation=request.to_correlation(),
    )


def bridge_governed_continuation_to_governance(
    request: GovernedContinuationRequest,
) -> GovernanceResolution:
    """Translate continuation request into canonical interrupt + human request."""
    interrupt = compose_continuation_interrupt(request)
    agent_decision = compose_continuation_agent_decision(request, interrupt)
    human_request = compose_continuation_human_request(request)
    policy_decision = PolicyDecision(
        action=PolicyAction.REQUIRE_HUMAN,
        reason="governed_continuation_required",
        policy_rule_id=request.policy_rule_id,
    )
    return GovernanceResolution(
        policy_decision=policy_decision,
        agent_decision=agent_decision,
        interrupt=interrupt,
        human_request=human_request,
    )


def bridge_governed_continuation_to_execution_result(
    request: GovernedContinuationRequest,
    *,
    agent_id: str | None = None,
) -> AgentExecutionResult:
    """Agent execution surface for Nexus graph pause composition."""
    resolution = bridge_governed_continuation_to_governance(request)
    return AgentExecutionResult(
        agent_id=agent_id or request.source_agent_id,
        run_id=request.run_id,
        status=AgentExecutionStatus.NEEDS_INPUT,
        summary=request.prompt,
        human_request=resolution.human_request,
        execution_interrupt=resolution.interrupt,
        agent_decision=resolution.agent_decision,
        policy_rule_id=request.policy_rule_id,
    )


def apply_governed_continuation_pause(
    task: Task,
    request: GovernedContinuationRequest,
) -> Task:
    """Enter canonical WAITING_FOR_HUMAN via HumanPauseCoordinator."""
    execution = bridge_governed_continuation_to_execution_result(request)
    return HumanPauseCoordinator.apply_pause(task, execution)
