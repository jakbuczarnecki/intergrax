# © Artur Czarnecki. All rights reserved.

"""Bridge governed continuation requests into Task/Human projection (legacy graph path).

Canonical lifecycle authority is :class:`~intergrax.contracts.execution_continuation.ExecutionContinuationPort`.
Nexus is a private internal orchestration subsystem of the Execution Engine — not a public engine.
"""

from __future__ import annotations

from uuid import uuid4

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_execution_result import (
    AgentExecutionResult,
    AgentExecutionStatus,
)
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
    compose_continuation_agent_decision,
    compose_continuation_interrupt,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.interrupts.handler import GovernanceResolution
from intergrax.runtime.execution.active_execution_continuation_store import (
    peek_active_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalHitlContinuationCapabilityError,
    InternalOrchestrationContinuation,
    establish_canonical_hitl_pause,
    require_internal_hitl_continuation,
)
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
        attempt_id=side_effect.attempt_id,
        execution_id=side_effect.execution_id,
        source_agent_id=source_agent_id,
        source_step_id=source_step_id,
        prompt=(
            f"Governed continuation required for operation {enforcement_operation_id}"
            f" ({decision.reason or decision.action.value})"
        ),
        operation_id=enforcement_operation_id,
        policy_rule_id=decision.policy_rule_id,
        policy_bundle_id=decision.policy_bundle_id,
        policy_bundle_version=decision.policy_bundle_version,
        policy_bundle_digest=decision.policy_bundle_digest,
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
    """Agent execution surface for internal graph pause composition (projection path)."""
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


def _resolve_hitl_continuation_for_bridge(
    hitl_continuation: InternalOrchestrationContinuation | None,
) -> InternalOrchestrationContinuation:
    if hitl_continuation is not None:
        return require_internal_hitl_continuation(hitl_continuation)
    active_store = peek_active_execution_continuation_state_store()
    if active_store is None:
        raise InternalHitlContinuationCapabilityError(
            "canonical execution continuation required for governed HITL pause; "
            "inject InternalOrchestrationContinuation or bind active "
            "ExecutionContinuationStateStore (no silent in-memory downgrade)",
        )
    deps = wire_execution_engine_continuation_dependencies(state_store=active_store)
    return InternalOrchestrationContinuation(
        port=deps.continuation,
        lifecycle_driver=deps.lifecycle_driver,
    )


def apply_governed_continuation_pause(
    task: Task,
    request: GovernedContinuationRequest,
    *,
    hitl_continuation: InternalOrchestrationContinuation | None = None,
) -> Task:
    """Establish canonical WAITING_FOR_HUMAN and project onto Task/Human view."""
    capability = _resolve_hitl_continuation_for_bridge(hitl_continuation)
    resolution = bridge_governed_continuation_to_governance(request)
    human_request = resolution.human_request
    assert human_request is not None
    identity = ExecutionContinuationIdentity(
        task_id=request.task_id,
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        execution_id=request.execution_id,
    )
    pause_id = f"pause_{human_request.request_id}"
    establish_canonical_hitl_pause(
        task,
        identity=identity,
        continuation_id=request.continuation_request_id,
        reason=request.reason,
        pause_id=pause_id,
        human_request_id=human_request.request_id,
        capability=capability,
        governed_correlation=request.to_correlation(),
        human_prompt=human_request.prompt,
        execution_interrupt=resolution.interrupt,
    )
    task.sync_metadata()
    return task
