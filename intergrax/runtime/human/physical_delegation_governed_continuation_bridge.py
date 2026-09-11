# © Artur Czarnecki. All rights reserved.

"""Bridge physical delegation governed continuation into canonical Nexus HITL pause."""

from __future__ import annotations

from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
)
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationGovernedContinuation,
    physical_delegation_governed_continuation_digest,
    physical_delegation_operation_id,
)
from intergrax.runtime.human.governed_continuation_bridge import (
    apply_governed_continuation_pause,
)
from intergrax.runtime.task.task import Task

__all__ = [
    "apply_physical_delegation_governed_continuation_pause",
    "project_physical_delegation_to_governed_continuation_request",
]


def project_physical_delegation_to_governed_continuation_request(
    continuation: PhysicalDelegationGovernedContinuation,
    *,
    source_agent_id: str,
    run_id: str,
    source_step_id: str | None = None,
) -> GovernedContinuationRequest:
    """Project exact physical continuation into generic governed continuation request."""
    evidence = continuation.governance_result.evidence
    decision = continuation.governance_result.decision
    continuation_digest = physical_delegation_governed_continuation_digest(continuation)
    return GovernedContinuationRequest(
        reason=ContinuationReason.COMPLIANCE,
        task_id=continuation.task_scope_id,
        run_id=run_id,
        source_agent_id=source_agent_id,
        source_step_id=source_step_id,
        prompt=(
            "Physical delegation "
            f"{continuation.delegation_id} requires human approval for specialist "
            f"{continuation.selected_identity.distribution_package_id}"
        ),
        operation_id=physical_delegation_operation_id(continuation.delegation_id),
        policy_rule_id=decision.policy_rule_id or evidence.policy_rule_id,
        policy_action=decision.action,
        correlation={
            "physical_delegation.continuation_digest": continuation_digest,
        },
    )


def apply_physical_delegation_governed_continuation_pause(
    task: Task,
    continuation: PhysicalDelegationGovernedContinuation,
    *,
    source_agent_id: str,
    run_id: str,
    source_step_id: str | None = None,
) -> Task:
    """Enter canonical WAITING_FOR_HUMAN with typed physical continuation bound on task."""
    gov = task.runtime.governance
    gov.physical_delegation_governed_continuation = continuation
    gov.physical_delegation_continuation_grant = None
    request = project_physical_delegation_to_governed_continuation_request(
        continuation,
        source_agent_id=source_agent_id,
        run_id=run_id,
        source_step_id=source_step_id,
    )
    return apply_governed_continuation_pause(task, request)
