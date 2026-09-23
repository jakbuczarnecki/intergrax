# © Artur Czarnecki. All rights reserved.

"""Compose governed continuation requests from declarative HITL pause artifacts."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.runtime.nexus.tools.agent_governance_approval_pause_bridge import (
    AgentGovernanceApprovalPauseRequired,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativePolicyHitlPauseRequired,
)


def compose_governed_continuation_from_declarative_hitl_pause(
    pause: DeclarativePolicyHitlPauseRequired,
    *,
    identity: ExecutionContinuationIdentity,
) -> GovernedContinuationRequest:
    signal = pause.signal
    return GovernedContinuationRequest(
        reason=ContinuationReason.COMPLIANCE,
        task_id=identity.task_id,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        source_agent_id=signal.agent_id,
        source_step_id=signal.step_id,
        prompt=(
            f"Declarative policy requires human approval before executing tool "
            f"'{signal.tool_id}'."
        ),
        operation_id=signal.invocation_scope_id,
    )


def compose_governed_continuation_from_agent_governance_pause(
    pause: AgentGovernanceApprovalPauseRequired,
    *,
    identity: ExecutionContinuationIdentity,
    invocation_scope_id: str,
) -> GovernedContinuationRequest:
    signal = pause.signal
    scope_id = invocation_scope_id
    return GovernedContinuationRequest(
        reason=ContinuationReason.AGENT_RUNTIME_GOVERNANCE,
        task_id=identity.task_id,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        source_agent_id=signal.agent_id,
        source_step_id=signal.step_id,
        prompt=(
            f"Agent runtime governance requires human approval before executing tool "
            f"'{signal.tool_id}'."
        ),
        operation_id=scope_id,
    )


__all__ = [
    "compose_governed_continuation_from_agent_governance_pause",
    "compose_governed_continuation_from_declarative_hitl_pause",
]
