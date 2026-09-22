# © Artur Czarnecki. All rights reserved.

"""Compose governed continuation requests from declarative HITL pause artifacts."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
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


__all__ = ["compose_governed_continuation_from_declarative_hitl_pause"]
