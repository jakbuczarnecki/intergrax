# © Artur Czarnecki. All rights reserved.

"""Typed control-flow when durable HITL materialization completed."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativePolicyHitlPauseRequired,
)


@dataclass(frozen=True, slots=True)
class ExecutionSuspendedWorkPauseRequired(RuntimeError):
    """Canonical durable HITL pause established — not execution failure."""

    pause: DeclarativePolicyHitlPauseRequired
    governed_request: GovernedContinuationRequest
    descriptor: SuspendedExecutionOperationDescriptor

    def __str__(self) -> str:
        return (
            "Execution suspended pending human approval "
            f"(continuation_id={self.governed_request.continuation_request_id}, "
            f"scope={self.descriptor.invocation_scope_id})."
        )


__all__ = ["ExecutionSuspendedWorkPauseRequired"]
