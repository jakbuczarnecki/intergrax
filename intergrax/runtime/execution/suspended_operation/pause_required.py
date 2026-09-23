# © Artur Czarnecki. All rights reserved.

"""Typed control-flow when durable HITL materialization completed."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.runtime.nexus.tools.agent_governance_approval_pause_bridge import (
    AgentGovernanceApprovalPauseRequired,
)
from intergrax.runtime.nexus.tools.declarative_policy_hitl_bridge import (
    DeclarativePolicyHitlPauseRequired,
)
from intergrax.runtime.nexus.tools.mse_governed_continuation_hitl_bridge import (
    GovernedContinuationHitlPauseRequired,
)


@dataclass(frozen=True, slots=True)
class ExecutionSuspendedWorkPauseRequired(RuntimeError):
    """Canonical durable HITL pause established — not execution failure."""

    governed_request: GovernedContinuationRequest
    descriptor: SuspendedExecutionOperationDescriptor
    declarative_pause: DeclarativePolicyHitlPauseRequired | None = None
    agent_governance_pause: AgentGovernanceApprovalPauseRequired | None = None
    mse_governed_pause: GovernedContinuationHitlPauseRequired | None = None

    def __post_init__(self) -> None:
        kinds = (
            self.declarative_pause is not None,
            self.agent_governance_pause is not None,
            self.mse_governed_pause is not None,
        )
        if sum(kinds) != 1:
            raise ValueError(
                "exactly one authority pause payload "
                "(declarative, agent_governance, or mse_governed) required",
            )

    @property
    def pause(self) -> DeclarativePolicyHitlPauseRequired:
        if self.declarative_pause is None:
            raise AttributeError("no declarative pause on this execution suspension")
        return self.declarative_pause

    def __str__(self) -> str:
        return (
            "Execution suspended pending human approval "
            f"(continuation_id={self.governed_request.continuation_request_id}, "
            f"scope={self.descriptor.invocation_scope_id})."
        )


__all__ = ["ExecutionSuspendedWorkPauseRequired"]
