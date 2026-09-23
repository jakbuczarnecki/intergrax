# © Artur Czarnecki. All rights reserved.

"""Per-resume consumption adapter wired into RuntimeState for re-entry (UCA-6C-R6-R5.6-H1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_governance_approval_consumption_port import (
    AgentGovernanceApprovalConsumptionError,
    AgentGovernanceApprovalConsumptionPort,
)
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleRecord,
)
from intergrax.contracts.agent_governance_verified_approval import (
    VerifiedAgentGovernanceHumanApproval,
)
from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.runtime.execution.suspended_operation.agent_governance_reentry_grant import (
    AgentGovernanceReentryGrantError,
    mark_agent_governance_grant_applied_after_governance,
)
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class SuspendedWorkAgentGovernanceApprovalConsumption(
    AgentGovernanceApprovalConsumptionPort,
):
    task: Task
    checkpoint_store: TaskCheckpointPersistence
    lifecycle_record: AgentGovernanceGrantLifecycleRecord
    claim_ownership: LeaseOwnership

    def mark_applied_after_governance_allow(
        self,
        verified: VerifiedAgentGovernanceHumanApproval,
    ) -> None:
        if verified.grant_id != self.lifecycle_record.grant.grant_id:
            raise AgentGovernanceApprovalConsumptionError("verified grant_id mismatch")
        try:
            mark_agent_governance_grant_applied_after_governance(
                task=self.task,
                checkpoint_store=self.checkpoint_store,
                lifecycle_record=self.lifecycle_record,
                claim_ownership=self.claim_ownership,
            )
        except AgentGovernanceReentryGrantError as exc:
            raise AgentGovernanceApprovalConsumptionError(str(exc)) from exc


__all__ = ["SuspendedWorkAgentGovernanceApprovalConsumption"]
