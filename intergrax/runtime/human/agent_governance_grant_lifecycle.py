# © Artur Czarnecki. All rights reserved.

"""Task-governance-backed Agent Governance grant lifecycle adapter (UCA-6C-R6-R5)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.agent_governance_grant_lifecycle_port import (
    AgentGovernanceGrantLifecycleMutationResult,
    AgentGovernanceGrantLifecycleOutcome,
    AgentGovernanceGrantLifecyclePort,
)
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleRecord,
    AgentGovernanceGrantLifecycleState,
    AgentGovernanceGrantReservation,
    AgentGovernanceHumanApprovalGrant,
    LogicalInvocationFingerprint,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.lease_claim import LeaseOwnership
from intergrax.runtime.task.task import Task


class TaskAgentGovernanceGrantLifecycleAdapter(AgentGovernanceGrantLifecyclePort):
    """Default adapter — canonical TaskGovernanceState fields only."""

    def __init__(self, *, task: Task) -> None:
        self._task = task

    def load(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
    ) -> AgentGovernanceGrantLifecycleRecord | None:
        if str(self._task.task_id) != str(task_id):
            return None
        return self._task.runtime.governance.agent_governance_human_approval_grant

    def persist_available_grant(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        grant: AgentGovernanceHumanApprovalGrant,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        return self._mutate(
            task_id=task_id,
            tenant_id=tenant_id,
            expected_lifecycle_revision=expected_lifecycle_revision,
            mutator=lambda record: AgentGovernanceGrantLifecycleRecord(
                grant=grant,
                lifecycle_state=AgentGovernanceGrantLifecycleState.AVAILABLE,
                lifecycle_revision=record.lifecycle_revision + 1 if record else 1,
                reservation=None,
            ),
        )

    def reserve(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        grant_id: str,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
        pause_generation: int,
        agent_governance_invocation_scope_id: str,
        task_id_link: TaskId,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
        policy_provenance_digest: str | None,
        owner_id: str,
        lease_expires_at: datetime,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        def mutator(record: AgentGovernanceGrantLifecycleRecord | None) -> AgentGovernanceGrantLifecycleRecord:
            if record is None:
                raise ValueError("grant missing")
            if record.grant.grant_id != grant_id:
                raise ValueError("grant id mismatch")
            if record.lifecycle_state is not AgentGovernanceGrantLifecycleState.AVAILABLE:
                raise ValueError("invalid state for reserve")
            reservation = AgentGovernanceGrantReservation(
                logical_invocation_fingerprint=logical_invocation_fingerprint,
                pause_generation=pause_generation,
                agent_governance_invocation_scope_id=agent_governance_invocation_scope_id,
                task_id=task_id_link,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                policy_provenance_digest=policy_provenance_digest,
                ownership=LeaseOwnership(
                    owner_id=owner_id,
                    lease_expires_at=lease_expires_at,
                    fence=1,
                ),
            )
            return AgentGovernanceGrantLifecycleRecord(
                grant=record.grant,
                lifecycle_state=AgentGovernanceGrantLifecycleState.RESERVED,
                lifecycle_revision=record.lifecycle_revision + 1,
                reservation=reservation,
            )

        return self._mutate(
            task_id=task_id,
            tenant_id=tenant_id,
            expected_lifecycle_revision=expected_lifecycle_revision,
            mutator=mutator,
        )

    def reclaim_reservation(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
        expected_fence: int,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        def mutator(record: AgentGovernanceGrantLifecycleRecord | None) -> AgentGovernanceGrantLifecycleRecord:
            if record is None or record.reservation is None:
                raise ValueError("reservation missing")
            ownership = record.reservation.ownership
            if ownership.fence != expected_fence:
                raise ValueError("stale fence")
            if ownership.lease_expires_at > datetime.now(timezone.utc):
                raise ValueError("lease not expired")
            new_fence = ownership.fence + 1
            updated_reservation = record.reservation.model_copy(
                update={
                    "ownership": LeaseOwnership(
                        owner_id=owner_id,
                        lease_expires_at=lease_expires_at,
                        fence=new_fence,
                    )
                }
            )
            return record.model_copy(
                update={
                    "lifecycle_revision": record.lifecycle_revision + 1,
                    "reservation": updated_reservation,
                }
            )

        return self._mutate(
            task_id=task_id,
            tenant_id=tenant_id,
            expected_lifecycle_revision=expected_lifecycle_revision,
            mutator=mutator,
        )

    def mark_applied(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        owner_id: str,
        fence: int,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        def mutator(record: AgentGovernanceGrantLifecycleRecord | None) -> AgentGovernanceGrantLifecycleRecord:
            if record is None or record.reservation is None:
                raise ValueError("reservation missing")
            ownership = record.reservation.ownership
            if ownership.owner_id != owner_id or ownership.fence != fence:
                raise ValueError("stale reservation owner")
            return AgentGovernanceGrantLifecycleRecord(
                grant=record.grant,
                lifecycle_state=AgentGovernanceGrantLifecycleState.APPLIED,
                lifecycle_revision=record.lifecycle_revision + 1,
                reservation=None,
            )

        return self._mutate(
            task_id=task_id,
            tenant_id=tenant_id,
            expected_lifecycle_revision=expected_lifecycle_revision,
            mutator=mutator,
        )

    def terminalize(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        reason: str,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        def mutator(record: AgentGovernanceGrantLifecycleRecord | None) -> AgentGovernanceGrantLifecycleRecord:
            if record is None:
                raise ValueError("grant missing")
            return AgentGovernanceGrantLifecycleRecord(
                grant=record.grant,
                lifecycle_state=AgentGovernanceGrantLifecycleState.TERMINAL,
                lifecycle_revision=record.lifecycle_revision + 1,
                reservation=None,
            )

        return self._mutate(
            task_id=task_id,
            tenant_id=tenant_id,
            expected_lifecycle_revision=expected_lifecycle_revision,
            mutator=mutator,
        )

    def _mutate(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        mutator: object,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        if str(self._task.task_id) != str(task_id):
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.NOT_FOUND,
            )
        current = self._task.runtime.governance.agent_governance_human_approval_grant
        current_revision = current.lifecycle_revision if current is not None else 0
        if current_revision != expected_lifecycle_revision:
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.STALE_REVISION,
                record=current,
            )
        try:
            updated = mutator(current)  # type: ignore[operator]
        except ValueError:
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.INVALID_STATE,
                record=current,
            )
        self._task.runtime.governance.agent_governance_human_approval_grant = updated
        self._task.sync_metadata()
        return AgentGovernanceGrantLifecycleMutationResult(
            outcome=AgentGovernanceGrantLifecycleOutcome.APPLIED,
            record=updated,
        )

__all__ = ["TaskAgentGovernanceGrantLifecycleAdapter"]
