# © Artur Czarnecki. All rights reserved.

"""Task-governance-backed Agent Governance grant lifecycle adapter (UCA-6C-R6-R5)."""

from __future__ import annotations

from collections.abc import Callable
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
from intergrax.runtime.long_running.checkpoint_builder import (
    build_task_checkpoint,
    resolve_task_runtime_checkpoint,
)
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.task.task import Task

GrantLifecycleMutator = Callable[
    [AgentGovernanceGrantLifecycleRecord | None],
    AgentGovernanceGrantLifecycleRecord,
]


class TaskAgentGovernanceGrantLifecycleAdapter(AgentGovernanceGrantLifecyclePort):
    """Canonical Task governance grant lifecycle via ``TaskCheckpointPersistence`` CAS."""

    def __init__(
        self,
        *,
        task: Task,
        checkpoint_store: TaskCheckpointPersistence,
    ) -> None:
        self._task = task
        self._checkpoint_store = checkpoint_store

    def load(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
    ) -> AgentGovernanceGrantLifecycleRecord | None:
        if not self._tenant_task_matches(task_id=task_id, tenant_id=tenant_id):
            return None
        self._sync_from_canonical_checkpoint_if_present()
        return self._task.runtime.governance.agent_governance_human_approval_grant

    def persist_available_grant(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        grant: AgentGovernanceHumanApprovalGrant,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        def mutator(
            record: AgentGovernanceGrantLifecycleRecord | None,
        ) -> AgentGovernanceGrantLifecycleRecord:
            if record is not None and record.lifecycle_state in {
                AgentGovernanceGrantLifecycleState.RESERVED,
                AgentGovernanceGrantLifecycleState.APPLIED,
            }:
                raise ValueError("active grant present")
            return AgentGovernanceGrantLifecycleRecord(
                grant=grant,
                lifecycle_state=AgentGovernanceGrantLifecycleState.AVAILABLE,
                lifecycle_revision=record.lifecycle_revision + 1 if record else 1,
                reservation=None,
            )

        return self._mutate(
            task_id=task_id,
            tenant_id=tenant_id,
            expected_lifecycle_revision=expected_lifecycle_revision,
            mutator=mutator,
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
        if not owner_id.strip():
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.INVALID_STATE,
            )
        if lease_expires_at.tzinfo is None or lease_expires_at <= datetime.now(
            timezone.utc
        ):
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.INVALID_STATE,
            )

        def mutator(
            record: AgentGovernanceGrantLifecycleRecord | None,
        ) -> AgentGovernanceGrantLifecycleRecord:
            if record is None:
                raise ValueError("grant missing")
            if record.grant.grant_id != grant_id:
                raise ValueError("grant id mismatch")
            if (
                record.lifecycle_state
                is not AgentGovernanceGrantLifecycleState.AVAILABLE
            ):
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
        if not owner_id.strip():
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.INVALID_STATE,
            )
        if lease_expires_at.tzinfo is None or lease_expires_at <= datetime.now(
            timezone.utc
        ):
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.INVALID_STATE,
            )

        def mutator(
            record: AgentGovernanceGrantLifecycleRecord | None,
        ) -> AgentGovernanceGrantLifecycleRecord:
            if record is None or record.reservation is None:
                raise ValueError("reservation missing")
            if (
                record.lifecycle_state
                is not AgentGovernanceGrantLifecycleState.RESERVED
            ):
                raise ValueError("invalid state for reclaim")
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
        def mutator(
            record: AgentGovernanceGrantLifecycleRecord | None,
        ) -> AgentGovernanceGrantLifecycleRecord:
            if record is None or record.reservation is None:
                raise ValueError("reservation missing")
            if (
                record.lifecycle_state
                is not AgentGovernanceGrantLifecycleState.RESERVED
            ):
                raise ValueError("invalid state for mark applied")
            ownership = record.reservation.ownership
            if ownership.owner_id != owner_id or ownership.fence != fence:
                raise ValueError("stale reservation owner")
            if ownership.lease_expires_at <= datetime.now(timezone.utc):
                raise ValueError("reservation lease expired")
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
        _ = reason

        def mutator(
            record: AgentGovernanceGrantLifecycleRecord | None,
        ) -> AgentGovernanceGrantLifecycleRecord:
            if record is None:
                raise ValueError("grant missing")
            if record.lifecycle_state is AgentGovernanceGrantLifecycleState.TERMINAL:
                raise ValueError("already terminal")
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

    def _tenant_task_matches(self, *, task_id: TaskId, tenant_id: str) -> bool:
        if str(self._task.task_id) != str(task_id):
            return False
        return self._task.tenant_id == tenant_id

    def _canonical_checkpoint_revision(self) -> int | None:
        latest = self._checkpoint_store.get_latest(
            str(self._task.task_id),
            self._task.tenant_id,
        )
        if latest is not None and latest.revision is not None:
            return latest.revision
        return None

    def _apply_checkpoint_to_task(self, checkpoint: TaskCheckpoint) -> None:
        restored = Task.model_validate(checkpoint.task_snapshot)
        self._task.runtime.governance = restored.runtime.governance
        self._task.runtime.orchestration.checkpoint_id = checkpoint.checkpoint_id
        self._task.runtime.orchestration.checkpoint_revision = checkpoint.revision
        self._task.runtime.orchestration.resume_token = checkpoint.resume_token
        self._task.sync_metadata()

    def _sync_from_canonical_checkpoint_if_present(self) -> None:
        latest = self._checkpoint_store.get_latest(
            str(self._task.task_id),
            self._task.tenant_id,
        )
        if latest is not None:
            self._apply_checkpoint_to_task(latest)

    def _reload_canonical_task_state(
        self,
    ) -> AgentGovernanceGrantLifecycleRecord | None:
        self._sync_from_canonical_checkpoint_if_present()
        return self._task.runtime.governance.agent_governance_human_approval_grant

    def _persist_through_checkpoint(
        self,
        updated_record: AgentGovernanceGrantLifecycleRecord,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        draft = self._task.model_copy(deep=True)
        draft.runtime.governance.agent_governance_human_approval_grant = updated_record
        draft.sync_metadata()
        runtime = resolve_task_runtime_checkpoint(self._task)
        checkpoint = build_task_checkpoint(
            draft,
            progress_message=self._task.runtime.orchestration.progress_message,
            resume_token=self._task.runtime.orchestration.resume_token,
            runtime=runtime,
        )
        expected_checkpoint_revision = self._canonical_checkpoint_revision()
        try:
            saved = self._checkpoint_store.save(
                checkpoint,
                expected_revision=expected_checkpoint_revision,
            )
        except StaleCheckpointWriteError:
            current = self._reload_canonical_task_state()
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.STALE_REVISION,
                record=current,
            )
        self._apply_checkpoint_to_task(saved)
        committed = self._task.runtime.governance.agent_governance_human_approval_grant
        return AgentGovernanceGrantLifecycleMutationResult(
            outcome=AgentGovernanceGrantLifecycleOutcome.APPLIED,
            record=committed,
        )

    def _mutate(
        self,
        *,
        task_id: TaskId,
        tenant_id: str,
        expected_lifecycle_revision: int,
        mutator: GrantLifecycleMutator,
    ) -> AgentGovernanceGrantLifecycleMutationResult:
        if not self._tenant_task_matches(task_id=task_id, tenant_id=tenant_id):
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.NOT_FOUND,
            )
        self._sync_from_canonical_checkpoint_if_present()
        current = self._task.runtime.governance.agent_governance_human_approval_grant
        current_revision = current.lifecycle_revision if current is not None else 0
        if current_revision != expected_lifecycle_revision:
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.STALE_REVISION,
                record=current,
            )
        try:
            updated = mutator(current)
        except ValueError:
            return AgentGovernanceGrantLifecycleMutationResult(
                outcome=AgentGovernanceGrantLifecycleOutcome.INVALID_STATE,
                record=current,
            )
        return self._persist_through_checkpoint(updated)


__all__ = [
    "GrantLifecycleMutator",
    "TaskAgentGovernanceGrantLifecycleAdapter",
]
