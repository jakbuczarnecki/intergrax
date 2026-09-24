# © Artur Czarnecki. All rights reserved.

"""Task-checkpoint-backed Agent Governance pause projection (UCA-6C-R6-R5.5-H1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.agent_decision import HumanRequest
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceHumanApprovalPending,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    build_task_checkpoint,
    resolve_task_runtime_checkpoint,
)
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.task.task import Task, TaskState


class AgentGovernancePauseProjectionOutcome(StrEnum):
    APPLIED = "applied"
    STALE_REVISION = "stale_revision"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class AgentGovernancePauseProjectionResult:
    outcome: AgentGovernancePauseProjectionOutcome
    pending: AgentGovernanceHumanApprovalPending | None = None


@dataclass(frozen=True, slots=True)
class _PauseProjectionSnapshot:
    pending: AgentGovernanceHumanApprovalPending | None
    checkpoint_revision: int | None


class TaskAgentGovernancePauseProjectionAdapter:
    """Canonical Agent Governance pending / HumanRequest / WAITING via checkpoint CAS."""

    def __init__(
        self,
        *,
        task: Task,
        checkpoint_store: TaskCheckpointPersistence,
    ) -> None:
        self._task = task
        self._checkpoint_store = checkpoint_store

    def clear_pending_durably(
        self,
    ) -> AgentGovernancePauseProjectionResult:
        snapshot = self._load_canonical_pause_snapshot()
        if snapshot.pending is None:
            return AgentGovernancePauseProjectionResult(
                outcome=AgentGovernancePauseProjectionOutcome.APPLIED,
                pending=None,
            )
        return self._clear_pending_through_checkpoint(
            expected_checkpoint_revision=snapshot.checkpoint_revision,
        )

    def persist_pause_projection(
        self,
        *,
        pending: AgentGovernanceHumanApprovalPending,
        human_request: HumanRequest,
    ) -> AgentGovernancePauseProjectionResult:
        snapshot = self._load_canonical_pause_snapshot()
        current = snapshot.pending
        if current is not None:
            if (
                current.agent_governance_invocation_scope_id
                == pending.agent_governance_invocation_scope_id
                and current.pause_id == pending.pause_id
            ):
                return AgentGovernancePauseProjectionResult(
                    outcome=AgentGovernancePauseProjectionOutcome.APPLIED,
                    pending=current,
                )
            return AgentGovernancePauseProjectionResult(
                outcome=AgentGovernancePauseProjectionOutcome.CONFLICT,
                pending=current,
            )
        return self._persist_through_checkpoint(
            pending=pending,
            human_request=human_request,
            expected_checkpoint_revision=snapshot.checkpoint_revision,
        )

    def _apply_checkpoint_to_task(self, checkpoint: TaskCheckpoint) -> None:
        restored = Task.model_validate(checkpoint.task_snapshot)
        self._task.runtime.governance = restored.runtime.governance
        self._task.state = restored.state
        self._task.runtime.orchestration.checkpoint_id = checkpoint.checkpoint_id
        self._task.runtime.orchestration.checkpoint_revision = checkpoint.revision
        self._task.runtime.orchestration.resume_token = checkpoint.resume_token
        self._task.sync_metadata()

    def _load_canonical_pause_snapshot(self) -> _PauseProjectionSnapshot:
        latest = self._checkpoint_store.get_latest(
            str(self._task.task_id),
            self._task.tenant_id,
        )
        if latest is None:
            return _PauseProjectionSnapshot(
                pending=self._task.runtime.governance.agent_governance_hitl_pending,
                checkpoint_revision=None,
            )
        self._apply_checkpoint_to_task(latest)
        return _PauseProjectionSnapshot(
            pending=self._task.runtime.governance.agent_governance_hitl_pending,
            checkpoint_revision=latest.revision,
        )

    def _clear_pending_through_checkpoint(
        self,
        *,
        expected_checkpoint_revision: int | None,
    ) -> AgentGovernancePauseProjectionResult:
        draft = self._task.model_copy(deep=True)
        draft.runtime.governance.agent_governance_hitl_pending = None
        draft.sync_metadata()
        runtime = resolve_task_runtime_checkpoint(self._task)
        checkpoint = build_task_checkpoint(
            draft,
            progress_message=self._task.runtime.orchestration.progress_message,
            resume_token=self._task.runtime.orchestration.resume_token,
            runtime=runtime,
        )
        try:
            saved = self._checkpoint_store.save(
                checkpoint,
                expected_revision=expected_checkpoint_revision,
            )
        except StaleCheckpointWriteError:
            snapshot = self._load_canonical_pause_snapshot()
            return AgentGovernancePauseProjectionResult(
                outcome=AgentGovernancePauseProjectionOutcome.STALE_REVISION,
                pending=snapshot.pending,
            )
        self._apply_checkpoint_to_task(saved)
        return AgentGovernancePauseProjectionResult(
            outcome=AgentGovernancePauseProjectionOutcome.APPLIED,
            pending=self._task.runtime.governance.agent_governance_hitl_pending,
        )

    def _persist_through_checkpoint(
        self,
        *,
        pending: AgentGovernanceHumanApprovalPending,
        human_request: HumanRequest,
        expected_checkpoint_revision: int | None,
    ) -> AgentGovernancePauseProjectionResult:
        draft = self._task.model_copy(deep=True)
        draft.runtime.governance.agent_governance_hitl_pending = pending
        canonical_human_request = human_request
        existing = self._task.runtime.governance.human_request
        if (
            canonical_human_request.governed_continuation is None
            and existing is not None
            and existing.governed_continuation is not None
        ):
            canonical_human_request = canonical_human_request.model_copy(
                update={"governed_continuation": existing.governed_continuation},
            )
        draft.runtime.governance.human_request = canonical_human_request
        draft.state = TaskState.WAITING_FOR_HUMAN
        draft.sync_metadata()
        runtime = resolve_task_runtime_checkpoint(self._task)
        checkpoint = build_task_checkpoint(
            draft,
            progress_message=self._task.runtime.orchestration.progress_message,
            resume_token=self._task.runtime.orchestration.resume_token,
            runtime=runtime,
        )
        try:
            saved = self._checkpoint_store.save(
                checkpoint,
                expected_revision=expected_checkpoint_revision,
            )
        except StaleCheckpointWriteError:
            snapshot = self._load_canonical_pause_snapshot()
            return AgentGovernancePauseProjectionResult(
                outcome=AgentGovernancePauseProjectionOutcome.STALE_REVISION,
                pending=snapshot.pending,
            )
        self._apply_checkpoint_to_task(saved)
        committed = self._task.runtime.governance.agent_governance_hitl_pending
        return AgentGovernancePauseProjectionResult(
            outcome=AgentGovernancePauseProjectionOutcome.APPLIED,
            pending=committed,
        )


__all__ = [
    "AgentGovernancePauseProjectionOutcome",
    "AgentGovernancePauseProjectionResult",
    "TaskAgentGovernancePauseProjectionAdapter",
]
