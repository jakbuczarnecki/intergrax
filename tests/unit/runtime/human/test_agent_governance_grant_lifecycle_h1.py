# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import List, Optional

import pytest

from intergrax.contracts.agent_governance_grant_lifecycle_port import (
    AgentGovernanceGrantLifecycleOutcome,
)
from intergrax.contracts.agent_governance_hitl import (
    AgentGovernanceGrantLifecycleRecord,
    AgentGovernanceGrantLifecycleState,
    AgentGovernanceHumanApprovalGrant,
    LogicalInvocationFingerprint,
    mint_agent_governance_invocation_scope_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.long_running.checkpoint_builder import build_task_checkpoint
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.human.agent_governance_grant_lifecycle import (
    TaskAgentGovernanceGrantLifecycleAdapter,
)
from intergrax.runtime.task.task import Task, TaskState

pytestmark = pytest.mark.unit


class _MemoryTaskCheckpointStore(TaskCheckpointPersistence):
    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], TaskCheckpoint] = {}
        self._sequence = 0

    def list_for_task(self, task_id: str, tenant_id: str) -> List[TaskCheckpoint]:
        latest = self._latest.get((task_id, tenant_id))
        return [latest] if latest else []

    def get_latest(self, task_id: str, tenant_id: str) -> Optional[TaskCheckpoint]:
        return self._latest.get((task_id, tenant_id))

    def get_by_token(
        self,
        task_id: str,
        tenant_id: str,
        resume_token: str,
    ) -> Optional[TaskCheckpoint]:
        latest = self.get_latest(task_id, tenant_id)
        if latest is not None and latest.resume_token == resume_token:
            return latest
        return None

    def list_paused(self) -> List[TaskCheckpoint]:
        return []

    def save(
        self,
        checkpoint: TaskCheckpoint,
        *,
        expected_revision: int | None = None,
    ) -> TaskCheckpoint:
        stream_key = (checkpoint.task_id, checkpoint.tenant_id)
        current = self._latest.get(stream_key)
        current_revision = current.revision if current is not None else None
        if current_revision is None:
            if expected_revision is not None:
                raise StaleCheckpointWriteError(
                    task_id=checkpoint.task_id,
                    tenant_id=checkpoint.tenant_id,
                    expected_revision=expected_revision,
                    actual_revision=None,
                )
            next_revision = 1
        else:
            if expected_revision != current_revision:
                raise StaleCheckpointWriteError(
                    task_id=checkpoint.task_id,
                    tenant_id=checkpoint.tenant_id,
                    expected_revision=expected_revision,
                    actual_revision=current_revision,
                )
            next_revision = current_revision + 1
        self._sequence += 1
        stored = checkpoint.model_copy(
            update={"revision": next_revision, "store_sequence": self._sequence},
        )
        self._latest[stream_key] = stored
        return stored

    def cancel(self, schedule_id: str) -> None:
        _ = schedule_id

    def schedule(self, entry: object) -> object:
        return entry

    def list_due(self, *, before_utc_iso: str, limit: int = 100) -> list:
        _ = before_utc_iso
        _ = limit
        return []

    def claim_due(
        self,
        *,
        before_utc_iso: str,
        owner_id: str,
        lease_seconds: int,
        limit: int = 100,
    ) -> list:
        _ = before_utc_iso
        _ = owner_id
        _ = lease_seconds
        _ = limit
        return []

    def complete_claim(self, claim: object) -> None:
        _ = claim

    def schedule_resume(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError

    def list_due_resumes(self, *args: object, **kwargs: object) -> list:
        return []

    def mark_resume_completed(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError


def _task(*, tenant_id: str = "tenant-a") -> Task:
    return Task(
        tenant_id=tenant_id,
        user_id="user-a",
        message="governance test",
        state=TaskState.WAITING_FOR_HUMAN,
    )


def _grant(task: Task) -> AgentGovernanceHumanApprovalGrant:
    scope = mint_agent_governance_invocation_scope_id()
    fingerprint = LogicalInvocationFingerprint(digest="sha256:" + ("a" * 64))
    return AgentGovernanceHumanApprovalGrant(
        grant_id="grant_1",
        agent_governance_invocation_scope_id=scope,
        pending_generation=1,
        logical_invocation_fingerprint=fingerprint,
        task_id=task.task_id,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        tenant_id=task.tenant_id,
        agent_id="agent-a",
        tool_id="tool-a",
        step_id="step-a",
        idempotency_key="idem",
        human_request_id="hr_1",
        pause_id="pause_1",
        approved_at="2026-09-23T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
    )


def _seed_checkpoint(task: Task, store: _MemoryTaskCheckpointStore) -> None:
    checkpoint = build_task_checkpoint(task)
    store.save(checkpoint, expected_revision=None)
    latest = store.get_latest(str(task.task_id), task.tenant_id)
    assert latest is not None
    task.runtime.orchestration.checkpoint_revision = latest.revision


def test_checkpoint_cas_second_writer_stale() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    adapter_a = TaskAgentGovernanceGrantLifecycleAdapter(
        task=task, checkpoint_store=store
    )
    task_b = Task.model_validate(task.model_dump(mode="json"))
    adapter_b = TaskAgentGovernanceGrantLifecycleAdapter(
        task=task_b, checkpoint_store=store
    )
    grant = _grant(task)
    first = adapter_a.persist_available_grant(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=0,
        grant=grant,
    )
    assert first.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED
    second = adapter_b.persist_available_grant(
        task_id=task_b.task_id,
        tenant_id=task_b.tenant_id,
        expected_lifecycle_revision=0,
        grant=grant,
    )
    assert second.outcome is AgentGovernanceGrantLifecycleOutcome.STALE_REVISION


def test_wrong_tenant_load_and_mutation_fail_closed() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    adapter = TaskAgentGovernanceGrantLifecycleAdapter(
        task=task, checkpoint_store=store
    )
    assert adapter.load(task_id=task.task_id, tenant_id="other-tenant") is None
    result = adapter.persist_available_grant(
        task_id=task.task_id,
        tenant_id="other-tenant",
        expected_lifecycle_revision=0,
        grant=_grant(task),
    )
    assert result.outcome is AgentGovernanceGrantLifecycleOutcome.NOT_FOUND
    assert task.runtime.governance.agent_governance_human_approval_grant is None


def test_lifecycle_transitions_and_guards() -> None:
    task = _task()
    store = _MemoryTaskCheckpointStore()
    _seed_checkpoint(task, store)
    adapter = TaskAgentGovernanceGrantLifecycleAdapter(
        task=task, checkpoint_store=store
    )
    grant = _grant(task)
    persisted = adapter.persist_available_grant(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=0,
        grant=grant,
    )
    assert persisted.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED
    assert persisted.record is not None
    rev = persisted.record.lifecycle_revision
    lease = datetime.now(UTC) + timedelta(minutes=5)
    reserved = adapter.reserve(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=rev,
        grant_id=grant.grant_id,
        logical_invocation_fingerprint=grant.logical_invocation_fingerprint,
        pause_generation=1,
        agent_governance_invocation_scope_id=grant.agent_governance_invocation_scope_id,
        task_id_link=grant.task_id,
        run_id=grant.run_id,
        attempt_id=grant.attempt_id,
        execution_id=grant.execution_id,
        policy_provenance_digest=None,
        owner_id="host-a",
        lease_expires_at=lease,
    )
    assert reserved.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED
    assert reserved.record is not None
    invalid_applied = adapter.mark_applied(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=rev,
        owner_id="host-a",
        fence=1,
    )
    assert (
        invalid_applied.outcome is AgentGovernanceGrantLifecycleOutcome.STALE_REVISION
    )
    applied = adapter.mark_applied(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=reserved.record.lifecycle_revision,
        owner_id="host-a",
        fence=1,
    )
    assert applied.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED
    terminal = adapter.terminalize(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=applied.record.lifecycle_revision,
        reason="consumed",
    )
    assert terminal.outcome is AgentGovernanceGrantLifecycleOutcome.APPLIED
    again = adapter.terminalize(
        task_id=task.task_id,
        tenant_id=task.tenant_id,
        expected_lifecycle_revision=terminal.record.lifecycle_revision,
        reason="noop",
    )
    assert again.outcome is AgentGovernanceGrantLifecycleOutcome.INVALID_STATE
