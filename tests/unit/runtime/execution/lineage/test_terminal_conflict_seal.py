# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    build_execution_lineage_attempt_scope,
)
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalConflictError,
    ExecutionTerminalOutcome,
    ExecutionTerminalRecord,
)
from intergrax.runtime.execution.lineage.persistence import (
    InMemoryExecutionLineagePersistence,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


class _ConflictTerminalService:
    def __init__(self, canonical: ExecutionTerminalRecord) -> None:
        self._canonical = canonical

    def commit_terminal_outcome(self, **kwargs: object) -> ExecutionTerminalRecord:
        raise ExecutionTerminalConflictError("already committed")

    def get_terminal_record(self, **kwargs: object) -> ExecutionTerminalRecord:
        return self._canonical


def test_terminal_conflict_reconciles_lineage_seal() -> None:
    persistence = InMemoryExecutionLineagePersistence()
    registry = AgentRegistry()
    loop = NexusLoop(registry, execution_lineage_persistence=persistence)
    task_id = mint_task_id()
    task = Task(
        task_id=task_id,
        tenant_id="tenant-a",
        user_id="user-a",
        message="terminal seal",
        context=TaskContext(capability="agent.demo"),
        state=TaskState.COMPLETED,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id=task.tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root)
    persistence.admit_root(scope, root, root)

    canonical = ExecutionTerminalRecord(
        tenant_id=task.tenant_id,
        task_id=task_id,
        run_id=run_id,
        outcome=ExecutionTerminalOutcome.COMPLETED,
        reason="completed",
        recorded_at_utc="2026-01-01T00:00:00Z",
    )
    object.__setattr__(loop, "_execution_terminal", _ConflictTerminalService(canonical))

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=root,
    )
    try:
        resolution = loop._commit_durable_terminal_authority(task)  # noqa: SLF001
    finally:
        reset_active_execution_identity(token)

    assert resolution.canonical_record == canonical
    seal = persistence.read_seal(scope)
    assert seal is not None
    assert seal.closure_kind is ExecutionLineageAttemptClosureKind.COMPLETED
