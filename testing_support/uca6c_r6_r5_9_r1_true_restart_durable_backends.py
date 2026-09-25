# © Artur Czarnecki. All rights reserved.

"""Shared durable backends for UCA-6C-R6-R5.9-R1 true restart qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.autonomous_work.document_store_worker_recovery_obstacle_capability_need_repository import (
    DocumentStoreWorkerRecoveryObstacleCapabilityNeedRepository,
)
from intergrax.runtime.persistence.sqlite_composition import (
    create_sqlite_idempotency_store,
)
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    reconnect_execution_engine_continuation_dependencies,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.durable_state_file import (
    ExecutionContinuationDurableStateFilePersistence,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
)
from intergrax.runtime.execution.document_store_execution_terminal_outcome_by_execution_id import (
    DocumentStoreExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from testing_support.uca6c_process_restart_durable_document_store import (
    ProcessRestartQualificationDocumentStore,
)


@dataclass
class Uca6cTrueRestartDurableBackends:
    """External persistence simulated for Host A/B/C — only these objects are shared."""

    document_store: ConditionalDocumentStore
    continuation_backing: ExecutionContinuationDurableBacking
    continuation_state_persistence: ExecutionContinuationDurableStateFilePersistence
    checkpoint_db_path: Path
    idempotency_db_path: Path
    aw_schema_name: str | None = None

    @classmethod
    def create(cls, tmp_path: Path) -> Uca6cTrueRestartDurableBackends:
        root = tmp_path / "uca6c-r59-r1-restart"
        root.mkdir(parents=True, exist_ok=True)
        continuation_path = root / "execution_continuation_durable_state.json"
        return cls(
            document_store=ProcessRestartQualificationDocumentStore(),
            continuation_backing=ExecutionContinuationDurableBacking(),
            continuation_state_persistence=ExecutionContinuationDurableStateFilePersistence(
                continuation_path,
            ),
            checkpoint_db_path=root / "task_checkpoints.db",
            idempotency_db_path=root / "tool_idempotency.db",
        )

    @classmethod
    def create_with_document_store(
        cls,
        tmp_path: Path,
        document_store: ConditionalDocumentStore,
    ) -> Uca6cTrueRestartDurableBackends:
        backends = cls.create(tmp_path)
        backends.document_store = document_store
        return backends

    def fresh_terminal_outcome_store(
        self,
    ) -> DocumentStoreExecutionTerminalOutcomeByExecutionIdStore:
        return DocumentStoreExecutionTerminalOutcomeByExecutionIdStore(
            self.document_store,
        )

    def fresh_document_need_repository(
        self,
    ) -> DocumentStoreWorkerRecoveryObstacleCapabilityNeedRepository:
        return DocumentStoreWorkerRecoveryObstacleCapabilityNeedRepository(
            self.document_store,
        )

    def fresh_checkpoint_store(self) -> SQLiteTaskCheckpointStore:
        return SQLiteTaskCheckpointStore(db_path=self.checkpoint_db_path)

    def fresh_idempotency_store(self) -> object:
        return create_sqlite_idempotency_store(db_path=self.idempotency_db_path)

    def host_a_continuation_dependencies(
        self,
    ) -> ExecutionEngineContinuationDependencies:
        store = backing_execution_continuation_state_store(self.continuation_backing)
        return wire_execution_engine_continuation_dependencies(state_store=store)

    def seal_continuation_for_process_death(self) -> None:
        self.continuation_state_persistence.persist_from_backing(
            self.continuation_backing,
        )
        assert self.continuation_state_persistence.path.is_file()
        self.continuation_backing = ExecutionContinuationDurableBacking()

    def host_b_continuation_dependencies(
        self,
    ) -> ExecutionEngineContinuationDependencies:
        store = self.continuation_state_persistence.load_state_store()
        return reconnect_execution_engine_continuation_dependencies(state_store=store)


__all__ = ["Uca6cTrueRestartDurableBackends"]
