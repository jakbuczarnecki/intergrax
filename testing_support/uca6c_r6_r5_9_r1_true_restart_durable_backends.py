# © Artur Czarnecki. All rights reserved.

"""Shared durable backends for UCA-6C-R6-R5.9-R1 true restart qualification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from intergrax.autonomous_work.document_store_worker_recovery_obstacle_capability_need_repository import (
    DocumentStoreWorkerRecoveryObstacleCapabilityNeedRepository,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    create_sqlite_idempotency_store,
)
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    reconnect_execution_engine_continuation_dependencies,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
)
from intergrax.runtime.execution.document_store_execution_terminal_outcome_by_execution_id import (
    DocumentStoreExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from testing_support.uca6c_process_restart_durable_document_store import (
    ProcessRestartQualificationDocumentStore,
)

StructuredJsonObject = dict[str, Any]


@dataclass
class Uca6cTrueRestartDurableBackends:
    """External persistence simulated for Host A/B/C — only these objects are shared."""

    document_store: ProcessRestartQualificationDocumentStore
    continuation_backing: ExecutionContinuationDurableBacking
    checkpoint_db_path: Path
    idempotency_db_path: Path
    continuation_export: StructuredJsonObject | None = field(default=None)
    aw_schema_name: str | None = None

    @classmethod
    def create(cls, tmp_path: Path) -> Uca6cTrueRestartDurableBackends:
        root = tmp_path / "uca6c-r59-r1-restart"
        root.mkdir(parents=True, exist_ok=True)
        return cls(
            document_store=ProcessRestartQualificationDocumentStore(),
            continuation_backing=ExecutionContinuationDurableBacking(),
            checkpoint_db_path=root / "task_checkpoints.db",
            idempotency_db_path=root / "tool_idempotency.db",
        )

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
        self.continuation_export = export_durable_continuation_state(
            self.continuation_backing,
        )

    def host_b_continuation_dependencies(
        self,
    ) -> ExecutionEngineContinuationDependencies:
        if self.continuation_export is None:
            raise RuntimeError("continuation export missing before Host B composition")
        store = execution_continuation_state_store_from_durable_export(
            self.continuation_export,
        )
        return reconnect_execution_engine_continuation_dependencies(state_store=store)


__all__ = ["Uca6cTrueRestartDurableBackends"]
