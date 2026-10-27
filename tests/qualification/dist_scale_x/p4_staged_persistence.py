# © Artur Czarnecki. All rights reserved.

"""Staged P4 persistence writes for crash/recovery boundaries (F01→F03→F07→F09)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.integrations.providers.document_store.postgresql.bundle import (
    create_postgresql_document_store,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
    DocumentBackedRuntimeEventStore,
)
from intergrax.runtime.execution.budget.persistence import DocumentStoreRunBudgetPersistence
from intergrax.runtime.execution.budget.snapshot import RunBudgetLedgerSnapshot
from intergrax.runtime.execution.decision_checkpoint_persistence import save_decision_checkpoint
from intergrax.runtime.execution.decision_finalization_conformance import (
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
    PostgreSQLTaskCheckpointStore,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.persistence.postgresql_opens import open_postgresql_decision_checkpoint_store
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

from tests.qualification.dist_scale_x._p4_support import prime_p4_qualification_stores
from tests.qualification.dist_scale_x.p4_decision_checkpoint_sample import (
    build_p4_qualification_decision_checkpoint,
)

BoundaryId = Literal["G-01", "G-02", "G-03", "G-04", "G-05", "G-06"]

BOUNDARY_SYNC: dict[BoundaryId, str] = {
    "G-01": "armed",
    "G-02": "after_f01",
    "G-03": "after_f03",
    "G-04": "after_f07",
    "G-05": "after_f09",
    "G-06": "after_complete",
}


@dataclass(slots=True)
class StagedWriteHandle:
    dsn: str
    schema: str
    tenant_id: str
    task_id: str
    run_id: str
    attempt_id: str
    event_id: str
    decision_id: str
    decision_subject: str
    boundary: BoundaryId


def _config(dsn: str, schema: str) -> PostgreSQLIntegrationConfig:
    return PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": dsn, "tenant_schema": schema},
    )


def execute_staged_boundary(
    *,
    dsn: str,
    schema: str,
    tenant_id: str,
    boundary: BoundaryId,
    task_id: str,
    subject: str = "staged-g",
    decision_id: str = "decision_0123456789abcdef0123456789abcdef",
    payload_pad: str = "s" * 1024,
) -> StagedWriteHandle:
    prime_p4_qualification_stores(dsn, schema)
    if boundary == "G-01":
        return StagedWriteHandle(
            dsn=dsn,
            schema=schema,
            tenant_id=tenant_id,
            task_id=task_id,
            run_id="",
            attempt_id="",
            event_id="",
            decision_id=decision_id,
            decision_subject=subject,
            boundary=boundary,
        )

    config = _config(dsn, schema)
    store = PostgreSQLTaskCheckpointStore(config=config)
    task = Task(
        tenant_id=tenant_id,
        user_id="staged",
        message=payload_pad[:1024],
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    task.task_id = task_id
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    event_store: DocumentBackedRuntimeEventStore | None = None
    try:
        saved = LongRunningCoordinator.persist_checkpoint(
            task,
            store,
            run_id=run_id,
            attempt_id=attempt_id,
            progress_message=payload_pad[:1024],
        )
        if boundary == "G-02":
            return StagedWriteHandle(
                dsn=dsn,
                schema=schema,
                tenant_id=tenant_id,
                task_id=saved.task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_id="",
                decision_id=decision_id,
                decision_subject=subject,
                boundary=boundary,
            )

        decision_store = open_postgresql_decision_checkpoint_store(
            config=config,
            payload_codecs=conformance_artifact_payload_codec_registry(),
        )
        save_decision_checkpoint(
            decision_store,
            checkpoint=build_p4_qualification_decision_checkpoint(
                tenant_id=tenant_id,
                subject=subject,
                decision_id=decision_id,
            ),
        )
        if boundary == "G-03":
            return StagedWriteHandle(
                dsn=dsn,
                schema=schema,
                tenant_id=tenant_id,
                task_id=saved.task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_id="",
                decision_id=decision_id,
                decision_subject=subject,
                boundary=boundary,
            )

        doc_store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
        event_store = DocumentBackedRuntimeEventStore(doc_store)
        event = sample_runtime_event(tenant_id=tenant_id, run_id=run_id)
        event_store.append(event, tenant_id=tenant_id)
        if boundary == "G-04":
            return StagedWriteHandle(
                dsn=dsn,
                schema=schema,
                tenant_id=tenant_id,
                task_id=saved.task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                event_id=event.event_id,
                decision_id=decision_id,
                decision_subject=subject,
                boundary=boundary,
            )

        budget_store = DocumentStoreRunBudgetPersistence(doc_store)
        budget_store.compare_and_swap_snapshot(
            tenant_id=tenant_id,
            run_id=run_id,
            expected=None,
            snapshot=RunBudgetLedgerSnapshot.empty(
                attempt_id=attempt_id,
                root_limits=RunBudget(max_total_tokens=10_000, max_tool_calls=100),
            ),
        )
        return StagedWriteHandle(
            dsn=dsn,
            schema=schema,
            tenant_id=tenant_id,
            task_id=saved.task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            event_id=event.event_id,
            decision_id=decision_id,
            decision_subject=subject,
            boundary=boundary,
        )
    finally:
        reset_active_execution_identity(token)
        if event_store is not None:
            event_store.close()


def handle_to_json(handle: StagedWriteHandle) -> str:
    return json.dumps(asdict(handle))


def handle_from_json(raw: str) -> StagedWriteHandle:
    data = json.loads(raw)
    return StagedWriteHandle(**data)

