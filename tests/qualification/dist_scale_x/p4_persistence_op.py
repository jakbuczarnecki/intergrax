# © Artur Czarnecki. All rights reserved.

"""Single WL-5 persistence operation across SX-F01, F03, F07, F09 (P4-A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
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

from tests.qualification.dist_scale_x._p4_support import (
    P4_QUAL_DECISION_ID,
    prime_p4_qualification_stores,
)
from tests.qualification.dist_scale_x.p4_decision_checkpoint_sample import (
    build_p4_qualification_decision_checkpoint,
)


@dataclass(slots=True)
class P4OpResult:
    kind: str
    tenant_id: str
    task_id: str
    error: str | None = None


def execute_p4_persistence_op(
    *,
    dsn: str,
    schema: str,
    tenant_id: str,
    op_index: int,
    payload_pad: str,
    task_id: str | None = None,
    document_store_factory: Callable[[], ConditionalDocumentStore] | None = None,
) -> P4OpResult:
    """~90% write (F01+F03+F07+F09), ~10% read (F01/F03)."""
    is_read = op_index % 10 == 0
    try:
        prime_p4_qualification_stores(dsn, schema)
    except Exception as exc:  # noqa: BLE001 — qualification surface
        return P4OpResult(
            kind="read_error" if is_read else "write_error",
            tenant_id=tenant_id,
            task_id=task_id or "",
            error=f"{type(exc).__name__}:{exc}",
        )
    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": dsn, "tenant_schema": schema},
    )
    store = PostgreSQLTaskCheckpointStore(config=config)
    if is_read:
        if not task_id:
            return P4OpResult(kind="read_skip", tenant_id=tenant_id, task_id="")
        latest = store.get_latest(task_id, tenant_id)
        if latest is None:
            return P4OpResult(kind="read_miss", tenant_id=tenant_id, task_id=task_id)
        return P4OpResult(kind="read", tenant_id=tenant_id, task_id=task_id)

    if document_store_factory is not None:
        doc_store = document_store_factory()
    else:
        doc_store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    event_store: DocumentBackedRuntimeEventStore | None = DocumentBackedRuntimeEventStore(
        doc_store,
    )
    budget_store = DocumentStoreRunBudgetPersistence(doc_store)

    task = Task(
        tenant_id=tenant_id,
        user_id="wl5",
        message=payload_pad[:1024],
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    if task_id:
        task.task_id = task_id
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        saved = LongRunningCoordinator.persist_checkpoint(
            task,
            store,
            run_id=run_id,
            attempt_id=attempt_id,
            progress_message=payload_pad[:1024],
        )
        decision_store = open_postgresql_decision_checkpoint_store(
            config=config,
            payload_codecs=conformance_artifact_payload_codec_registry(),
        )
        save_decision_checkpoint(
            decision_store,
            checkpoint=build_p4_qualification_decision_checkpoint(
                tenant_id=tenant_id,
                subject=f"wl5-{op_index}",
                decision_id=P4_QUAL_DECISION_ID,
            ),
        )
        event = sample_runtime_event(tenant_id=tenant_id, run_id=run_id)
        event_store.append(event, tenant_id=tenant_id)
        snap = RunBudgetLedgerSnapshot.empty(
            attempt_id=attempt_id,
            root_limits=RunBudget(max_total_tokens=10_000, max_tool_calls=100),
        )
        budget_store.compare_and_swap_snapshot(
            tenant_id=tenant_id,
            run_id=run_id,
            expected=None,
            snapshot=snap,
        )
        result = P4OpResult(kind="write", tenant_id=tenant_id, task_id=saved.task_id)
        return result
    except Exception as exc:  # noqa: BLE001 — qualification surface
        return P4OpResult(
            kind="write_error",
            tenant_id=tenant_id,
            task_id=task.task_id,
            error=f"{type(exc).__name__}:{exc}",
        )
    finally:
        reset_active_execution_identity(token)
        if event_store is not None:
            event_store.close()


def mint_qual_task_id() -> str:
    return str(mint_task_id())
