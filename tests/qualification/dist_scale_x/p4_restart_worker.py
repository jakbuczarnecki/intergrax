# © Artur Czarnecki. All rights reserved.

"""Durable restart worker — write then reload F01/F03/F07/F09 (P4 H)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

from tests.qualification.dist_scale_x._p4_support import ensure_qual_pg_schema
from tests.qualification.dist_scale_x.p4_decision_checkpoint_sample import (
    build_p4_qualification_decision_checkpoint,
)

_H_SUBJECT = "restart-h01"
_H_DECISION_ID = "decision_0123456789abcdef0123456789abcdef"


def _write_bundle(dsn: str, schema: str, tenant_id: str) -> dict[str, str]:
    ensure_qual_pg_schema(dsn, schema)
    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": dsn, "tenant_schema": schema},
    )
    store = PostgreSQLTaskCheckpointStore(config=config)
    task = Task(
        tenant_id=tenant_id,
        user_id="h1",
        message="restart",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
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
        )
        decision_checkpoint = build_p4_qualification_decision_checkpoint(
            tenant_id=tenant_id,
            subject=_H_SUBJECT,
            decision_id=_H_DECISION_ID,
        )
        decision_store = open_postgresql_decision_checkpoint_store(
            config=config,
            payload_codecs=conformance_artifact_payload_codec_registry(),
        )
        save_decision_checkpoint(decision_store, checkpoint=decision_checkpoint)
        doc = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
        event_store = DocumentBackedRuntimeEventStore(doc)
        event_store.append(sample_runtime_event(tenant_id=tenant_id, run_id=run_id), tenant_id=tenant_id)
        budget_store = DocumentStoreRunBudgetPersistence(doc)
        budget_store.compare_and_swap_snapshot(
            tenant_id=tenant_id,
            run_id=run_id,
            expected=None,
            snapshot=RunBudgetLedgerSnapshot.empty(
                attempt_id=attempt_id,
                root_limits=RunBudget(max_total_tokens=10_000, max_tool_calls=100),
            ),
        )
        event_store.close()
    finally:
        reset_active_execution_identity(token)
    return {
        "tenant_id": tenant_id,
        "task_id": saved.task_id,
        "run_id": run_id,
        "decision_subject": _H_SUBJECT,
        "decision_id": _H_DECISION_ID,
    }


def _reload_bundle(dsn: str, schema: str, identity: dict[str, str]) -> dict[str, object]:
    ensure_qual_pg_schema(dsn, schema)
    tenant_id = identity["tenant_id"]
    task_id = identity["task_id"]
    run_id = identity["run_id"]
    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": dsn, "tenant_schema": schema},
    )
    f01 = PostgreSQLTaskCheckpointStore(config=config).get_latest(task_id, tenant_id)
    key = build_p4_qualification_decision_checkpoint(
        tenant_id=tenant_id,
        subject=identity["decision_subject"],
        decision_id=identity["decision_id"],
    ).finalization.key
    f03 = open_postgresql_decision_checkpoint_store(
        config=config,
        payload_codecs=conformance_artifact_payload_codec_registry(),
    ).load_materialized(key=key)
    doc = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    budget = DocumentStoreRunBudgetPersistence(doc).load_snapshot(tenant_id=tenant_id, run_id=run_id)
    return {
        "f01_revision": f01.revision if f01 else None,
        "f03_present": f03 is not None,
        "f09_present": budget is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--tenant-id", default="tenant-h")
    parser.add_argument("--mode", choices=("write", "reload"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--identity", default="")
    args = parser.parse_args()
    out = Path(args.out)
    if args.mode == "write":
        out.write_text(json.dumps(_write_bundle(args.dsn, args.schema, args.tenant_id)), encoding="utf-8")
        return
    identity = json.loads(Path(args.identity).read_text(encoding="utf-8"))
    out.write_text(
        json.dumps(_reload_bundle(args.dsn, args.schema, identity)),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
