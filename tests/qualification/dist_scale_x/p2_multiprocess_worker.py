# © Artur Czarnecki. All rights reserved.

"""Entry points for DIST-SCALE-X P2 multi-process DocumentStore workers (OS processes)."""

from __future__ import annotations

import argparse
import time

from intergrax.integrations.providers.document_store.postgresql.bundle import (
    create_postgresql_document_store,
)
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.providers.document_store.colocated_worker import DocumentStoreTaskWorker
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.background_execution.admission_wiring import (
    wire_background_execution_admission_dependencies,
)
from intergrax.runtime.observability.document_store_causal_evidence_persistence import (
    wire_causal_evidence_persistence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel


class _Out(BaseModel):
    tenant_id: str
    counter: int


def _build_worker(
    *,
    dsn: str,
    schema: str,
    task_name: str,
) -> DocumentStoreTaskWorker:
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity,
    ):
        _ = run_id, payload, idempotency_key, execution_identity
        return ToolExecutionResult.ok(_Out(tenant_id=tenant_id, counter=1))

    registry.register(task_name, handler)
    admission = wire_background_execution_admission_dependencies(document_store=store)
    return DocumentStoreTaskWorker(
        queue,
        registry,
        claim_limit=2,
        identity_persistence=admission.identity_persistence,
        attempt_lifecycle=admission.attempt_lifecycle,
        execution_terminal=admission.execution_terminal,
        causal_evidence_persistence=wire_causal_evidence_persistence(document_store=store),
    )


def drain_loop(dsn: str, schema: str, task_name: str, stop_after_idle_s: float) -> int:
    worker = _build_worker(dsn=dsn, schema=schema, task_name=task_name)
    drained = 0
    idle_start: float | None = None
    while True:
        count = worker.drain_once()
        if count:
            drained += count
            idle_start = None
            continue
        if idle_start is None:
            idle_start = time.monotonic()
        elif time.monotonic() - idle_start >= stop_after_idle_s:
            break
        time.sleep(0.05)
    return drained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--task-name", default="dist_scale_x.p2.task")
    parser.add_argument("--stop-idle-s", type=float, default=2.0)
    args = parser.parse_args()
    drain_loop(args.dsn, args.schema, args.task_name, args.stop_idle_s)


if __name__ == "__main__":
    main()
