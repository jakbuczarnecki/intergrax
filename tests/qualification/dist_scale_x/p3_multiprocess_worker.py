# © Artur Czarnecki. All rights reserved.

"""P3 DocumentStore worker with shared effect counter (WL-3 hot key)."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

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
    ok: bool = True


def _bump_effect_counter(db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS effects (k TEXT PRIMARY KEY, v INTEGER NOT NULL)")
        conn.execute("INSERT INTO effects(k,v) VALUES('logical',1) ON CONFLICT(k) DO UPDATE SET v=v+1")
        row = conn.execute("SELECT v FROM effects WHERE k='logical'").fetchone()
        conn.commit()
        return int(row[0])


def _build_worker(
    *,
    dsn: str,
    schema: str,
    task_name: str,
    effect_db: str,
) -> DocumentStoreTaskWorker:
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    registry = TaskExecutionRegistry()
    effect_path = Path(effect_db)

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity,
    ):
        _ = tenant_id, run_id, payload, idempotency_key, execution_identity
        _bump_effect_counter(effect_path)
        return ToolExecutionResult.ok(_Out())

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


def drain_loop(
    dsn: str,
    schema: str,
    task_name: str,
    stop_after_idle_s: float,
    effect_db: str,
) -> int:
    worker = _build_worker(dsn=dsn, schema=schema, task_name=task_name, effect_db=effect_db)
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


def read_effect_count(effect_db: str) -> int:
    path = Path(effect_db)
    if not path.is_file():
        return 0
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT v FROM effects WHERE k='logical'").fetchone()
    return int(row[0]) if row else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--task-name", default="dist_scale_x.p3.wl3")
    parser.add_argument("--stop-idle-s", type=float, default=2.0)
    parser.add_argument("--effect-db", required=True)
    args = parser.parse_args()
    drain_loop(args.dsn, args.schema, args.task_name, args.stop_idle_s, args.effect_db)


if __name__ == "__main__":
    main()
