# © Artur Czarnecki. All rights reserved.

"""RabbitMQ M worker for DIST-SCALE-X P3 WL-6 (fault injection + effect ledger)."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.integrations.providers.message_bus.rabbitmq.bundle import (
    create_rabbitmq_integration,
    create_rabbitmq_worker,
)
from intergrax.queueing.worker.execution import RetryableHandlerError
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel

from tests.qualification.dist_scale_x._p2_support import (
    P2_RABBIT_HOST,
    P2_RABBIT_PASS,
    P2_RABBIT_USER,
)

P3_RABBIT_TASK = "dist_scale_x.p3.wl6.rabbit"
_FAULT_RATE_PERCENT = 20


class _Out(BaseModel):
    logical_id: str
    ok: bool = True


def _state_conn(state_db: Path) -> sqlite3.Connection:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(state_db, timeout=30.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wl6_state (k TEXT PRIMARY KEY, v TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wl6_effects (logical_id TEXT PRIMARY KEY)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wl6_metrics (k TEXT PRIMARY KEY, v INTEGER NOT NULL)"
    )
    conn.commit()
    return conn


def _fault_injection_enabled(state_db: Path) -> bool:
    with _state_conn(state_db) as conn:
        row = conn.execute(
            "SELECT v FROM wl6_state WHERE k='fault_injection'"
        ).fetchone()
        return row is not None and row[0] == "1"


def _bump_physical_attempts(state_db: Path) -> None:
    with _state_conn(state_db) as conn:
        conn.execute(
            "INSERT INTO wl6_metrics(k,v) VALUES('physical_attempts',1) "
            "ON CONFLICT(k) DO UPDATE SET v=v+1"
        )
        conn.commit()


def _record_effect_once(state_db: Path, logical_id: str) -> bool:
    with _state_conn(state_db) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO wl6_effects(logical_id) VALUES(?)",
            (logical_id,),
        )
        conn.commit()
        return cur.rowcount > 0


def _transient_fault(logical_id: str) -> bool:
    return (hash(logical_id) % 100) < _FAULT_RATE_PERCENT


def poll_loop(
    queue_name: str,
    stop_idle_s: float,
    *,
    kv_db: str,
    state_db: str,
) -> int:
    state_path = Path(state_db)
    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity,
    ):
        _ = run_id, idempotency_key, execution_identity
        _bump_physical_attempts(state_path)
        body = json.loads(payload.decode("utf-8"))
        logical_id = str(body["logical_id"])
        if str(body.get("tenant_id")) != tenant_id:
            raise RuntimeError("tenant payload mismatch")
        if _fault_injection_enabled(state_path) and _transient_fault(logical_id):
            raise RetryableHandlerError("transient broker handler fault")
        _record_effect_once(state_path, logical_id)
        return ToolExecutionResult.ok(_Out(logical_id=logical_id))

    registry.register(P3_RABBIT_TASK, handler)
    kv = SqliteDistributedKVStore(Path(kv_db))
    bundle = create_rabbitmq_integration(
        kv_store=kv,
        host=P2_RABBIT_HOST,
        username=P2_RABBIT_USER,
        password=P2_RABBIT_PASS,
        queue=queue_name,
    )
    worker = create_rabbitmq_worker(
        kv_store=kv,
        execution_registry=registry,
        host=P2_RABBIT_HOST,
        username=P2_RABBIT_USER,
        password=P2_RABBIT_PASS,
        queue=queue_name,
        consumer=bundle.consumer,
        poll_timeout_seconds=0.3,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )
    processed = 0
    idle_start: float | None = None
    while True:
        raw = bundle.consumer.poll(timeout_seconds=0.3)
        if raw is not None:
            try:
                worker.process_message(raw_payload=raw)
            except RetryableHandlerError:
                pass
            processed += 1
            idle_start = None
            continue
        if idle_start is None:
            idle_start = time.monotonic()
        elif time.monotonic() - idle_start >= stop_idle_s:
            break
    return processed


def set_fault_injection(state_db: Path, enabled: bool) -> None:
    with _state_conn(state_db) as conn:
        conn.execute(
            "INSERT INTO wl6_state(k,v) VALUES('fault_injection',?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            ("1" if enabled else "0",),
        )
        conn.commit()


def read_wl6_metrics(state_db: Path) -> dict[str, int]:
    with _state_conn(state_db) as conn:
        attempts = conn.execute(
            "SELECT v FROM wl6_metrics WHERE k='physical_attempts'"
        ).fetchone()
        effects = conn.execute("SELECT COUNT(*) FROM wl6_effects").fetchone()
    return {
        "physical_attempts": int(attempts[0]) if attempts else 0,
        "semantic_effects": int(effects[0]) if effects else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default=f"dist-scale-x-p3-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--stop-idle-s", type=float, default=2.0)
    parser.add_argument("--kv-db", required=True)
    parser.add_argument("--state-db", required=True)
    args = parser.parse_args()
    processed = 0
    try:
        processed = poll_loop(
            args.queue,
            args.stop_idle_s,
            kv_db=args.kv_db,
            state_db=args.state_db,
        )
    finally:
        metrics = Path(args.state_db).parent
        metrics.mkdir(parents=True, exist_ok=True)
        (metrics / f"worker_{os.getpid()}.json").write_text(
            json.dumps({"processed": processed}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
