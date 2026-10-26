# © Artur Czarnecki. All rights reserved.

"""Celery dispatcher + RabbitMQ AMQP broker for DIST-SCALE-X P3 WL-6 (canonical RetryPolicy owner)."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from celery import Celery
from pydantic import BaseModel

from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.queueing.worker.execution import RetryableHandlerError
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.queueing.worker.retry_event import RetryEvent
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.queueing.worker_bootstrap import create_celery_worker_app
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore
from intergrax.tools.execution_models import ToolExecutionResult
from tests.qualification.dist_scale_x._p2_support import (
    P2_RABBIT_HOST,
    P2_RABBIT_PASS,
    P2_RABBIT_PORT,
    P2_RABBIT_USER,
)

P3_CELERY_TASK = "dist_scale_x.p3.wl6.celery"
_FAULT_RATE_PERCENT = 20
_LOCK_TTL_SECONDS = 600

WL6_RETRY_POLICY = RetryPolicy(
    max_retries=8,
    initial_backoff_seconds=0.25,
    backoff_multiplier=1.5,
    max_backoff_seconds=12.0,
    jitter=False,
    retry_on_lock_conflict=True,
    retry_on_handler_exception=True,
)


class _Out(BaseModel):
    logical_id: str
    ok: bool = True


def _broker_url() -> str:
    return (
        f"amqp://{P2_RABBIT_USER}:{P2_RABBIT_PASS}@"
        f"{P2_RABBIT_HOST}:{P2_RABBIT_PORT}//"
    )


def _state_conn(state_db: Path) -> sqlite3.Connection:
    state_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(state_db, timeout=30.0)
    conn.execute("CREATE TABLE IF NOT EXISTS wl6_state (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wl6_metrics (k TEXT PRIMARY KEY, v INTEGER NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wl6_semantic_effects (logical_id TEXT PRIMARY KEY)"
    )
    conn.commit()
    return conn


def set_fault_injection(state_db: Path, enabled: bool) -> None:
    with _state_conn(state_db) as conn:
        conn.execute(
            "INSERT INTO wl6_state(k,v) VALUES('fault_injection',?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            ("1" if enabled else "0",),
        )
        conn.commit()


def _fault_enabled(state_db: Path) -> bool:
    with _state_conn(state_db) as conn:
        row = conn.execute("SELECT v FROM wl6_state WHERE k='fault_injection'").fetchone()
    return row is not None and row[0] == "1"


def _bump_metric(state_db: Path, key: str) -> None:
    with _state_conn(state_db) as conn:
        conn.execute(
            f"INSERT INTO wl6_metrics(k,v) VALUES(?,1) ON CONFLICT(k) DO UPDATE SET v=v+1",
            (key,),
        )
        conn.commit()


def read_wl6_celery_metrics(state_db: Path) -> dict[str, int]:
    with _state_conn(state_db) as conn:
        rows = conn.execute("SELECT k, v FROM wl6_metrics").fetchall()
        effects = conn.execute("SELECT COUNT(*) FROM wl6_semantic_effects").fetchone()
    out = {str(k): int(v) for k, v in rows}
    out["semantic_effects"] = int(effects[0]) if effects else 0
    return out


def build_wl6_celery_app(
    *,
    state_db: Path,
    kv_db: Path,
    idem_db: Path,
    result_db: Path,
) -> Celery:
    registry = TaskExecutionRegistry()

    def handler(
        *,
        tenant_id: str,
        run_id: str,
        payload: bytes,
        idempotency_key,
        execution_identity,
    ):
        _ = run_id, execution_identity
        _bump_metric(state_db, "physical_attempts")
        body = json.loads(payload.decode("utf-8"))
        logical_id = str(body["logical_id"])
        if str(body.get("tenant_id")) != tenant_id:
            raise RuntimeError("tenant payload mismatch")
        if _fault_enabled(state_db) and (hash(logical_id) % 100) < _FAULT_RATE_PERCENT:
            raise RetryableHandlerError("transient wl6 handler fault")
        with _state_conn(state_db) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO wl6_semantic_effects(logical_id) VALUES(?)",
                (logical_id,),
            )
            conn.commit()
        return ToolExecutionResult.ok(_Out(logical_id=logical_id))

    registry.register(P3_CELERY_TASK, handler)

    def on_retry_scheduled(event: RetryEvent) -> None:
        _bump_metric(state_db, "dispatcher_retries")
        _ = event

    kv = SqliteDistributedKVStore(kv_db)
    idem = SQLiteIdempotencyStore(str(idem_db))
    backend = f"db+sqlite:///{result_db.as_posix()}"

    return create_celery_worker_app(
        app_name="dist-scale-x-p3-wl6",
        broker_url=_broker_url(),
        backend_url=backend,
        registry=registry,
        idempotency_store=idem,
        retry_policy=WL6_RETRY_POLICY,
        lock_ttl_seconds=_LOCK_TTL_SECONDS,
        completed_ttl_seconds=3600,
        on_retry_scheduled=on_retry_scheduled,
        kv_store=kv,
        causal_evidence_persistence=InMemoryCausalEvidencePersistence(),
    )


def _maybe_build_app() -> Celery:
    state = os.environ.get("DIST_SCALE_X_P3_WL6_STATE_DB")
    if not state:
        return Celery("dist-scale-x-p3-wl6-unconfigured")
    base = Path(state).parent
    return build_wl6_celery_app(
        state_db=Path(state),
        kv_db=Path(os.environ["DIST_SCALE_X_P3_WL6_KV_DB"]),
        idem_db=Path(os.environ["DIST_SCALE_X_P3_WL6_IDEM_DB"]),
        result_db=Path(os.environ.get("DIST_SCALE_X_P3_WL6_RESULT_DB", str(base / "celery_results.db"))),
    )


app = _maybe_build_app()
