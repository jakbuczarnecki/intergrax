# © Artur Czarnecki. All rights reserved.

"""Celery dispatcher + RabbitMQ AMQP for DIST-SCALE-X P3 WL-6 (outside qual authority scan)."""

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
from intergrax.tools.execution_models import ToolExecutionResult

P3_CELERY_TASK = "dist_scale_x.p3.wl6.celery"
_FAULT_RATE_PERCENT = 20
_LOCK_TTL_SECONDS = 600
_RABBIT_HOST = "localhost"
_RABBIT_PORT = 5672
_RABBIT_USER = "intergrax"
_RABBIT_PASS = "intergrax"

WL6_RETRY_POLICY = RetryPolicy(
    max_retries=24,
    initial_backoff_seconds=0.15,
    backoff_multiplier=1.4,
    max_backoff_seconds=8.0,
    jitter=False,
    retry_on_lock_conflict=True,
    retry_on_handler_exception=True,
)


class _Out(BaseModel):
    logical_id: str
    ok: bool = True


def _broker_url() -> str:
    return f"amqp://{_RABBIT_USER}:{_RABBIT_PASS}@{_RABBIT_HOST}:{_RABBIT_PORT}//"


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
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wl6_logical_attempts "
        "(logical_id TEXT PRIMARY KEY, attempts INTEGER NOT NULL)"
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
            "INSERT INTO wl6_metrics(k,v) VALUES(?,1) ON CONFLICT(k) DO UPDATE SET v=v+1",
            (key,),
        )
        conn.commit()


def read_completed_logical_ids(state_db: Path) -> set[str]:
    with _state_conn(state_db) as conn:
        rows = conn.execute("SELECT logical_id FROM wl6_semantic_effects").fetchall()
    return {str(r[0]) for r in rows}


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
        with _state_conn(state_db) as conn:
            conn.execute(
                "INSERT INTO wl6_logical_attempts(logical_id, attempts) VALUES(?,1) "
                "ON CONFLICT(logical_id) DO UPDATE SET attempts=attempts+1",
                (logical_id,),
            )
            row = conn.execute(
                "SELECT attempts FROM wl6_logical_attempts WHERE logical_id=?",
                (logical_id,),
            ).fetchone()
            conn.commit()
        attempt_num = int(row[0]) if row else 1
        if (
            _fault_enabled(state_db)
            and attempt_num == 1
            and (hash(logical_id) % 100) < _FAULT_RATE_PERCENT
        ):
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

    return create_celery_worker_app(
        app_name="dist-scale-x-p3-wl6",
        broker_url=_broker_url(),
        backend_url="rpc://",
        registry=registry,
        idempotency_store=None,
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
    return build_wl6_celery_app(
        state_db=Path(state),
        kv_db=Path(os.environ["DIST_SCALE_X_P3_WL6_KV_DB"]),
    )


app = _maybe_build_app()
