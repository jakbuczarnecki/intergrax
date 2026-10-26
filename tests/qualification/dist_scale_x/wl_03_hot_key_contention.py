# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P3 WL-3 hot idempotency key contention."""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from intergrax.integrations.providers.document_store.postgresql.bundle import (
    create_postgresql_document_store,
)
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from testing_support.uca6c_r6_r5_9_r4_postgresql_durable_document_store import (
    drop_postgresql_document_schema,
)

from tests.qualification.dist_scale_x._p2_support import resolve_pg_dsn
from tests.qualification.dist_scale_x._p3_support import (
    P3_MAX_IN_FLIGHT,
    P3_WORKERS,
    WL3_MEASURE_SECONDS,
    WL3_WARMUP_SECONDS,
    new_run_id,
    p3_record,
    require_p3_infra,
    write_p3_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot
from tests.qualification.dist_scale_x.p3_multiprocess_worker import read_effect_count

pytestmark = [pytest.mark.qualification, pytest.mark.integration, pytest.mark.network]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_TASK = "dist_scale_x.p3.wl3"
_IDEM_KEY = "dist-scale-x-p3-wl3-hot-key"


def test_p3_wl03_hot_idempotency_key_single_logical_effect() -> None:
    require_p3_infra()
    run_id = new_run_id("wl03-hot")
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p3_wl3_{uuid.uuid4().hex}"
    session_dir = _REPO_ROOT / ".tmp" / "session" / "dist-scale-x-p3-wl3" / run_id
    session_dir.mkdir(parents=True, exist_ok=True)
    effect_db = session_dir / "effect_counter.db"
    idle_s = float(WL3_WARMUP_SECONDS + WL3_MEASURE_SECONDS + 180)
    module = "tests.qualification.dist_scale_x.p3_multiprocess_worker"
    procs = [
        subprocess.Popen(
            [
                _PYTHON,
                "-m",
                module,
                "--dsn",
                dsn,
                "--schema",
                schema,
                "--task-name",
                _TASK,
                "--effect-db",
                str(effect_db),
                "--stop-idle-s",
                str(idle_s),
            ],
            cwd=_REPO_ROOT,
        )
        for _ in range(P3_WORKERS)
    ]
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    tenant = "tenant-wl3"
    unique_handles: set[str] = set()
    submitted = 0
    open_in_flight = 0
    succeeded = 0
    try:
        warmup_deadline = time.monotonic() + WL3_WARMUP_SECONDS
        while time.monotonic() < warmup_deadline:
            if open_in_flight < P3_MAX_IN_FLIGHT:
                handle = queue.enqueue(
                    TaskRequest(
                        tenant_id=tenant,
                        run_id=f"r-{submitted}",
                        task_name=_TASK,
                        payload=b"wl3",
                        idempotency_key=_IDEM_KEY,
                    )
                )
                unique_handles.add(handle.task_id)
                submitted += 1
                open_in_flight += 1
            time.sleep(0.01)
        measure_deadline = time.monotonic() + WL3_MEASURE_SECONDS
        while time.monotonic() < measure_deadline:
            rows = queue.list_tasks(tenant, limit=200)
            open_in_flight = sum(
                1 for r in rows if r.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            )
            while open_in_flight < P3_MAX_IN_FLIGHT:
                handle = queue.enqueue(
                    TaskRequest(
                        tenant_id=tenant,
                        run_id=f"r-{submitted}",
                        task_name=_TASK,
                        payload=b"wl3",
                        idempotency_key=_IDEM_KEY,
                    )
                )
                unique_handles.add(handle.task_id)
                submitted += 1
                open_in_flight += 1
                if time.monotonic() >= measure_deadline:
                    break
            time.sleep(0.02)
        drain_deadline = time.monotonic() + 120.0
        while time.monotonic() < drain_deadline:
            rows = queue.list_tasks(tenant, limit=200)
            pending = sum(1 for r in rows if r.status is TaskStatus.PENDING)
            if pending == 0:
                break
            time.sleep(0.1)
        final_rows = queue.list_tasks(tenant, limit=200)
        succeeded = sum(1 for r in final_rows if r.status is TaskStatus.SUCCEEDED)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        time.sleep(2.0)
        drop_postgresql_document_schema(schema, dsn=dsn)
    effects = read_effect_count(str(effect_db))
    assert len(unique_handles) == 1, f"queue idempotency collapsed to one task_id, got {len(unique_handles)}"
    assert succeeded == 1, f"expected one terminal SUCCEEDED task, got {succeeded}"
    assert effects == 1, f"expected exactly one semantic effect, got {effects}"
    assert submitted > 10
    topology = (
        f"P3 WL-3; workers={P3_WORKERS}; in_flight_cap={P3_MAX_IN_FLIGHT}; "
        f"idempotency_key={_IDEM_KEY}; effect_db={effect_db.name}"
    )
    record = p3_record(
        run_id=run_id,
        suffix="wl03",
        workload_id="WL-3",
        dimension_ids=("D",),
        duration_seconds=float(WL3_WARMUP_SECONDS + WL3_MEASURE_SECONDS),
        topology_description=topology,
        fault_description="none",
        metrics=DistScaleXMetricsSnapshot(
            extra={
                "submitted": float(submitted),
                "unique_task_ids": float(len(unique_handles)),
                "logical_effects": float(effects),
            },
        ),
    )
    write_p3_manifest(run_id=run_id, records=(record,), topology_description=topology)
