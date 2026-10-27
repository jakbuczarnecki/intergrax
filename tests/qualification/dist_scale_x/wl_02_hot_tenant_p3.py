# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P3 WL-2 hot tenant continuation (85% measured)."""

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
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from testing_support.uca6c_r6_r5_9_r4_postgresql_durable_document_store import (
    drop_postgresql_document_schema,
)

from tests.qualification.dist_scale_x._p3_support import (
    P3_MAX_IN_FLIGHT,
    P3_WORKERS,
    WL2_MEASURE_SECONDS,
    WL2_WARMUP_SECONDS,
    new_run_id,
    p3_record,
    percentile_ms,
    require_p3_pg_infra,
    write_p3_manifest,
)
from tests.qualification.dist_scale_x.p3_task_load import (
    P3_HOT_SHARE_TARGET,
    run_p3_hot_tenant_completed_work_load,
)
from tests.qualification.dist_scale_x._p2_support import resolve_pg_dsn
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot

pytestmark = [pytest.mark.qualification, pytest.mark.integration, pytest.mark.network]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_TASK = "dist_scale_x.p3.wl2"


def _spawn_workers(dsn: str, schema: str, count: int, idle_s: float) -> list[subprocess.Popen[bytes]]:
    module = "tests.qualification.dist_scale_x.p2_multiprocess_worker"
    procs = []
    for _ in range(count):
        procs.append(
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
                    "--stop-idle-s",
                    str(idle_s),
                ],
                cwd=_REPO_ROOT,
            )
        )
    return procs


def _terminate(procs: list[subprocess.Popen[bytes]]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()


def _workers_alive(procs: list[subprocess.Popen[bytes]]) -> None:
    for p in procs:
        if p.poll() is not None:
            raise AssertionError(f"worker exited {p.returncode}")


def test_p3_wl02_hot_tenant_85_percent_measured() -> None:
    require_p3_pg_infra()
    run_id = new_run_id("wl02-p3")
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p3_wl2_{uuid.uuid4().hex}"
    idle_s = float(WL2_WARMUP_SECONDS + WL2_MEASURE_SECONDS + 180)
    # Two workers reduce Windows PG ephemeral port pressure (P2 lesson @ ee5a8371e).
    workers = _spawn_workers(dsn, schema, 2, idle_s)
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    hot = "tenant-hot"
    cold = ("tenant-cold-a", "tenant-cold-b")
    try:
        run_p3_hot_tenant_completed_work_load(
            queue,
            duration_s=float(WL2_WARMUP_SECONDS),
            cold_tenants=cold,
            hot_tenant=hot,
            task_name=_TASK,
            max_in_flight=P3_MAX_IN_FLIGHT,
            worker_health=lambda: _workers_alive(workers),
        )
        accounting, ledger = run_p3_hot_tenant_completed_work_load(
            queue,
            duration_s=float(WL2_MEASURE_SECONDS),
            cold_tenants=cold,
            hot_tenant=hot,
            task_name=_TASK,
            max_in_flight=P3_MAX_IN_FLIGHT,
            worker_health=lambda: _workers_alive(workers),
        )
    finally:
        _terminate(workers)
        time.sleep(2.0)
        drop_postgresql_document_schema(schema, dsn=dsn)
    hot_lat = list(accounting.completion_latencies_by_tenant.get(hot, ()))
    cold_lat = [
        ms
        for tid, samples in accounting.completion_latencies_by_tenant.items()
        if tid in cold
        for ms in samples
    ]
    topology = (
        f"P3 WL-2; workers=2; in_flight={P3_MAX_IN_FLIGHT}; "
        f"hot_share_target={P3_HOT_SHARE_TARGET}; warmup={WL2_WARMUP_SECONDS}s; "
        f"measure={WL2_MEASURE_SECONDS}s"
    )
    record = p3_record(
        run_id=run_id,
        suffix="wl02",
        workload_id="WL-2",
        dimension_ids=("D", "K"),
        duration_seconds=float(WL2_WARMUP_SECONDS + WL2_MEASURE_SECONDS),
        topology_description=topology,
        fault_description="none",
        metrics=DistScaleXMetricsSnapshot(
            throughput_per_sec=accounting.useful_throughput_per_sec(float(WL2_MEASURE_SECONDS)),
            p95_latency_ms=percentile_ms(accounting.completion_latencies_ms, 95),
            extra={
                "measured_hot_share": ledger.measured_hot_share(),
                "submitted_hot": float(ledger.submitted_hot),
                "submitted_cold": float(ledger.submitted_cold),
                "p95_hot_ms": percentile_ms(hot_lat, 95) or 0.0,
                "p95_cold_ms": percentile_ms(cold_lat, 95) or 0.0,
                "completed_hot": float(accounting.completions_by_tenant.get(hot, 0)),
            },
        ),
    )
    write_p3_manifest(run_id=run_id, records=(record,), topology_description=topology)
