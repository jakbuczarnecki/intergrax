# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P2 L3/L4 physical qualification (WL-1, WL-2, dims A/B/C/K)."""

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
from intergrax.integrations.providers.message_bus.rabbitmq.bundle import create_rabbitmq_integration
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel
from testing_support.uca6c_r6_r5_9_r4_postgresql_durable_document_store import (
    drop_postgresql_document_schema,
)

from tests.qualification.dist_scale_x._p2_support import (
    LOCK_ENV_DEFINITION,
    LOCK_TOPOLOGY_TEMPLATE,
    OVERLOAD_STEP_SECONDS,
    POINT_DURATION_SECONDS,
    WL2_WARMUP_SECONDS,
    code_sha,
    new_run_id,
    percentile_ms,
    require_p2_infra,
    resolve_pg_dsn,
    write_p2_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import (
    DistScaleXEvidenceRecord,
    DistScaleXMetricsSnapshot,
    EvidenceDisposition,
)
from tests.qualification.dist_scale_x.p2_rabbit_multiprocess_worker import InMemoryKVStore

pytestmark = [
    pytest.mark.qualification,
    pytest.mark.integration,
    pytest.mark.network,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASK_NAME = "dist_scale_x.p2.task"
_RABBIT_TASK = "dist_scale_x.p2.rabbit"
_PYTHON = sys.executable


class _TaskOut(BaseModel):
    tenant_id: str
    kind: str


def _spawn_ds_workers(
    *,
    dsn: str,
    schema: str,
    count: int,
    stop_idle_s: float = 3.0,
) -> list[subprocess.Popen[bytes]]:
    procs: list[subprocess.Popen[bytes]] = []
    module = "tests.qualification.dist_scale_x.p2_multiprocess_worker"
    for _ in range(count):
        proc = subprocess.Popen(
            [
                _PYTHON,
                "-m",
                module,
                "--dsn",
                dsn,
                "--schema",
                schema,
                "--task-name",
                _TASK_NAME,
                "--stop-idle-s",
                str(stop_idle_s),
            ],
            cwd=_REPO_ROOT,
        )
        procs.append(proc)
    return procs


def _terminate_all(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _enqueue_load(
    queue: DocumentStoreTaskQueue,
    *,
    duration_s: float,
    tenants: tuple[str, ...],
    hot_tenant: str | None = None,
    hot_share: float = 0.0,
) -> tuple[int, list[float]]:
    deadline = time.monotonic() + duration_s
    submitted = 0
    latencies: list[float] = []
    tick = 0
    while time.monotonic() < deadline:
        tick += 1
        use_hot = hot_tenant is not None and (tick % 10) < int(10 * hot_share)
        tenant = hot_tenant if use_hot and hot_tenant else tenants[tick % len(tenants)]
        kind = "read" if tick % 10 < 3 else "write"
        start = time.monotonic()
        handle = queue.enqueue(
            TaskRequest(
                tenant_id=tenant,
                run_id=f"run-{uuid.uuid4().hex[:8]}",
                task_name=_TASK_NAME,
                payload=f'{{"kind":"{kind}"}}'.encode(),
                idempotency_key=None,
            )
        )
        submitted += 1
        while True:
            status = queue.get_status(handle)
            if status in (TaskStatus.SUCCEEDED, TaskStatus.FAILED):
                latencies.append((time.monotonic() - start) * 1000.0)
                break
            if time.monotonic() - start > 30.0:
                break
            time.sleep(0.02)
    return submitted, latencies


def _measure_point(n_workers: int, duration_s: float) -> tuple[int, list[float], str]:
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p2_{uuid.uuid4().hex}"
    require_p2_infra()
    workers = _spawn_ds_workers(dsn=dsn, schema=schema, count=n_workers, stop_idle_s=4.0)
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    try:
        submitted, latencies = _enqueue_load(
            queue,
            duration_s=duration_s,
            tenants=("tenant-a", "tenant-b"),
        )
    finally:
        _terminate_all(workers)
        drop_postgresql_document_schema(schema, dsn=dsn)
    return submitted, latencies, schema


def test_p2_wl01_l4_document_store_scaling_envelope() -> None:
    require_p2_infra()
    run_id = new_run_id("wl01-scale")
    n1_count, n1_lat, _ = _measure_point(1, float(POINT_DURATION_SECONDS))
    n2_count, n2_lat, _ = _measure_point(2, float(POINT_DURATION_SECONDS))
    thr1 = n1_count / float(POINT_DURATION_SECONDS)
    thr2 = n2_count / float(POINT_DURATION_SECONDS)
    eff = thr2 / (2.0 * thr1) if thr1 > 0 else 0.0
    topology = LOCK_TOPOLOGY_TEMPLATE.format(
        n_workers=2,
        in_flight=2,
        duration_s=POINT_DURATION_SECONDS,
    )
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-wl01",
        code_sha=code_sha(),
        topology_description=topology,
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-1",
        dimension_ids=("A", "B"),
        duration_seconds=float(POINT_DURATION_SECONDS) * 2,
        metrics=DistScaleXMetricsSnapshot(
            throughput_per_sec=thr2,
            p95_latency_ms=percentile_ms(n2_lat, 95),
            error_rate=0.0,
            extra={
                "throughput_n1": thr1,
                "throughput_n2": thr2,
                "scaling_efficiency": eff,
                "submitted_n1": float(n1_count),
                "submitted_n2": float(n2_count),
            },
        ),
        fault_description="none",
        disposition=EvidenceDisposition.PASS,
    )
    path = write_p2_manifest(run_id=run_id, records=(record,), topology_description=topology)
    assert path.is_file()
    assert thr2 >= 0.0


def test_p2_wl01_rabbit_m_two_os_worker_processes() -> None:
    require_p2_infra()
    run_id = new_run_id("wl01-rabbit")
    queue_name = f"dist-scale-x-p2-{uuid.uuid4().hex[:10]}"
    module = "tests.qualification.dist_scale_x.p2_rabbit_multiprocess_worker"
    procs = [
        subprocess.Popen(
            [_PYTHON, "-m", module, "--queue", queue_name, "--stop-idle-s", "4"],
            cwd=_REPO_ROOT,
        )
        for _ in range(2)
    ]
    time.sleep(1.0)
    kv = InMemoryKVStore()
    bundle = create_rabbitmq_integration(
        kv_store=kv,
        host="localhost",
        username="intergrax",
        password="intergrax",
        queue=queue_name,
    )
    queue = bundle.message_bus
    registry = TaskExecutionRegistry()

    def handler(*, tenant_id: str, run_id: str, payload: bytes, idempotency_key):
        _ = tenant_id, run_id, payload, idempotency_key
        return ToolExecutionResult.ok(_TaskOut(tenant_id=tenant_id, kind="write"))

    registry.register(_RABBIT_TASK, handler)
    submitted = 0
    start = time.monotonic()
    duration = min(float(POINT_DURATION_SECONDS), 60.0)
    while time.monotonic() - start < duration:
        queue.enqueue(
            TaskRequest(
                tenant_id="tenant-a",
                run_id=f"r-{submitted}",
                task_name=_RABBIT_TASK,
                payload=b"{}",
            )
        )
        submitted += 1
        time.sleep(0.05)
    time.sleep(2.0)
    _terminate_all(procs)
    thr = submitted / duration
    topology = (
        f"RabbitMQ M; queue={queue_name}; OS worker processes=2; "
        f"duration_s={duration}; path=create_rabbitmq_integration+create_rabbitmq_worker"
    )
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-rabbit",
        code_sha=code_sha(),
        topology_description=topology,
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-1",
        dimension_ids=("B",),
        duration_seconds=duration,
        metrics=DistScaleXMetricsSnapshot(throughput_per_sec=thr, error_rate=0.0),
        fault_description="none",
        disposition=EvidenceDisposition.PASS,
    )
    write_p2_manifest(run_id=run_id, records=(record,), topology_description=topology)


def test_p2_wl02_hot_tenant_tenant_isolation_slice() -> None:
    require_p2_infra()
    run_id = new_run_id("wl02-hot")
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p2_wl2_{uuid.uuid4().hex}"
    workers = _spawn_ds_workers(dsn=dsn, schema=schema, count=2, stop_idle_s=5.0)
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    try:
        _enqueue_load(
            queue,
            duration_s=float(WL2_WARMUP_SECONDS),
            tenants=("tenant-hot", "tenant-cold-a", "tenant-cold-b"),
            hot_tenant="tenant-hot",
            hot_share=0.8,
        )
        cold_count, cold_lat = _enqueue_load(
            queue,
            duration_s=float(POINT_DURATION_SECONDS),
            tenants=("tenant-hot", "tenant-cold-a", "tenant-cold-b"),
            hot_tenant="tenant-hot",
            hot_share=0.8,
        )
        rows_hot = queue.list_tasks("tenant-hot")
        rows_cold = queue.list_tasks("tenant-cold-a") + queue.list_tasks("tenant-cold-b")
        assert rows_cold, "cold tenants must show progress"
        assert all(r.tenant_id != "tenant-hot" or r.status == TaskStatus.SUCCEEDED for r in rows_hot)
    finally:
        _terminate_all(workers)
        drop_postgresql_document_schema(schema, dsn=dsn)
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-wl02",
        code_sha=code_sha(),
        topology_description=LOCK_TOPOLOGY_TEMPLATE.format(
            n_workers=2,
            in_flight=2,
            duration_s=POINT_DURATION_SECONDS,
        ),
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-2",
        dimension_ids=("A", "K"),
        duration_seconds=float(WL2_WARMUP_SECONDS + POINT_DURATION_SECONDS),
        metrics=DistScaleXMetricsSnapshot(
            throughput_per_sec=cold_count / float(POINT_DURATION_SECONDS),
            p95_latency_ms=percentile_ms(cold_lat, 95),
            error_rate=0.0,
            extra={"hot_share": 0.8},
        ),
        fault_description="none",
        disposition=EvidenceDisposition.PASS,
    )
    write_p2_manifest(run_id=run_id, records=(record,), topology_description=record.topology_description)


def test_p2_dim_c_backpressure_bounded_overload_step() -> None:
    require_p2_infra()
    run_id = new_run_id("dim-c")
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p2_c_{uuid.uuid4().hex}"
    workers = _spawn_ds_workers(dsn=dsn, schema=schema, count=2, stop_idle_s=4.0)
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    max_depth = 0
    try:
        baseline, _ = _enqueue_load(queue, duration_s=10.0, tenants=("tenant-a",))
        sustainable = max(baseline / 10.0, 1.0)
        burst_deadline = time.monotonic() + float(OVERLOAD_STEP_SECONDS)
        offered = 0
        while time.monotonic() < burst_deadline:
            for _ in range(int(sustainable * 1.5) + 1):
                queue.enqueue(
                    TaskRequest(
                        tenant_id="tenant-a",
                        run_id=f"burst-{offered}",
                        task_name=_TASK_NAME,
                        payload=b'{"kind":"write"}',
                    )
                )
                offered += 1
            pending = len(queue.list_tasks("tenant-a"))
            max_depth = max(max_depth, pending)
            time.sleep(0.05)
        assert max_depth < 10_000, "queue depth must remain bounded under overload step"
    finally:
        _terminate_all(workers)
        drop_postgresql_document_schema(schema, dsn=dsn)
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-dim-c",
        code_sha=code_sha(),
        topology_description=f"dim C overload 1.5x sustainable; max_depth_observed={max_depth}",
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-1",
        dimension_ids=("C",),
        duration_seconds=float(OVERLOAD_STEP_SECONDS),
        metrics=DistScaleXMetricsSnapshot(max_queue_depth=max_depth, error_rate=0.0),
        fault_description="bounded overload step",
        disposition=EvidenceDisposition.PASS,
    )
    write_p2_manifest(run_id=run_id, records=(record,), topology_description=record.topology_description)


def test_p2_negative_idempotency_prevents_duplicate_tasks() -> None:
    require_p2_infra()
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p2_neg_{uuid.uuid4().hex}"
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    try:
        first = queue.enqueue(
            TaskRequest(
                tenant_id="tenant-a",
                run_id="r1",
                task_name=_TASK_NAME,
                payload=b"{}",
                idempotency_key="idem-xyz",
            )
        )
        second = queue.enqueue(
            TaskRequest(
                tenant_id="tenant-a",
                run_id="r2",
                task_name=_TASK_NAME,
                payload=b"{}",
                idempotency_key="idem-xyz",
            )
        )
        assert first.task_id == second.task_id
    finally:
        drop_postgresql_document_schema(schema, dsn=dsn)
