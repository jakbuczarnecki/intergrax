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
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.integrations.providers.message_bus.rabbitmq.bundle import create_rabbitmq_integration
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue
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
from tests.qualification.dist_scale_x.p2_broker_reconcile import (
    assert_broker_run_acceptance,
    reconcile_broker_task_handles,
)
from tests.qualification.dist_scale_x.p2_dim_c_overload import (
    assert_overload_acceptance,
    run_rate_controlled_overload_step,
)
from tests.qualification.dist_scale_x.p2_task_load import (
    P2_MAX_IN_FLIGHT,
    P2_TASK_PAYLOAD_BYTES,
    TaskLoadAccounting,
    assert_cross_tenant_queue_fail_closed,
    pending_task_count,
    run_document_store_completed_work_load,
)

pytestmark = [
    pytest.mark.qualification,
    pytest.mark.integration,
    pytest.mark.network,
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASK_NAME = "dist_scale_x.p2.task"
_RABBIT_TASK = "dist_scale_x.p2.rabbit"
_PYTHON = sys.executable
_RABBIT_DRAIN_SECONDS = 120.0


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


def _assert_workers_alive(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is not None:
            raise AssertionError(f"worker process exited early with code {proc.returncode}")


def _measure_point(n_workers: int, duration_s: float) -> tuple[TaskLoadAccounting, str]:
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p2_{uuid.uuid4().hex}"
    require_p2_infra()
    workers = _spawn_ds_workers(dsn=dsn, schema=schema, count=n_workers, stop_idle_s=6.0)
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    try:
        accounting = run_document_store_completed_work_load(
            queue,
            duration_s=duration_s,
            tenants=("tenant-a", "tenant-b"),
            task_name=_TASK_NAME,
            max_in_flight=P2_MAX_IN_FLIGHT,
            worker_health=lambda: _assert_workers_alive(workers),
        )
        accounting.assert_acceptance_reconciled()
    finally:
        _terminate_all(workers)
        time.sleep(3.0)
        drop_postgresql_document_schema(schema, dsn=dsn)
    return accounting, schema


def test_p2_wl01_l4_document_store_scaling_envelope() -> None:
    require_p2_infra()
    run_id = new_run_id("wl01-scale")
    duration = float(POINT_DURATION_SECONDS)
    n1, _ = _measure_point(1, duration)
    time.sleep(15.0)
    n2, _ = _measure_point(2, duration)
    thr1 = n1.useful_throughput_per_sec(duration)
    thr2 = n2.useful_throughput_per_sec(duration)
    eff = thr2 / (2.0 * thr1) if thr1 > 0 else 0.0
    topology = LOCK_TOPOLOGY_TEMPLATE.format(
        n_workers=2,
        in_flight=P2_MAX_IN_FLIGHT,
        duration_s=POINT_DURATION_SECONDS,
    )
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-wl01",
        code_sha=code_sha(),
        topology_description=topology,
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-1",
        dimension_ids=("A", "B"),
        duration_seconds=duration * 2,
        metrics=DistScaleXMetricsSnapshot(
            throughput_per_sec=thr2,
            p95_latency_ms=percentile_ms(n2.completion_latencies_ms, 95),
            error_rate=0.0,
            extra={
                "throughput_n1_completed": thr1,
                "throughput_n2_completed": thr2,
                "scaling_efficiency": eff,
                "submitted_n1": float(n1.submitted),
                "completed_n1": float(n1.completed),
                "submitted_n2": float(n2.submitted),
                "completed_n2": float(n2.completed),
                "failed_n2": float(n2.failed),
                "timed_out_n2": float(n2.timed_out),
                "payload_bytes": float(P2_TASK_PAYLOAD_BYTES),
            },
        ),
        fault_description="none",
        disposition=EvidenceDisposition.PASS,
    )
    path = write_p2_manifest(run_id=run_id, records=(record,), topology_description=topology)
    assert path.is_file()
    assert thr2 > 0.0


def test_p2_wl01_rabbit_m_two_os_worker_processes() -> None:
    require_p2_infra()
    run_id = new_run_id("wl01-rabbit")
    queue_name = f"dist-scale-x-p2-{uuid.uuid4().hex[:10]}"
    metrics_dir = _REPO_ROOT / ".tmp" / "session" / "dist-scale-x-p2-rabbit" / run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    kv_db = metrics_dir / "broker_kv.db"
    duration = float(POINT_DURATION_SECONDS)
    worker_idle_s = duration + _RABBIT_DRAIN_SECONDS + 30.0
    module = "tests.qualification.dist_scale_x.p2_rabbit_multiprocess_worker"
    procs = [
        subprocess.Popen(
            [
                _PYTHON,
                "-m",
                module,
                "--queue",
                queue_name,
                "--stop-idle-s",
                str(worker_idle_s),
                "--kv-db",
                str(kv_db),
            ],
            cwd=_REPO_ROOT,
        )
        for _ in range(2)
    ]
    time.sleep(1.5)
    kv = SqliteDistributedKVStore(kv_db)
    bundle = create_rabbitmq_integration(
        kv_store=kv,
        host="localhost",
        username="intergrax",
        password="intergrax",
        queue=queue_name,
    )
    queue = bundle.message_bus
    handles: list = []
    start = time.monotonic()
    while time.monotonic() - start < duration:
        _assert_workers_alive(procs)
        handles.append(
            queue.enqueue(
                TaskRequest(
                    tenant_id="tenant-a",
                    run_id=f"r-{len(handles)}",
                    task_name=_RABBIT_TASK,
                    payload=b"x" * P2_TASK_PAYLOAD_BYTES,
                )
            )
        )
        time.sleep(0.05)
    submitted = len(handles)
    drain_deadline = time.monotonic() + _RABBIT_DRAIN_SECONDS
    while time.monotonic() < drain_deadline:
        _assert_workers_alive(procs)
        broker_partial = reconcile_broker_task_handles(queue, handles)
        if broker_partial.pending == 0 and broker_partial.succeeded >= submitted:
            break
        time.sleep(0.25)
    _terminate_all(procs)
    broker = reconcile_broker_task_handles(queue, handles)
    assert_broker_run_acceptance(broker)
    thr = broker.succeeded / duration
    topology = (
        f"RabbitMQ M; queue={queue_name}; OS worker processes=2; "
        f"duration_s={duration}; path=create_rabbitmq_integration+create_rabbitmq_worker; "
        f"terminal_evidence=BrokerBackedTaskQueue KV after process_message (auto_ack consume)"
    )
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-rabbit",
        code_sha=code_sha(),
        topology_description=topology,
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-1",
        dimension_ids=("B",),
        duration_seconds=duration,
        metrics=DistScaleXMetricsSnapshot(
            throughput_per_sec=thr,
            error_rate=0.0,
            extra={
                "submitted": float(broker.submitted),
                "broker_terminal_succeeded": float(broker.succeeded),
                "broker_terminal_failed": float(broker.failed),
                "broker_pending_after_drain": float(broker.pending),
            },
        ),
        fault_description="none",
        disposition=EvidenceDisposition.PASS,
    )
    write_p2_manifest(run_id=run_id, records=(record,), topology_description=topology)


def test_p2_b02_broker_kv_pending_without_worker_negative() -> None:
    """Without consumers, broker KV must remain non-terminal (no false PASS)."""
    require_p2_infra()
    queue_name = f"dist-scale-x-p2-neg-{uuid.uuid4().hex[:10]}"
    kv_db = _REPO_ROOT / ".tmp" / "session" / "dist-scale-x-p2-rabbit" / "negative-kv.db"
    kv_db.parent.mkdir(parents=True, exist_ok=True)
    if kv_db.is_file():
        kv_db.unlink()
    kv = SqliteDistributedKVStore(kv_db)
    bundle = create_rabbitmq_integration(
        kv_store=kv,
        host="localhost",
        username="intergrax",
        password="intergrax",
        queue=queue_name,
    )
    handle = bundle.message_bus.enqueue(
        TaskRequest(
            tenant_id="tenant-a",
            run_id="neg-1",
            task_name=_RABBIT_TASK,
            payload=b"{}",
        )
    )
    accounting = reconcile_broker_task_handles(bundle.message_bus, [handle])
    assert accounting.pending == 1
    assert accounting.succeeded == 0
    with pytest.raises(AssertionError, match="outstanding"):
        assert_broker_run_acceptance(accounting)


def test_p2_wl02_hot_tenant_tenant_isolation_slice() -> None:
    require_p2_infra()
    run_id = new_run_id("wl02-hot")
    dsn = resolve_pg_dsn()
    schema = f"dist_scale_x_p2_wl2_{uuid.uuid4().hex}"
    tenants = ("tenant-hot", "tenant-cold-a", "tenant-cold-b")
    workers = _spawn_ds_workers(dsn=dsn, schema=schema, count=2, stop_idle_s=8.0)
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    try:
        warmup = run_document_store_completed_work_load(
            queue,
            duration_s=float(WL2_WARMUP_SECONDS),
            tenants=tenants,
            task_name=_TASK_NAME,
            hot_tenant="tenant-hot",
            hot_share=0.8,
            max_in_flight=P2_MAX_IN_FLIGHT,
            worker_health=lambda: _assert_workers_alive(workers),
        )
        warmup.assert_acceptance_reconciled()
        measure = run_document_store_completed_work_load(
            queue,
            duration_s=float(POINT_DURATION_SECONDS),
            tenants=tenants,
            task_name=_TASK_NAME,
            hot_tenant="tenant-hot",
            hot_share=0.8,
            max_in_flight=P2_MAX_IN_FLIGHT,
            worker_health=lambda: _assert_workers_alive(workers),
        )
        measure.assert_acceptance_reconciled()
        cold_completed = (
            measure.completions_by_tenant.get("tenant-cold-a", 0)
            + measure.completions_by_tenant.get("tenant-cold-b", 0)
        )
        assert cold_completed > 0, "cold tenants must complete work during hot window"
        assert_cross_tenant_queue_fail_closed(
            queue,
            owner_tenant="tenant-cold-a",
            victim_tenant="tenant-hot",
        )
        rows_hot = queue.list_tasks("tenant-hot")
        assert all(r.tenant_id == "tenant-hot" for r in rows_hot)
    finally:
        _terminate_all(workers)
        drop_postgresql_document_schema(schema, dsn=dsn)
    cold_thr = cold_completed / float(POINT_DURATION_SECONDS)
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-wl02",
        code_sha=code_sha(),
        topology_description=LOCK_TOPOLOGY_TEMPLATE.format(
            n_workers=2,
            in_flight=P2_MAX_IN_FLIGHT,
            duration_s=POINT_DURATION_SECONDS,
        ),
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-2",
        dimension_ids=("A", "K"),
        duration_seconds=float(WL2_WARMUP_SECONDS + POINT_DURATION_SECONDS),
        metrics=DistScaleXMetricsSnapshot(
            throughput_per_sec=cold_thr,
            p95_latency_ms=percentile_ms(measure.completion_latencies_ms, 95),
            error_rate=0.0,
            extra={
                "hot_share": 0.8,
                "cold_completed": float(cold_completed),
                "completed_total": float(measure.completed),
            },
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
    workers = _spawn_ds_workers(dsn=dsn, schema=schema, count=2, stop_idle_s=8.0)
    time.sleep(1.0)
    store = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    queue = DocumentStoreTaskQueue(store)
    overload = None
    sustainable = 0.0
    try:
        baseline = run_document_store_completed_work_load(
            queue,
            duration_s=float(POINT_DURATION_SECONDS),
            tenants=("tenant-a", "tenant-b"),
            task_name=_TASK_NAME,
            max_in_flight=P2_MAX_IN_FLIGHT,
            worker_health=lambda: _assert_workers_alive(workers),
        )
        baseline.assert_acceptance_reconciled()
        sustainable = baseline.useful_throughput_per_sec(float(POINT_DURATION_SECONDS))
        assert sustainable > 0.0
        overload = run_rate_controlled_overload_step(
            queue,
            sustainable_completed_per_sec=sustainable,
            duration_s=float(OVERLOAD_STEP_SECONDS),
            task_name=_TASK_NAME,
        )
        assert_overload_acceptance(overload)
        drain_deadline = time.monotonic() + 60.0
        while time.monotonic() < drain_deadline:
            _assert_workers_alive(workers)
            depth = pending_task_count(queue, "tenant-a") + pending_task_count(queue, "tenant-b")
            overload.depth_samples.append(depth)
            if depth == 0:
                break
            time.sleep(0.25)
        overload.depth_after_drain = pending_task_count(queue, "tenant-a") + pending_task_count(
            queue,
            "tenant-b",
        )
        assert overload.depth_after_drain < overload.depth_before_drain or overload.depth_after_drain == 0, (
            "queue must drain or recover after overload step"
        )
    finally:
        _terminate_all(workers)
        drop_postgresql_document_schema(schema, dsn=dsn)
    assert overload is not None
    record = DistScaleXEvidenceRecord(
        record_id=f"{run_id}-dim-c",
        code_sha=code_sha(),
        topology_description=(
            f"dim C: N=2 baseline {POINT_DURATION_SECONDS}s; target_offered={overload.target_offered_per_sec:.3f}/s; "
            f"actual_offered={overload.actual_offered_per_sec:.3f}/s; step={OVERLOAD_STEP_SECONDS}s; "
            f"depth_samples={len(overload.depth_samples)}"
        ),
        environment_definition=LOCK_ENV_DEFINITION,
        workload_id="WL-1",
        dimension_ids=("C",),
        duration_seconds=float(POINT_DURATION_SECONDS + OVERLOAD_STEP_SECONDS),
        metrics=DistScaleXMetricsSnapshot(
            max_queue_depth=overload.max_queue_depth,
            error_rate=0.0,
            extra={
                "sustainable_completed_per_sec": sustainable,
                "target_offered_per_sec": overload.target_offered_per_sec,
                "actual_offered_per_sec": overload.actual_offered_per_sec,
                "submitted_overload": float(overload.submitted),
                "completed_terminal_overload": float(overload.completed_terminal),
                "outstanding_after_overload": float(overload.outstanding_after_step),
                "depth_before_drain": float(overload.depth_before_drain),
                "depth_after_drain": float(overload.depth_after_drain),
            },
        ),
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
