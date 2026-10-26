# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P3 WL-6 RabbitMQ M retry storm (mandatory broker path)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.integrations.providers.message_bus.rabbitmq.bundle import create_rabbitmq_integration
from intergrax.queueing.contracts.task_queue import TaskRequest, TaskStatus
from tests.qualification.dist_scale_x._p3_support import (
    WL6_DRAIN_CAP_SECONDS,
    WL6_FAULT_SECONDS,
    new_run_id,
    p3_record,
    require_p3_infra,
    write_p3_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot
from tests.qualification.dist_scale_x.p2_broker_reconcile import reconcile_broker_task_handles
from tests.qualification.dist_scale_x.p3_rabbit_multiprocess_worker import (
    P3_RABBIT_TASK,
    read_wl6_metrics,
    set_fault_injection,
)

pytestmark = [pytest.mark.qualification, pytest.mark.integration, pytest.mark.network]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_TENANTS = ("tenant-a", "tenant-b")


@dataclass
class _LogicalWork:
    logical_id: str
    tenant_id: str
    handles: list = field(default_factory=list)


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
            raise AssertionError(f"rabbit worker exited early code={proc.returncode}")


def _enqueue(
    queue,
    work: _LogicalWork,
) -> None:
    payload = json.dumps(
        {"logical_id": work.logical_id, "tenant_id": work.tenant_id},
    ).encode("utf-8")
    handle = queue.enqueue(
        TaskRequest(
            tenant_id=work.tenant_id,
            run_id=f"{work.logical_id}-{len(work.handles)}",
            task_name=P3_RABBIT_TASK,
            payload=payload,
        )
    )
    work.handles.append(handle)


def _latest_status(queue, work: _LogicalWork) -> TaskStatus:
    handle = work.handles[-1]
    return queue.get_status(handle)


def _redrive_failed(queue, work: _LogicalWork) -> bool:
    status = _latest_status(queue, work)
    if status is TaskStatus.SUCCEEDED:
        return False
    if status is TaskStatus.FAILED:
        _enqueue(queue, work)
        return True
    return False


def test_p3_wl06_rabbit_m_retry_storm_fault_window_and_drain() -> None:
    require_p3_infra()
    run_id = new_run_id("wl06-rabbit")
    queue_name = f"dist-scale-x-p3-wl6-{uuid.uuid4().hex[:10]}"
    session_dir = _REPO_ROOT / ".tmp" / "session" / "dist-scale-x-p3-wl6-rabbit" / run_id
    session_dir.mkdir(parents=True, exist_ok=True)
    kv_db = session_dir / "broker_kv.db"
    state_db = session_dir / "wl6_state.db"
    if kv_db.is_file():
        kv_db.unlink()
    set_fault_injection(state_db, True)

    worker_idle_s = WL6_FAULT_SECONDS + WL6_DRAIN_CAP_SECONDS + 90.0
    module = "tests.qualification.dist_scale_x.p3_rabbit_multiprocess_worker"
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
                "--state-db",
                str(state_db),
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

    works: dict[str, _LogicalWork] = {}
    logical_counter = 0
    fault_start = time.monotonic()
    while time.monotonic() - fault_start < WL6_FAULT_SECONDS:
        _assert_workers_alive(procs)
        tenant = _TENANTS[logical_counter % len(_TENANTS)]
        logical_id = f"wl6-{logical_counter:06d}"
        logical_counter += 1
        work = _LogicalWork(logical_id=logical_id, tenant_id=tenant)
        works[logical_id] = work
        _enqueue(queue, work)
        for work_item in works.values():
            _redrive_failed(queue, work_item)
        time.sleep(0.04)

    logical_submissions = len(works)
    set_fault_injection(state_db, False)

    drain_start = time.monotonic()
    while time.monotonic() - drain_start < WL6_DRAIN_CAP_SECONDS:
        _assert_workers_alive(procs)
        pending = 0
        for work_item in works.values():
            status = _latest_status(queue, work_item)
            if status is TaskStatus.SUCCEEDED:
                continue
            pending += 1
            _redrive_failed(queue, work_item)
        if pending == 0:
            break
        time.sleep(0.25)

    latest_handles = [w.handles[-1] for w in works.values()]
    broker = reconcile_broker_task_handles(queue, latest_handles)
    succeeded_logical = sum(
        1 for w in works.values() if _latest_status(queue, w) is TaskStatus.SUCCEEDED
    )
    metrics = read_wl6_metrics(state_db)
    physical_attempts = metrics["physical_attempts"]
    semantic_effects = metrics["semantic_effects"]
    amp = physical_attempts / float(max(logical_submissions, 1))

    _terminate_all(procs)

    assert logical_submissions > 50, "insufficient logical submissions during fault window"
    assert succeeded_logical == logical_submissions, (
        f"unresolved logical work: {logical_submissions - succeeded_logical}"
    )
    assert semantic_effects == logical_submissions, (
        f"duplicate or missing effects: effects={semantic_effects} logical={logical_submissions}"
    )
    assert physical_attempts >= logical_submissions
    assert amp < 50.0, f"unbounded retry amplification: {amp}"
    assert broker.pending == 0
    assert broker.succeeded == logical_submissions

    topology = (
        f"P3 WL-6 RabbitMQ M; queue={queue_name}; OS workers=2; "
        f"fault_window_s={WL6_FAULT_SECONDS}; drain_cap_s={WL6_DRAIN_CAP_SECONDS}; "
        f"fault_rate=20% RetryableHandlerError; path=create_rabbitmq_integration+create_rabbitmq_worker; "
        f"redrive=broker FAILED re-enqueue (client redelivery); tenants>=2"
    )
    record = p3_record(
        run_id=run_id,
        suffix="wl06-rabbit",
        workload_id="WL-6",
        dimension_ids=("C", "E"),
        duration_seconds=WL6_FAULT_SECONDS + min(
            WL6_DRAIN_CAP_SECONDS,
            time.monotonic() - drain_start,
        ),
        topology_description=topology,
        fault_description="20% transient RetryableHandlerError during fault window",
        metrics=DistScaleXMetricsSnapshot(
            retry_amplification_factor=amp,
            extra={
                "logical_submissions": float(logical_submissions),
                "physical_attempts": float(physical_attempts),
                "semantic_effects": float(semantic_effects),
                "broker_terminal_succeeded": float(broker.succeeded),
                "broker_terminal_failed": float(broker.failed),
                "broker_pending": float(broker.pending),
            },
        ),
    )
    write_p3_manifest(run_id=run_id, records=(record,), topology_description=topology)
