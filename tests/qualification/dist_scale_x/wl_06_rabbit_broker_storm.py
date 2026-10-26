# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X P3 WL-6 — RabbitMQ M via AMQP + canonical Celery RetryPolicy (B01)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._p3_support import (
    WL6_DRAIN_CAP_SECONDS,
    WL6_FAULT_SECONDS,
    new_run_id,
    p3_record,
    require_p3_infra,
    write_p3_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import DistScaleXMetricsSnapshot
from testing_support.dist_scale_x_p3_wl6_celery_app import (
    P3_CELERY_TASK,
    WL6_RETRY_POLICY,
    build_wl6_celery_app,
    read_wl6_celery_metrics,
    set_fault_injection,
)

pytestmark = [pytest.mark.qualification, pytest.mark.integration, pytest.mark.network]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_TENANTS = ("tenant-a", "tenant-b")


def _terminate_all(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def _spawn_celery_workers(
    *,
    session_dir: Path,
    worker_count: int,
) -> list[subprocess.Popen[bytes]]:
    env = os.environ.copy()
    env["DIST_SCALE_X_P3_WL6_STATE_DB"] = str(session_dir / "wl6_state.db")
    env["DIST_SCALE_X_P3_WL6_KV_DB"] = str(session_dir / "kv.db")
    procs: list[subprocess.Popen[bytes]] = []
    module = "testing_support.dist_scale_x_p3_wl6_celery_app"
    for idx in range(worker_count):
        procs.append(
            subprocess.Popen(
                [
                    _PYTHON,
                    "-m",
                    "celery",
                    "-A",
                    module,
                    "worker",
                    "-n",
                    f"dist-scale-x-wl6-{idx}-{uuid.uuid4().hex[:6]}@%h",
                    "--pool=solo",
                    "-c",
                    "1",
                    "--loglevel=warning",
                ],
                cwd=_REPO_ROOT,
                env=env,
            )
        )
    return procs


def test_p3_wl06_rabbit_m_celery_dispatcher_retry_storm() -> None:
    require_p3_infra()
    run_id = new_run_id("wl06-rabbit")
    session_dir = _REPO_ROOT / ".tmp" / "session" / "dist-scale-x-p3-wl6-rabbit" / run_id
    session_dir.mkdir(parents=True, exist_ok=True)
    state_db = session_dir / "wl6_state.db"
    kv_db = session_dir / "kv.db"
    for path in (state_db, kv_db):
        if path.is_file():
            path.unlink()

    set_fault_injection(state_db, True)
    procs = _spawn_celery_workers(session_dir=session_dir, worker_count=2)
    time.sleep(4.0)

    app = build_wl6_celery_app(state_db=state_db, kv_db=kv_db)
    logical_submissions = 0
    async_results = []
    target_submissions = 60
    fault_start = time.monotonic()
    while (
        logical_submissions < target_submissions
        and time.monotonic() - fault_start < WL6_FAULT_SECONDS
    ):
        tenant = _TENANTS[logical_submissions % len(_TENANTS)]
        logical_id = f"wl6-{logical_submissions:05d}"
        logical_submissions += 1
        payload = json.dumps({"logical_id": logical_id, "tenant_id": tenant}).encode("utf-8")
        async_results.append(
            app.send_task(
                "intergrax.execute",
                kwargs={
                    "logical_task_name": P3_CELERY_TASK,
                    "tenant_id": tenant,
                    "run_id": f"run-{logical_id}",
                    "payload": payload,
                    "idempotency_key": logical_id,
                },
            )
        )
        time.sleep(0.15)
    while time.monotonic() - fault_start < WL6_FAULT_SECONDS:
        time.sleep(0.25)

    set_fault_injection(state_db, False)
    drain_start = time.monotonic()

    while time.monotonic() - drain_start < WL6_DRAIN_CAP_SECONDS:
        metrics = read_wl6_celery_metrics(state_db)
        if metrics.get("semantic_effects", 0) >= logical_submissions:
            break
        time.sleep(1.0)

    _ = async_results

    _terminate_all(procs)
    metrics = read_wl6_celery_metrics(state_db)
    physical = metrics.get("physical_attempts", 0)
    dispatcher_retries = metrics.get("dispatcher_retries", 0)
    effects = metrics.get("semantic_effects", 0)
    amp = physical / float(max(logical_submissions, 1))

    assert logical_submissions > 40
    assert effects == logical_submissions, (
        f"semantic effects {effects} != logical submissions {logical_submissions}"
    )
    assert physical >= logical_submissions
    assert dispatcher_retries > 0, "canonical RetryPolicy must schedule retries under fault load"
    assert amp < 50.0

    topology = (
        "P3 WL-6 RabbitMQ M (AMQP broker); retry_owner=queueing.worker.dispatcher.RetryPolicy "
        f"via create_celery_worker_app; max_retries={WL6_RETRY_POLICY.max_retries}; "
        f"fault_window_s={WL6_FAULT_SECONDS}; drain_cap_s={WL6_DRAIN_CAP_SECONDS}; "
        "OS celery workers=2; no qualification test-loop redrive; "
        "create_rabbitmq_worker poll path documented non-retrying (auto_ack)"
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
        fault_description="20% RetryableHandlerError; Celery dispatcher retry/backoff",
        metrics=DistScaleXMetricsSnapshot(
            retry_amplification_factor=amp,
            extra={
                "logical_submissions": float(logical_submissions),
                "physical_attempts": float(physical),
                "dispatcher_retries_scheduled": float(dispatcher_retries),
                "semantic_effects": float(effects),
            },
        ),
    )
    write_p3_manifest(run_id=run_id, records=(record,), topology_description=topology)
