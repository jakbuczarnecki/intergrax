# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X WL-5 persistence-heavy (P4-A)."""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from tests.qualification.dist_scale_x._p4_support import (
    P4_ENV_DEFINITION,
    WL5_MEASURE_SECONDS,
    WL5_PAYLOAD_BYTES,
    WL5_WARMUP_SECONDS,
    WL5_WRITE_RATE_CAP,
    WL5_WRITERS,
    code_sha,
    new_run_id,
    qual_pg_schema,
    require_p4_pg_infra,
    resolve_pg_dsn,
    write_p3_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import (
    CorrectnessInvariantViolation,
    DistScaleXEvidenceRecord,
    DistScaleXMetricsSnapshot,
    EvidenceDisposition,
)
from tests.qualification.dist_scale_x.p4_persistence_op import (
    execute_p4_persistence_op,
    mint_qual_task_id,
)

pytestmark = pytest.mark.qualification

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_TENANTS = ("tenant-a", "tenant-b")


def _rate_limited_driver(
    *,
    dsn: str,
    schema: str,
    duration_s: float,
    writers: int,
    rate_cap: float,
    payload_bytes: int,
) -> tuple[int, int, int, list[str]]:
    """Aggregate driver: caps offered ops/s across concurrent writers."""
    pad = "w" * payload_bytes
    offered = 0
    completed = 0
    errors = 0
    violations: list[str] = []
    min_interval = 1.0 / rate_cap if rate_cap > 0 else 0.0
    deadline = time.monotonic() + duration_s
    next_tick = time.monotonic()
    task_by_tenant: dict[str, str] = {}

    def _one_op(tenant: str, index: int) -> None:
        nonlocal completed, errors
        tid = task_by_tenant.get(tenant)
        if index % 10 != 0 or tid is None:
            tid = mint_qual_task_id()
            task_by_tenant[tenant] = tid
        result = execute_p4_persistence_op(
            dsn=dsn,
            schema=schema,
            tenant_id=tenant,
            op_index=index,
            payload_pad=pad,
            task_id=tid if index % 10 == 0 else None,
        )
        if result.error:
            errors += 1
            violations.append(result.error)
        else:
            completed += 1
            if result.kind == "write":
                task_by_tenant[tenant] = result.task_id

    futures = []
    with ThreadPoolExecutor(max_workers=writers) as pool:
        tick = 0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            tenant = _TENANTS[tick % len(_TENANTS)]
            offered += 1
            futures.append(pool.submit(_one_op, tenant, tick))
            tick += 1
            next_tick += min_interval
        for future in as_completed(futures):
            future.result()
    return offered, completed, errors, violations


def _spawn_os_workers(
    *,
    dsn: str,
    schema: str,
    stop_file: Path,
    metrics_dir: Path,
    duration_s: float,
) -> list[subprocess.Popen[bytes]]:
    module = "tests.qualification.dist_scale_x.p4_multiprocess_worker"
    procs: list[subprocess.Popen[bytes]] = []
    for idx, tenant in enumerate(_TENANTS[:WL5_WRITERS]):
        metrics = metrics_dir / f"worker_{idx}.json"
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
                    "--tenant-id",
                    tenant,
                    "--payload-bytes",
                    str(WL5_PAYLOAD_BYTES),
                    "--stop-file",
                    str(stop_file),
                    "--metrics-file",
                    str(metrics),
                    "--worker-index",
                    str(idx),
                ],
                cwd=_REPO_ROOT,
            ),
        )
    time.sleep(duration_s)
    stop_file.write_text("stop", encoding="utf-8")
    return procs


def _terminate(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_wl05_persistence_heavy_baseline() -> None:
    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    run_id = new_run_id("wl05-p4")
    schema = qual_pg_schema(run_id)
    offered_w, completed_w, errors_w, viol_w = _rate_limited_driver(
        dsn=dsn,
        schema=schema,
        duration_s=float(WL5_WARMUP_SECONDS),
        writers=WL5_WRITERS,
        rate_cap=WL5_WRITE_RATE_CAP,
        payload_bytes=WL5_PAYLOAD_BYTES,
    )
    offered_m, completed_m, errors_m, viol_m = _rate_limited_driver(
        dsn=dsn,
        schema=schema,
        duration_s=float(WL5_MEASURE_SECONDS),
        writers=WL5_WRITERS,
        rate_cap=WL5_WRITE_RATE_CAP,
        payload_bytes=WL5_PAYLOAD_BYTES,
    )
    tmp = _REPO_ROOT / ".tmp" / "session" / f"p4-wl05-{run_id}"
    tmp.mkdir(parents=True, exist_ok=True)
    stop_file = tmp / "stop"
    metrics_dir = tmp / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    procs = _spawn_os_workers(
        dsn=dsn,
        schema=schema,
        stop_file=stop_file,
        metrics_dir=metrics_dir,
        duration_s=min(5.0, float(WL5_MEASURE_SECONDS)),
    )
    _terminate(procs)

    violations = tuple(viol_w + viol_m)
    correctness = ()
    if violations:
        correctness = (CorrectnessInvariantViolation.LOST_EVIDENCE,)
    manifest_path = write_p3_manifest(
        run_id=run_id,
        records=(
            DistScaleXEvidenceRecord(
                record_id=f"{run_id}-wl5-base",
                code_sha=code_sha(),
                topology_description=f"P4 WL-5 PG M; schema={schema}; writers={WL5_WRITERS}",
                environment_definition=P4_ENV_DEFINITION,
                workload_id="WL-5",
                dimension_ids=("A",),
                duration_seconds=float(WL5_WARMUP_SECONDS + WL5_MEASURE_SECONDS),
                fault_description="baseline fault rate 0%",
                disposition=EvidenceDisposition.PASS
                if not correctness
                else EvidenceDisposition.FAIL,
                metrics=DistScaleXMetricsSnapshot(
                    throughput_per_sec=(completed_w + completed_m)
                    / max(1.0, float(WL5_WARMUP_SECONDS + WL5_MEASURE_SECONDS)),
                    error_rate=(errors_w + errors_m)
                    / max(1.0, float(offered_w + offered_m)),
                    extra={
                        "offered_operations": float(offered_w + offered_m),
                        "completed_operations": float(completed_w + completed_m),
                        "os_worker_processes": float(len(procs)),
                        "write_rate_cap": WL5_WRITE_RATE_CAP,
                    },
                ),
                correctness_violations=correctness,
            ),
        ),
        topology_description=f"P4 WL-5 PG M; tenants={_TENANTS}; dsn=redacted",
    )
    assert manifest_path.exists()
    assert not violations, violations[:5]

