# © Artur Czarnecki. All rights reserved.

"""DIST-SCALE-X WL-5 persistence-heavy (P4-A) — full envelope."""

from __future__ import annotations

import json
import subprocess
import sys
import time
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
    prime_p4_qualification_stores,
    require_p4_pg_infra,
    resolve_pg_dsn,
    write_p4_manifest,
)
from tests.qualification.dist_scale_x.evidence_schema import (
    CorrectnessInvariantViolation,
    DistScaleXEvidenceRecord,
    DistScaleXMetricsSnapshot,
    EvidenceDisposition,
)
from tests.qualification.dist_scale_x.p4_wl5_gate import Wl5RateGate

pytestmark = pytest.mark.qualification

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTHON = sys.executable
_TENANTS = ("tenant-a", "tenant-b")
_READ_SHARE_TOLERANCE = 0.05


def _spawn_workers(
    *,
    dsn: str,
    schema: str,
    gate_db: Path,
    stop_file: Path,
    metrics_dir: Path,
    phase_file: Path,
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
                    "--gate-db",
                    str(gate_db),
                    "--writers",
                    str(WL5_WRITERS),
                    "--phase-file",
                    str(phase_file),
                ],
                cwd=_REPO_ROOT,
            ),
        )
    return procs


def _offer_phase(gate: Wl5RateGate, duration_s: float) -> int:
    interval = 1.0 / WL5_WRITE_RATE_CAP
    deadline = time.monotonic() + duration_s
    seq = 0
    next_offer = time.monotonic()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now < next_offer:
            time.sleep(min(0.01, next_offer - now))
            continue
        gate.offer(seq)
        seq += 1
        next_offer += interval
    return seq


def _drain_gate(gate: Wl5RateGate, timeout_s: float = 45.0) -> None:
    deadline = time.monotonic() + timeout_s
    while gate.pending_offers() > 0 and time.monotonic() < deadline:
        time.sleep(0.1)


def _collect_metrics(metrics_dir: Path) -> dict[str, dict[str, float]]:
    warmup = {"writes": 0.0, "reads": 0.0, "errors": 0.0}
    measure = {"writes": 0.0, "reads": 0.0, "errors": 0.0}
    for path in metrics_dir.glob("worker_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for phase, bucket in (("warmup", warmup), ("measure", measure)):
            phase_data = payload.get(phase, {})
            bucket["writes"] += float(phase_data.get("writes", 0))
            bucket["reads"] += float(phase_data.get("reads", 0))
            bucket["errors"] += float(phase_data.get("errors", 0))
    return {"warmup": warmup, "measure": measure}


def _run_wl5_phase(
    *,
    gate: Wl5RateGate,
    duration_s: float,
) -> int:
    offered = _offer_phase(gate, duration_s)
    _drain_gate(gate)
    return offered


def test_wl05_persistence_heavy_baseline() -> None:
    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    run_id = new_run_id("wl05-p4")
    schema = qual_pg_schema(run_id)
    prime_p4_qualification_stores(dsn, schema)
    session_dir = _REPO_ROOT / ".tmp" / "session" / f"p4-wl05-{run_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    gate_db = session_dir / "gate.db"
    stop_file = session_dir / "stop"
    phase_file = session_dir / "phase"
    metrics_dir = session_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    if stop_file.exists():
        stop_file.unlink()
    if phase_file.exists():
        phase_file.unlink()
    phase_file.write_text("warmup", encoding="utf-8")
    gate = Wl5RateGate(gate_db)
    procs = _spawn_workers(
        dsn=dsn,
        schema=schema,
        gate_db=gate_db,
        stop_file=stop_file,
        metrics_dir=metrics_dir,
        phase_file=phase_file,
    )
    try:
        warmup_offered = _run_wl5_phase(gate=gate, duration_s=float(WL5_WARMUP_SECONDS))
        phase_file.write_text("measure", encoding="utf-8")
        measure_offered = _run_wl5_phase(gate=gate, duration_s=float(WL5_MEASURE_SECONDS))
        stop_file.write_text("stop", encoding="utf-8")
        for proc in procs:
            proc.wait(timeout=60)
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
    for proc in procs:
        assert proc.returncode == 0, f"worker exit {proc.returncode}"
    phase_metrics = _collect_metrics(metrics_dir)
    warmup_metrics = phase_metrics["warmup"]
    measure_metrics = phase_metrics["measure"]
    total_writes = measure_metrics["writes"]
    total_reads = measure_metrics["reads"]
    total_ops = total_writes + total_reads
    read_share = total_reads / total_ops if total_ops else 0.0
    target_read = 0.10
    violations: list[str] = []
    if measure_metrics["errors"] > 0:
        violations.append(f"errors={measure_metrics['errors']}")
    if total_ops > 50 and abs(read_share - target_read) > _READ_SHARE_TOLERANCE:
        violations.append(f"read_share={read_share:.3f}")
    correctness = (CorrectnessInvariantViolation.LOST_EVIDENCE,) if violations else ()
    manifest_path = write_p4_manifest(
        run_id=run_id,
        records=(
            DistScaleXEvidenceRecord(
                record_id=f"{run_id}-wl5-base",
                code_sha=code_sha(),
                topology_description=(
                    f"P4 WL-5 PG M; schema={schema}; writers={WL5_WRITERS}; "
                    f"worker_pids={[p.pid for p in procs]}"
                ),
                environment_definition=P4_ENV_DEFINITION,
                workload_id="WL-5",
                dimension_ids=("A",),
                duration_seconds=float(WL5_WARMUP_SECONDS + WL5_MEASURE_SECONDS),
                fault_description="baseline fault rate 0%",
                disposition=EvidenceDisposition.PASS if not correctness else EvidenceDisposition.FAIL,
                metrics=DistScaleXMetricsSnapshot(
                    throughput_per_sec=total_ops / max(1.0, float(WL5_MEASURE_SECONDS)),
                    error_rate=measure_metrics["errors"] / max(1.0, total_ops),
                    extra={
                        "warmup_offered": float(warmup_offered),
                        "measure_offered": float(measure_offered),
                        "warmup_writes": warmup_metrics["writes"],
                        "warmup_reads": warmup_metrics["reads"],
                        "measure_writes": total_writes,
                        "measure_reads": total_reads,
                        "measure_read_share": read_share,
                        "warmup_errors": warmup_metrics["errors"],
                        "measure_errors": measure_metrics["errors"],
                        "write_rate_cap": WL5_WRITE_RATE_CAP,
                    },
                ),
                correctness_violations=correctness,
            ),
        ),
        topology_description=f"P4 WL-5; tenants={_TENANTS}; session={session_dir.name}",
    )
    assert manifest_path.exists()
    assert not violations, violations
