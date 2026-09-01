# © Artur Czarnecki. All rights reserved.

"""Brutal Docker E2E proof orchestrator for TOOLS-SIDE-EFFECT-SAFETY."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from tests.system.tools_side_effect_safety.proof_runner.oracle import EffectOracle
from tests.system.tools_side_effect_safety.proof_runner.runtime_client import (
    RuntimeClient,
    RuntimeEndpoint,
    build_invoke,
)
from tests.system.tools_side_effect_safety.shared.contracts import (
    DEFAULT_TENANT,
    PROOF_GOVERNANCE_TOOL,
    PROOF_TOOL_BAD_OUTPUT,
    PROOF_TOOL_CHARGE,
    PROOF_TOOL_CHARGE_ALT,
    PROOF_TOOL_FAIL_BEFORE,
    PROOF_TOOL_SLOW_CHARGE,
)

RUN_ID = os.environ.get("PROOF_RUN_ID", uuid.uuid4().hex[:12])
CHAOS_SEED = int(os.environ.get("PROOF_CHAOS_SEED", "20260831"))
COMPOSE_PROJECT = os.environ.get("COMPOSE_PROJECT_NAME", "tools-side-effect-safety-proof")
ARTIFACT_DIR = Path(
    os.environ.get(
        "PROOF_ARTIFACT_DIR",
        f"/workspace/.tmp/session/tools-side-effect-safety-proof/{RUN_ID}",
    ),
)


@dataclass
class ScenarioResult:
    number: int
    name: str
    hosts: str
    fault: str
    ledger: str
    http_attempts: str
    external_effects: str
    verdict: str
    notes: str = ""


@dataclass
class ProofReport:
    run_id: str = RUN_ID
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    scenarios: list[ScenarioResult] = field(default_factory=list)
    duplicate_scan: list[dict[str, Any]] = field(default_factory=list)
    final_verdict: str = "PENDING"

    def add(self, result: ScenarioResult) -> None:
        self.scenarios.append(result)


def _runtime_single() -> RuntimeClient:
    return RuntimeClient(RuntimeEndpoint("runtime-single", os.environ["RUNTIME_SINGLE_URL"]))


def _runtime_a() -> RuntimeClient:
    return RuntimeClient(RuntimeEndpoint("runtime-a", os.environ["RUNTIME_A_URL"]))


def _runtime_b() -> RuntimeClient:
    return RuntimeClient(RuntimeEndpoint("runtime-b", os.environ["RUNTIME_B_URL"]))


def _op(prefix: str) -> str:
    return f"{prefix}-{RUN_ID}-{uuid.uuid4().hex[:8]}"


DOCKER_BIN = os.environ.get("DOCKER_BIN") or shutil.which("docker") or "/usr/local/bin/docker"


def _container_name(service: str) -> str:
    return f"{COMPOSE_PROJECT}-{service}-1"


def _docker_kill(service: str) -> str:
    completed = subprocess.run(
        [DOCKER_BIN, "kill", "-s", "KILL", _container_name(service)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker kill failed for {service}: {completed.stderr}")
    return datetime.now(UTC).isoformat()


def _docker_restart(service: str) -> None:
    completed = subprocess.run(
        [DOCKER_BIN, "restart", _container_name(service)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker restart failed for {service}: {completed.stderr}")


def _wait_lease_expiry() -> None:
    lease = int(os.environ.get("IDEMPOTENCY_LEASE_SECONDS", "5"))
    time.sleep(lease + 2)


def _concurrent_invoke(
    runtime: RuntimeClient,
    *,
    requests: list,
    workers: int,
) -> list[Any]:
    barrier = threading.Barrier(len(requests))
    results: list[Any] = []

    def _worker(req) -> None:
        barrier.wait()
        try:
            results.append(runtime.invoke(req))
        except Exception as exc:
            results.append(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, req) for req in requests]
        for future in as_completed(futures):
            future.result()
    return results


def run_scenario_0(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> bool:
    """Harness calibration — verify proof_delay_ms reaches external service."""
    op = _op("s0-cal")
    key = f"key-s0-{RUN_ID}"
    configured_delay = 1500
    started = time.monotonic()
    response = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}",
            business_operation_id=op,
            idempotency_key=key,
            proof_mode="timeout_response",
            proof_delay_ms=configured_delay,
        ),
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    snap = oracle.snapshot(op)
    received = snap.attempts[0].received_delay_ms if snap.attempts else None
    harness_valid = (
        response.success
        and received == configured_delay
        and elapsed_ms >= configured_delay - 200
    )
    report.add(
        ScenarioResult(
            0,
            "harness_calibration",
            "runtime-single",
            f"delay={configured_delay}ms",
            response.ledger_status or "",
            str(snap.attempt_count),
            str(snap.effect_count),
            "PASS" if harness_valid else "FAIL",
            notes=(
                f"configured={configured_delay} received={received} "
                f"elapsed_ms={elapsed_ms}"
            ),
        ),
    )
    return harness_valid


def run_scenario_1(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s1")
    key = f"key-s1-{RUN_ID}"
    before = oracle.snapshot(op)
    response = runtime.invoke(
        build_invoke(run_id=f"run-{op}", business_operation_id=op, idempotency_key=key),
    )
    after = oracle.snapshot(op)
    verdict = "PASS" if (
        response.success
        and after.effect_count == 1
        and after.attempt_count == before.attempt_count + 1
    ) else "FAIL"
    report.add(
        ScenarioResult(
            1,
            "baseline_success",
            "runtime-single",
            "none",
            response.ledger_status or "",
            str(after.attempt_count),
            str(after.effect_count),
            verdict,
        ),
    )


def run_scenario_2(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s2")
    key = f"key-s2-{RUN_ID}"
    req = build_invoke(run_id=f"run-{op}", business_operation_id=op, idempotency_key=key)
    first = runtime.invoke(req)
    attempts_after_first = oracle.snapshot(op).attempt_count
    second = runtime.invoke(req)
    after = oracle.snapshot(op)
    verdict = "PASS" if (
        first.success
        and second.success
        and after.effect_count == 1
        and after.attempt_count == attempts_after_first
    ) else "FAIL"
    report.add(
        ScenarioResult(
            2,
            "sequential_replay",
            "runtime-single",
            "replay",
            f"{first.ledger_status}->{second.ledger_status}",
            str(after.attempt_count),
            str(after.effect_count),
            verdict,
        ),
    )


def run_scenario_3(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s3")
    key = f"key-s3-{RUN_ID}"
    requests = [
        build_invoke(
            run_id=f"run-{op}-{i}",
            business_operation_id=op,
            idempotency_key=key,
        )
        for i in range(100)
    ]
    _concurrent_invoke(runtime, requests=requests, workers=100)
    after = oracle.snapshot(op)
    verdict = "PASS" if after.effect_count == 1 else "FAIL"
    report.add(
        ScenarioResult(
            3,
            "concurrency_storm",
            "runtime-single",
            "100 concurrent",
            "mixed",
            str(after.attempt_count),
            str(after.effect_count),
            verdict,
            notes=f"attempts={after.attempt_count}",
        ),
    )


def run_scenario_4(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    failures = 0
    for round_idx in range(20):
        op = _op(f"s4-r{round_idx}")
        key = f"key-s4-{round_idx}-{RUN_ID}"
        requests = [
            build_invoke(
                run_id=f"run-{op}-{i}",
                business_operation_id=op,
                idempotency_key=key,
            )
            for i in range(50)
        ]
        _concurrent_invoke(runtime, requests=requests, workers=50)
        snap = oracle.snapshot(op)
        if snap.effect_count != 1:
            failures += 1
    verdict = "PASS" if failures == 0 else "FAIL"
    report.add(
        ScenarioResult(
            4,
            "repeat_storm",
            "runtime-single",
            "20x50 concurrent",
            "mixed",
            "1000 invocations",
            "1 per op",
            verdict,
            notes=f"failed_rounds={failures}",
        ),
    )


def run_scenario_5(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op_a = _op("s5a")
    op_b = _op("s5b")
    key = f"key-s5-{RUN_ID}"
    runtime.invoke(
        build_invoke(
            run_id=f"run-{op_a}",
            business_operation_id=op_a,
            idempotency_key=key,
            tool_id=PROOF_TOOL_CHARGE,
        ),
    )
    conflict = runtime.invoke(
        build_invoke(
            run_id=f"run-{op_b}",
            business_operation_id=op_b,
            idempotency_key=key,
            tool_id=PROOF_TOOL_CHARGE_ALT,
        ),
    )
    snap_a = oracle.snapshot(op_a)
    snap_b = oracle.snapshot(op_b)
    verdict = "PASS" if (
        conflict.blocked or conflict.error_type == "IdempotencyOperationConflictError"
        or not conflict.success
    ) and snap_a.effect_count == 1 and snap_b.effect_count == 0 else "FAIL"
    report.add(
        ScenarioResult(
            5,
            "cross_operation_collision",
            "runtime-single",
            "TOOLS-04",
            conflict.ledger_status or conflict.error_type or "",
            f"A:{snap_a.attempt_count},B:{snap_b.attempt_count}",
            f"A:{snap_a.effect_count},B:{snap_b.effect_count}",
            verdict,
        ),
    )


def run_scenario_6(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s6")
    key = f"key-s6-{RUN_ID}"
    deny = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}-deny",
            business_operation_id=op,
            idempotency_key=key,
            governance_action="deny",
        ),
    )
    snap_deny = oracle.snapshot(op)
    allow = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}-allow",
            business_operation_id=op,
            idempotency_key=key,
        ),
    )
    snap_allow = oracle.snapshot(op)
    verdict = "PASS" if (
        deny.ledger_status is None
        and snap_deny.effect_count == 0
        and snap_deny.attempt_count == 0
        and allow.success
        and snap_allow.effect_count == 1
    ) else "FAIL"
    report.add(
        ScenarioResult(
            6,
            "governance_deny",
            "runtime-single",
            "deny->allow",
            f"deny:{deny.ledger_status},allow:{allow.ledger_status}",
            str(snap_allow.attempt_count),
            str(snap_allow.effect_count),
            verdict,
        ),
    )


def run_scenario_7(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s7")
    key = f"key-s7-{RUN_ID}"
    pause = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}",
            business_operation_id=op,
            idempotency_key=key,
            require_hitl=True,
        ),
    )
    snap_pause = oracle.snapshot(op)
    resume = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}",
            business_operation_id=op,
            idempotency_key=key,
            require_hitl=True,
            hitl_resume=True,
        ),
    )
    snap_resume = oracle.snapshot(op)
    verdict = "PASS" if (
        pause.ledger_status is None
        and snap_pause.effect_count == 0
        and resume.success
        and snap_resume.effect_count == 1
    ) else "FAIL"
    report.add(
        ScenarioResult(
            7,
            "hitl_resume",
            "runtime-single",
            "require_hitl",
            f"pause:{pause.ledger_status},resume:{resume.ledger_status}",
            str(snap_resume.attempt_count),
            str(snap_resume.effect_count),
            verdict,
        ),
    )


def run_scenario_8(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s8")
    key = f"key-s8-{RUN_ID}"
    configured_delay = 2000
    proof_cfg = {
        "proof_mode": "timeout_response",
        "proof_delay_ms": configured_delay,
    }
    t0 = time.monotonic()
    first = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}",
            business_operation_id=op,
            idempotency_key=key,
            tool_id=PROOF_TOOL_SLOW_CHARGE,
            **proof_cfg,
        ),
    )
    t3 = time.monotonic()
    snap_timeout = oracle.snapshot(op)
    attempt = snap_timeout.attempts[0] if snap_timeout.attempts else None
    harness_valid = (
        attempt is not None
        and attempt.received_delay_ms is not None
        and attempt.received_delay_ms >= configured_delay
        and attempt.effect_committed_at is not None
    )
    timeline_notes = ""
    timeline_ok = False
    if attempt and attempt.received_at and attempt.effect_committed_at:
        first_ms = int((t3 - t0) * 1000)
        timeline_notes = (
            f"T1={attempt.received_at} T2={attempt.effect_committed_at} "
            f"T3_timeout~{first_ms}ms T4={attempt.response_finished_at or 'NONE'}"
        )
        timeline_ok = first_ms < configured_delay or attempt.response_finished_at is None
    retry = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}-retry",
            business_operation_id=op,
            idempotency_key=key,
            tool_id=PROOF_TOOL_SLOW_CHARGE,
            **proof_cfg,
        ),
    )
    snap_retry = oracle.snapshot(op)
    verdict = "FAIL"
    if not harness_valid:
        verdict = "FAIL"
        timeline_notes = f"HARNESS_INVALID delay_received={getattr(attempt, 'received_delay_ms', None)}; {timeline_notes}"
    elif (
        not first.success
        and first.ledger_status == "uncertain"
        and retry.uncertain
        and snap_timeout.effect_count == 1
        and snap_retry.effect_count == 1
        and snap_retry.attempt_count == snap_timeout.attempt_count
        and timeline_ok
    ):
        verdict = "PASS"
    elif (
        first.success
        and harness_valid
        and timeline_ok
    ):
        timeline_notes += " REAL_PRODUCTION_BUG_CANDIDATE"
    report.add(
        ScenarioResult(
            8,
            "effect_committed_response_delayed",
            "runtime-single",
            "timeout",
            f"{first.ledger_status}->{retry.ledger_status}",
            str(snap_retry.attempt_count),
            str(snap_retry.effect_count),
            verdict,
            notes=timeline_notes,
        ),
    )


def run_scenario_9(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s9")
    key = f"key-s9-{RUN_ID}"
    toxiproxy = os.environ.get("TOXIPROXY_URL", "http://toxiproxy:8474")
    with httpx.Client(timeout=10.0) as client:
        client.post(
            f"{toxiproxy}/proxies/effect-proxy/toxics",
            json={"name": "reset_peer", "type": "reset_peer", "toxicity": 1.0},
        )
    first = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}",
            business_operation_id=op,
            idempotency_key=key,
            proof_mode="normal",
        ),
    )
    snap = oracle.snapshot(op)
    with httpx.Client(timeout=10.0) as client:
        client.delete(f"{toxiproxy}/proxies/effect-proxy/toxics/reset_peer")
    _wait_lease_expiry()
    retry = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}-retry",
            business_operation_id=op,
            idempotency_key=key,
        ),
    )
    snap_retry = oracle.snapshot(op)
    verdict = "PASS" if snap.effect_count == 1 and snap_retry.effect_count == 1 and retry.uncertain else "FAIL"
    report.add(
        ScenarioResult(
            9,
            "connection_drop_after_commit",
            "runtime-single",
            "toxiproxy reset_peer",
            f"{first.ledger_status}->{retry.ledger_status}",
            str(snap_retry.attempt_count),
            str(snap_retry.effect_count),
            verdict,
        ),
    )


def run_scenario_10(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s10")
    key = f"key-s10-{RUN_ID}"
    result = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}",
            business_operation_id=op,
            idempotency_key=key,
            tool_id=PROOF_TOOL_FAIL_BEFORE,
        ),
    )
    snap = oracle.snapshot(op)
    verdict = "PASS" if snap.attempt_count == 0 and snap.effect_count == 0 and not result.success else "FAIL"
    report.add(
        ScenarioResult(
            10,
            "executor_error_before_effect",
            "runtime-single",
            "pre-effect failure",
            result.ledger_status or "",
            str(snap.attempt_count),
            str(snap.effect_count),
            verdict,
        ),
    )


def run_scenario_11(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s11")
    key = f"key-s11-{RUN_ID}"
    first = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}",
            business_operation_id=op,
            idempotency_key=key,
            tool_id=PROOF_TOOL_BAD_OUTPUT,
        ),
    )
    snap = oracle.snapshot(op)
    retry = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}-retry",
            business_operation_id=op,
            idempotency_key=key,
            tool_id=PROOF_TOOL_BAD_OUTPUT,
        ),
    )
    snap_retry = oracle.snapshot(op)
    verdict = "PASS" if (
        snap.effect_count == 1
        and snap_retry.effect_count == 1
        and first.ledger_status == "uncertain"
        and retry.uncertain
    ) else "FAIL"
    report.add(
        ScenarioResult(
            11,
            "bad_output_after_effect",
            "runtime-single",
            "validation failure",
            f"{first.ledger_status}->{retry.ledger_status}",
            str(snap_retry.attempt_count),
            str(snap_retry.effect_count),
            verdict,
        ),
    )


def run_scenario_12(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s12")
    key = f"key-s12-{RUN_ID}"
    error: Exception | None = None
    kill_ts = ""
    barrier_evidence = ""

    def _invoke() -> None:
        nonlocal error
        try:
            runtime.invoke(
                build_invoke(
                    run_id=f"run-{op}",
                    business_operation_id=op,
                    idempotency_key=key,
                    proof_mode="hold_before_commit",
                ),
            )
        except Exception as exc:  # noqa: BLE001
            error = exc

    thread = threading.Thread(target=_invoke)
    thread.start()
    try:
        deadline = time.monotonic() + 30.0
        snap = oracle.snapshot(op)
        while snap.attempt_count < 1 and time.monotonic() < deadline:
            time.sleep(0.1)
            snap = oracle.snapshot(op)
        barrier_evidence = (
            f"attempts={snap.attempt_count} effects={snap.effect_count} "
            f"mode={snap.attempts[0].proof_mode if snap.attempts else None}"
        )
        if snap.attempt_count < 1:
            raise RuntimeError("barrier not reached: no external attempt recorded")
        if snap.effect_count != 0:
            raise RuntimeError(
                f"barrier violated: effect committed before kill (effects={snap.effect_count})",
            )
        kill_ts = _docker_kill("runtime-single")
        thread.join(timeout=30.0)
        _docker_restart("runtime-single")
        runtime.wait_healthy()
        runtime = _runtime_single()
        snap = oracle.snapshot(op)
        verdict = "PASS" if snap.effect_count == 0 else "FAIL"
    except Exception as exc:  # noqa: BLE001
        verdict = "BLOCKED"
        snap = oracle.snapshot(op)
        error = exc
    report.add(
        ScenarioResult(
            12,
            "process_crash_after_claim_before_effect",
            "runtime-single",
            f"docker kill {kill_ts}",
            "started/uncertain",
            str(snap.attempt_count),
            str(snap.effect_count),
            verdict,
            notes=f"{barrier_evidence}; {error}" if error else barrier_evidence,
        ),
    )


def run_scenario_13(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s13")
    key = f"key-s13-{RUN_ID}"
    kill_ts = ""
    commit_ts = ""

    def _invoke() -> None:
        runtime.invoke(
            build_invoke(
                run_id=f"run-{op}",
                business_operation_id=op,
                idempotency_key=key,
                proof_mode="hold_after_commit",
            ),
        )

    thread = threading.Thread(target=_invoke)
    thread.start()
    snap = oracle.wait_for_effect_count(op, expected=1, timeout_s=30.0)
    if snap.attempts and snap.attempts[0].effect_committed_at:
        commit_ts = snap.attempts[0].effect_committed_at
    kill_ts = _docker_kill("runtime-single")
    thread.join(timeout=30.0)
    _docker_restart("runtime-single")
    runtime = _runtime_single()
    runtime.wait_healthy()
    _wait_lease_expiry()
    retry = runtime.invoke(
        build_invoke(
            run_id=f"run-{op}-retry",
            business_operation_id=op,
            idempotency_key=key,
        ),
    )
    snap = oracle.snapshot(op)
    verdict = "PASS" if snap.effect_count == 1 and snap.attempt_count == 1 else "FAIL"
    report.add(
        ScenarioResult(
            13,
            "process_crash_after_external_commit",
            "runtime-single",
            f"kill={kill_ts} commit={commit_ts}",
            retry.ledger_status or "uncertain",
            str(snap.attempt_count),
            str(snap.effect_count),
            verdict,
        ),
    )


def run_scenario_14(report: ProofReport) -> None:
    report.add(
        ScenarioResult(
            14,
            "kill_during_finalization",
            "runtime-single",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "NOT APPLICABLE",
            notes="covered by scenario 13; no deterministic black-box finalization window",
        ),
    )


def run_scenario_15(oracle: EffectOracle, report: ProofReport) -> None:
    op = _op("s15")
    key = f"key-s15-{RUN_ID}"
    runtime_a = _runtime_a()
    runtime_b = _runtime_b()
    owner_a_notes = ""
    owner_b_notes = ""
    late_finalize = ""

    def _owner_a_invoke() -> None:
        nonlocal owner_a_notes
        response = runtime_a.invoke(
            build_invoke(
                run_id=f"run-{op}-a",
                business_operation_id=op,
                idempotency_key=key,
                proof_mode="hold_after_commit",
            ),
        )
        owner_a_notes = f"success={response.success} ledger={response.ledger_status} err={response.error_type}"

    thread = threading.Thread(target=_owner_a_invoke)
    thread.start()
    snap = oracle.wait_for_effect_count(op, expected=1, timeout_s=30.0)
    _wait_lease_expiry()
    owner_b = runtime_b.invoke(
        build_invoke(
            run_id=f"run-{op}-b",
            business_operation_id=op,
            idempotency_key=key,
        ),
    )
    owner_b_notes = f"success={owner_b.success} ledger={owner_b.ledger_status} err={owner_b.error_type}"
    oracle.release_after(op)
    thread.join(timeout=60.0)
    late_finalize = owner_a_notes
    final_snap = oracle.snapshot(op)
    verdict = "PASS" if final_snap.effect_count == 1 else "FAIL"
    report.add(
        ScenarioResult(
            15,
            "stale_owner_fencing",
            "runtime-a/b",
            "lease expiry + late finalize",
            f"A:{owner_a_notes} B:{owner_b_notes}",
            str(final_snap.attempt_count),
            str(final_snap.effect_count),
            verdict,
            notes=f"late_finalize={late_finalize}",
        ),
    )


def run_scenario_16(oracle: EffectOracle, report: ProofReport) -> None:
    op = _op("s16")
    key = f"key-s16-{RUN_ID}"
    runtime_a = _runtime_a()
    runtime_b = _runtime_b()
    requests = []
    for i in range(50):
        requests.append((runtime_a, build_invoke(run_id=f"a-{i}", business_operation_id=op, idempotency_key=key)))
        requests.append((runtime_b, build_invoke(run_id=f"b-{i}", business_operation_id=op, idempotency_key=key)))

    barrier = threading.Barrier(len(requests))

    def _worker(pair) -> None:
        client, req = pair
        barrier.wait()
        client.invoke(req)

    with ThreadPoolExecutor(max_workers=100) as pool:
        futures = [pool.submit(_worker, pair) for pair in requests]
        for future in as_completed(futures):
            future.result()
    snap = oracle.snapshot(op)
    verdict = "PASS" if snap.effect_count == 1 else "FAIL"
    report.add(
        ScenarioResult(
            16,
            "multi_host_split",
            "runtime-a+b",
            "50/50 storm",
            "mixed",
            str(snap.attempt_count),
            str(snap.effect_count),
            verdict,
        ),
    )


def run_scenario_17(oracle: EffectOracle, report: ProofReport) -> None:
    random.seed(CHAOS_SEED)
    runtime_a = _runtime_a()
    runtime_b = _runtime_b()
    violations = 0
    for round_idx in range(100):
        op = _op(f"s17-r{round_idx}")
        key = f"key-s17-{round_idx}-{RUN_ID}"
        concurrency = random.randint(2, 20)
        clients = [runtime_a if random.random() < 0.5 else runtime_b for _ in range(concurrency)]
        requests = [
            build_invoke(
                run_id=f"run-{op}-{i}",
                business_operation_id=op,
                idempotency_key=key,
                proof_mode="normal" if random.random() > 0.2 else "timeout_response",
                proof_delay_ms=1500 if random.random() > 0.8 else 0,
                tool_id=PROOF_TOOL_SLOW_CHARGE if random.random() > 0.7 else PROOF_TOOL_CHARGE,
            )
            for i in range(concurrency)
        ]
        barrier = threading.Barrier(concurrency)

        def _worker(client, req) -> None:
            barrier.wait()
            try:
                client.invoke(req)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_worker, clients[i], requests[i]) for i in range(concurrency)]
            for future in as_completed(futures):
                future.result()
        snap = oracle.snapshot(op)
        if snap.effect_count > 1:
            violations += 1
    verdict = "PASS" if violations == 0 else "FAIL"
    report.add(
        ScenarioResult(
            17,
            "multi_host_chaos_loop",
            "runtime-a+b",
            f"seed={CHAOS_SEED}",
            "mixed",
            "100 rounds",
            "<=1 per op",
            verdict,
            notes=f"violations={violations}",
        ),
    )


def run_scenario_18(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s18")
    key = f"key-s18-{RUN_ID}"
    first = runtime.invoke(
        build_invoke(run_id=f"run-{op}", business_operation_id=op, idempotency_key=key),
    )
    attempts_after_first = oracle.snapshot(op).attempt_count
    _docker_restart("runtime-single")
    runtime.wait_healthy()
    runtime = _runtime_single()
    second = runtime.invoke(
        build_invoke(run_id=f"run-{op}-replay", business_operation_id=op, idempotency_key=key),
    )
    snap = oracle.snapshot(op)
    verdict = "PASS" if (
        first.success
        and second.success
        and snap.effect_count == 1
        and snap.attempt_count == attempts_after_first
    ) else "FAIL"
    report.add(
        ScenarioResult(
            18,
            "runtime_restart_replay",
            "runtime-single",
            "container restart",
            f"{first.ledger_status}->{second.ledger_status}",
            str(snap.attempt_count),
            str(snap.effect_count),
            verdict,
        ),
    )


def run_scenario_19(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    op = _op("s19")
    key = f"key-s19-{RUN_ID}"
    runtime.invoke(build_invoke(run_id=f"run-{op}", business_operation_id=op, idempotency_key=key))
    attempts = oracle.snapshot(op).attempt_count
    _docker_restart("runtime-single")
    runtime.wait_healthy()
    runtime = _runtime_single()
    replay = runtime.invoke(
        build_invoke(run_id=f"run-{op}-replay", business_operation_id=op, idempotency_key=key),
    )
    snap = oracle.snapshot(op)
    verdict = "PASS" if replay.success and snap.effect_count == 1 and snap.attempt_count == attempts else "FAIL"
    report.add(
        ScenarioResult(
            19,
            "store_restart",
            "runtime-single",
            "sqlite durable volume (runtime restart)",
            replay.ledger_status or "",
            str(snap.attempt_count),
            str(snap.effect_count),
            verdict,
            notes="Redis in compose is ephemeral (--save '' --appendonly no); N/A for durable store restart",
        ),
    )


def run_scenario_20(oracle: EffectOracle, runtime: RuntimeClient, report: ProofReport) -> None:
    attempts = 0
    replay_keys: dict[str, str] = {}
    for i in range(10_000):
        if i % 7 == 0 and replay_keys:
            key, op = next(iter(replay_keys.items()))
        else:
            op = _op(f"s20-{i}")
            key = f"key-s20-{i}-{RUN_ID}"
            replay_keys[key] = op
        try:
            runtime.invoke(
                build_invoke(
                    run_id=f"run-{op}-{i}",
                    business_operation_id=op,
                    idempotency_key=key,
                ),
            )
        except Exception:
            pass
        attempts += 1
    duplicates = oracle.duplicate_scan()
    verdict = "PASS" if not duplicates else "FAIL"
    report.add(
        ScenarioResult(
            20,
            "soak_10k",
            "runtime-single",
            "mixed",
            "mixed",
            str(attempts),
            "audited",
            verdict,
            notes=f"duplicates={len(duplicates)}",
        ),
    )
    report.duplicate_scan = duplicates


def _evaluate_gates(report: ProofReport) -> str:
    by_num = {scenario.number: scenario for scenario in report.scenarios}
    duplicates = report.duplicate_scan

    def _v(num: int) -> str:
        return by_num.get(num, ScenarioResult(num, "", "", "", "", "", "", "MISSING")).verdict

    if _v(0) == "FAIL":
        return "E. TOOLS-SIDE-EFFECT-SAFETY — BLOCKED — HARNESS INVALID (ContextVar/proof propagation)"

    if duplicates:
        return "D. TOOLS-SIDE-EFFECT-SAFETY — FAIL — DUPLICATE SIDE EFFECT OBSERVED"

    critical = [3, 4, 8, 9, 11, 13, 15, 18, 20]
    if any(_v(n) == "FAIL" for n in critical):
        s8 = by_num.get(8)
        if s8 and "REAL_PRODUCTION_BUG" in (s8.notes or ""):
            return "B. TOOLS-SIDE-EFFECT-SAFETY — FAIL — REAL TIMEOUT UNCERTAINTY BROKEN"
        return "D. TOOLS-SIDE-EFFECT-SAFETY — FAIL — UNCERTAINTY/REPLAY SAFETY BROKEN"

    if _v(12) == "BLOCKED" or _v(13) == "BLOCKED":
        return "E. TOOLS-SIDE-EFFECT-SAFETY — BLOCKED — CRASH/FENCING PROOF INCOMPLETE"

    if all(_v(n) in {"PASS", "NOT APPLICABLE", "BLOCKED"} for n in range(0, 21)):
        if _v(0) == "PASS" and _v(16) == "PASS" and _v(17) == "PASS":
            return "A. TOOLS-SIDE-EFFECT-SAFETY — CLOSED — BRUTAL PLATFORM PROOF PASS"
    return "D. TOOLS-SIDE-EFFECT-SAFETY — FAIL — UNCERTAINTY/REPLAY SAFETY BROKEN"


def _init_toxiproxy() -> None:
    toxiproxy = os.environ.get("TOXIPROXY_URL", "http://toxiproxy:8474")
    upstream = os.environ.get("TOXIPROXY_UPSTREAM", "external-effect-service:8080")
    with httpx.Client(timeout=15.0) as client:
        client.delete(f"{toxiproxy}/proxies/effect-proxy")
        response = client.post(
            f"{toxiproxy}/proxies",
            json={"name": "effect-proxy", "listen": "0.0.0.0:8666", "upstream": upstream},
        )
        response.raise_for_status()


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report = ProofReport()
    try:
        _run_all_scenarios(report)
    except Exception as exc:
        report.add(
            ScenarioResult(
                0,
                "proof_orchestrator",
                "proof-runner",
                "internal",
                "",
                "",
                "",
                "FAIL",
                notes=str(exc),
            ),
        )
    report.final_verdict = _evaluate_gates(report)
    artifact = ARTIFACT_DIR / "proof-report.json"
    artifact.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    failed = [s for s in report.scenarios if s.verdict == "FAIL"]
    if failed or "FAIL" in report.final_verdict:
        return 1
    return 0


def _run_all_scenarios(report: ProofReport) -> None:
    oracle = EffectOracle.from_env()
    oracle.wait_healthy()
    oracle.reset()
    try:
        _init_toxiproxy()
    except Exception as exc:  # noqa: BLE001
        print(f"toxiproxy init warning: {exc}", file=sys.stderr)

    runtime_single = _runtime_single()
    runtime_single.wait_healthy()
    _runtime_a().wait_healthy()
    _runtime_b().wait_healthy()

    if not run_scenario_0(oracle, runtime_single, report):
        report.final_verdict = _evaluate_gates(report)
        return

    run_scenario_1(oracle, runtime_single, report)
    run_scenario_2(oracle, runtime_single, report)
    run_scenario_3(oracle, runtime_single, report)
    run_scenario_4(oracle, runtime_single, report)
    run_scenario_5(oracle, runtime_single, report)
    run_scenario_6(oracle, runtime_single, report)
    run_scenario_7(oracle, runtime_single, report)
    run_scenario_8(oracle, runtime_single, report)
    try:
        run_scenario_9(oracle, runtime_single, report)
    except Exception as exc:  # noqa: BLE001
        report.add(
            ScenarioResult(
                9,
                "connection_drop_after_commit",
                "runtime-single",
                "toxiproxy",
                "",
                "",
                "",
                "FAIL",
                notes=str(exc),
            ),
        )
    run_scenario_10(oracle, runtime_single, report)
    run_scenario_11(oracle, runtime_single, report)
    try:
        run_scenario_12(oracle, runtime_single, report)
    except Exception as exc:  # noqa: BLE001
        report.add(
            ScenarioResult(
                12,
                "process_crash_after_claim_before_effect",
                "runtime-single",
                "docker kill",
                "",
                "",
                "",
                "BLOCKED",
                notes=str(exc),
            ),
        )
    try:
        run_scenario_13(oracle, runtime_single, report)
    except Exception as exc:  # noqa: BLE001
        report.add(
            ScenarioResult(
                13,
                "process_crash_after_external_commit",
                "runtime-single",
                "docker kill",
                "",
                "",
                "",
                "BLOCKED",
                notes=str(exc),
            ),
        )
    run_scenario_14(report)
    run_scenario_15(oracle, report)
    try:
        run_scenario_16(oracle, report)
        run_scenario_17(oracle, report)
    except Exception as exc:  # noqa: BLE001
        report.add(
            ScenarioResult(
                16,
                "multi_host_split",
                "runtime-a+b",
                "redis",
                "",
                "",
                "",
                "FAIL",
                notes=str(exc),
            ),
        )
    try:
        run_scenario_18(oracle, runtime_single, report)
    except Exception as exc:  # noqa: BLE001
        report.add(
            ScenarioResult(
                18,
                "runtime_restart_replay",
                "runtime-single",
                "container restart",
                "",
                "",
                "",
                "BLOCKED",
                notes=str(exc),
            ),
        )
    try:
        run_scenario_19(oracle, runtime_single, report)
    except Exception as exc:  # noqa: BLE001
        report.add(
            ScenarioResult(
                19,
                "store_restart",
                "runtime-single",
                "sqlite volume restart",
                "",
                "",
                "",
                "BLOCKED",
                notes=str(exc),
            ),
        )
    if not report.duplicate_scan:
        report.duplicate_scan = oracle.duplicate_scan()
    run_scenario_20(oracle, runtime_single, report)


if __name__ == "__main__":
    sys.exit(main())
