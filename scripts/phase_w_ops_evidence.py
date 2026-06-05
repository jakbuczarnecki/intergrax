#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Operational L3 harness evidence (Phase W-OPS.5) — distinct from V-V6 contract gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_CYCLES_PATH = REPO_ROOT / "build" / "architecture_hardening" / "release_cycles.json"
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


class OperationalHarnessCheck(BaseModel):
    check_id: str
    passed: bool
    detail: str = ""


class OperationalMaturityEvidence(BaseModel):
    schema_version: str = "1.0.0"
    release_cycles_completed: int = 0
    release_cycles_required: int = 2
    checks: list[OperationalHarnessCheck] = Field(default_factory=list)
    operational_l3_ready: bool = False
    note: str = (
        "Distinct from phase_v_closeout_gate (synthetic thresholds). "
        "Set W_OPS_RELEASE_CYCLES when release board signs off."
    )


def _run_pytest(*targets: str) -> bool:
    if not targets:
        return False
    cmd = [sys.executable, "-m", "pytest", *targets, "-q", "--tb=line"]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return completed.returncode == 0


def _resolve_release_cycles() -> int:
    env_raw = (os.getenv("W_OPS_RELEASE_CYCLES") or "").strip()
    if env_raw:
        try:
            return max(0, int(env_raw))
        except ValueError:
            return 0
    if not RELEASE_CYCLES_PATH.is_file():
        return 0
    from intergrax.runtime.architecture.release_cycle_tracker import load_release_cycle_tracker

    return load_release_cycle_tracker(RELEASE_CYCLES_PATH).completed_count


def _shadow_trend_export_ok() -> bool:
    try:
        from intergrax.runtime.architecture.online_evaluation_registry import (
            InMemoryOnlineEvaluationRegistry,
        )
        from intergrax.runtime.architecture.online_evaluation_trend import (
            export_shadow_evaluation_trend,
        )
        from intergrax.runtime.architecture.online_evaluation import record_shadow_observation

        registry = InMemoryOnlineEvaluationRegistry()
        record_shadow_observation(
            run_id="w-ops-evidence",
            agent_id="echo",
            scenario_id="harness.ops",
            passed=True,
            score=1.0,
            registry=registry,
        )
        report = export_shadow_evaluation_trend(
            "w-ops-evidence-probe",
            registry=registry,
            clear_registry_after_export=True,
        )
        return len(report.snapshots) >= 1
    except Exception:
        return False


def collect_operational_checks() -> OperationalMaturityEvidence:
    release_cycles = _resolve_release_cycles()

    slo_doc = REPO_ROOT / "docs" / "HARNESS_ENVIRONMENT.md"
    checks = [
        OperationalHarnessCheck(
            check_id="idempotency_gate",
            passed=_run_pytest("tests/unit/runtime/tools/test_idempotent_invoker.py"),
            detail="Side-effect tool idempotency (W-OPS.1)",
        ),
        OperationalHarnessCheck(
            check_id="circuit_breaker_gate",
            passed=_run_pytest("tests/unit/integrations/test_integration_circuit_breaker.py"),
            detail="Integration circuit breaker (W-OPS.2)",
        ),
        OperationalHarnessCheck(
            check_id="long_running_gate",
            passed=_run_pytest(
                "tests/integration/runtime/long_running/test_long_running_scheduler_j4.py"
            ),
            detail="Scheduler / checkpoint reliability (W-OPS.3)",
        ),
        OperationalHarnessCheck(
            check_id="requires_skills_gate",
            passed=_run_pytest("tests/unit/skills/test_harness_requires_skills_demo.py"),
            detail="Harness requires_skills demo (W-OPS.9)",
        ),
        OperationalHarnessCheck(
            check_id="harness_lab_health_gate",
            passed=_run_pytest("tests/unit/integrations/test_harness_lab_health.py"),
            detail="Lab stable stack health probes (W-OPS.10)",
        ),
        OperationalHarnessCheck(
            check_id="shadow_eval_gate",
            passed=_run_pytest(
                "tests/unit/runtime/architecture/test_online_evaluation.py",
                "tests/unit/runtime/architecture/test_online_evaluation_registry.py",
                "tests/unit/runtime/architecture/test_online_evaluation_trend.py",
                "tests/unit/runtime/architecture/test_runtime_shadow_evaluation.py",
            ),
            detail="Shadow evaluation registry + RuntimeEngine hook (W-OPS.11)",
        ),
        OperationalHarnessCheck(
            check_id="shadow_trend_export",
            passed=_shadow_trend_export_ok(),
            detail="Shadow evaluation trend export API (W-OPS.11)",
        ),
        OperationalHarnessCheck(
            check_id="slo_documentation",
            passed=slo_doc.is_file() and "Harness SLO catalog" in slo_doc.read_text(encoding="utf-8"),
            detail="SLO catalog in HARNESS_ENVIRONMENT.md (W-OPS.4)",
        ),
        OperationalHarnessCheck(
            check_id="memory_platform_gate",
            passed=_run_pytest(
                "tests/unit/applications/test_memory_wiring.py",
                "tests/unit/applications/test_memory_profile_runtime_bridge.py",
                "tests/unit/applications/test_reference_hosts_memory_bridge.py",
            ),
            detail="H-APP memory bridge + reference host wiring (MEM closeout)",
        ),
        OperationalHarnessCheck(
            check_id="release_cycles",
            passed=release_cycles >= 2,
            detail=f"W_OPS_RELEASE_CYCLES={release_cycles} (need >= 2 for operational L3)",
        ),
    ]
    code_checks_passed = all(
        item.passed for item in checks if item.check_id != "release_cycles"
    )
    operational_l3_ready = code_checks_passed and release_cycles >= 2
    return OperationalMaturityEvidence(
        release_cycles_completed=release_cycles,
        checks=checks,
        operational_l3_ready=operational_l3_ready,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit non-zero when operational L3 is not ready.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "build" / "architecture_hardening" / "operational_maturity_evidence.json",
    )
    args = parser.parse_args()

    evidence = collect_operational_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    print(f"phase-w-ops evidence: operational_l3_ready={evidence.operational_l3_ready}")
    print(f"artifacts: {args.output.parent.as_posix()}")
    if args.enforce and not evidence.operational_l3_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
