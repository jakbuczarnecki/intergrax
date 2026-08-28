#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Run local CI profiles from .github/workflows/unit-tests.yml (Regression gate)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

from script_paths import SCRIPT_PATHS as _SCRIPT_PATHS  # noqa: E402


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class Job:
    name: str
    steps: tuple[Step, ...]


def _uv_sync() -> Step:
    return Step("Install dependencies (CI minimal)", ["uv", "sync", "--extra", "dev-ci", "--frozen"])


def _ci_smoke_job() -> Job:
    return Job(
        "ci-smoke",
        (
            _uv_sync(),
            Step("CI smoke unit tests", ["uv", "run", "python", "scripts/ci/run_ci_smoke_pytest.py"]),
            Step(
                "Tier boundary audits (instant)",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/maintenance/check_harness_no_getattr.py",
                    "&&",
                    "uv",
                    "run",
                    "python",
                    "scripts/maintenance/check_agents_no_tier3_imports.py",
                    "&&",
                    "uv",
                    "run",
                    "python",
                    "scripts/ci/check_ci_gate_test_purity.py",
                ],
                env={"INTERGRAX_CI_TEST_MARKER": "ci_smoke"},
            ),
        ),
    )


def _gate_tests_job() -> Job:
    return Job(
        "gate-tests",
        (
            _uv_sync(),
            Step(
                "Full unit regression gate",
                [
                    "uv",
                    "run",
                    "pytest",
                    "tests/unit",
                    "-m",
                    "gate and not no_ci",
                    "-n",
                    "auto",
                    "-q",
                    "--tb=line",
                ],
            ),
        ),
    )


def _gate_governance_tier_job() -> Job:
    scripts = [
        "check_agents_vendor_imports.py",
        "check_capability_routing.py",
        "check_agent_step_security.py",
        "check_agent_threat_model.py",
        "check_agent_acp_close_ci.py",
        "check_contract_schema_versions.py",
        "check_acp_ci_conformance_matrix.py --scripts-only",
        "check_agent_typed_state.py",
        "check_production_chat_agent_imports.py",
        "check_integration_vendor_imports.py",
        "check_harness_no_getattr.py",
        "check_ci_gate_test_purity.py",
        "check_application_production_gates.py",
        "check_legacy_modules_removed.py",
        "check_agent_skill_resolution.py",
        "check_harness_registry_resolution.py",
        "check_agent_registry_bypass.py",
        "check_langgraph_not_required.py",
        "check_agents_no_tier3_imports.py",
        "check_docs_domain_pairs.py",
        "check_scaffold_harness_alignment.py",
        "check_plugin_catalog.py",
        "check_legacy_tool_plan_booleans.py",
    ]
    command: list[str] = []
    for index, script in enumerate(scripts):
        if index:
            command.append("&&")
        parts = script.split()
        command.extend(["uv", "run", "python", f"scripts/{_SCRIPT_PATHS[parts[0]]}", *parts[1:]])
    return Job(
        "gate-governance-tier",
        (
            _uv_sync(),
            Step(
                "Tier boundary and registry audits",
                command,
                env={"INTERGRAX_CI_TEST_MARKER": "gate and not no_ci"},
            ),
        ),
    )


def _gate_governance_wiring_job() -> Job:
    scripts = [
        "check_harness_capability_graph_wiring.py",
        "check_harness_observability_wiring.py",
        "check_observability_gates.py",
        "check_harness_reliability_wiring.py",
        "check_harness_security_wiring.py",
        "check_harness_guardrail_wiring.py",
        "check_harness_security_promote_gate.py",
        "check_harness_security_defense_plugins.py",
        "check_harness_encryption_policy.py",
        "check_harness_security_spine_signals.py",
        "check_harness_cost_wiring.py",
        "check_harness_evaluation_wiring.py",
    ]
    command: list[str] = []
    for index, script in enumerate(scripts):
        if index:
            command.append("&&")
        command.extend(["uv", "run", "python", f"scripts/{_SCRIPT_PATHS[script]}"])
    command.extend(["&&", "uv", "run", "intergrax", "doctor", "--ci"])
    return Job(
        "gate-governance-wiring",
        (_uv_sync(), Step("Harness wiring and observability audits", command)),
    )


def _gate_closeout_job() -> Job:
    return Job(
        "gate-closeout",
        (
            _uv_sync(),
            Step(
                "Phase V architecture closeout gate",
                ["uv", "run", "python", "scripts/release/phase_v_closeout_gate.py", "--enforce", "--enforce-l4"],
            ),
            Step(
                "Phase W-ADAPT runtime L4 closeout gate",
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/release/phase_w_adapt_closeout_gate.py",
                    "--enforce-l4-runtime",
                ],
            ),
            Step(
                "Phase W-OPS operational evidence (nightly / full dispatch)",
                ["uv", "run", "python", "scripts/release/phase_w_ops_evidence.py"],
            ),
            Step(
                "RAG load/soak nightly report (RAG-MAINT-02)",
                ["uv", "run", "python", "scripts/release/rag_load_soak_report.py"],
            ),
            Step(
                "Export harness spec schemas",
                ["uv", "run", "python", "scripts/release/export_harness_spec_schemas.py"],
            ),
            Step(
                "Export capability catalog feed",
                ["uv", "run", "python", "scripts/release/export_capability_catalog_feed.py"],
            ),
        ),
    )


def _jobs_for_profile(profile: str) -> list[Job]:
    if profile == "smoke":
        return [_ci_smoke_job()]
    if profile == "full":
        return [
            _gate_tests_job(),
            _gate_governance_tier_job(),
            _gate_governance_wiring_job(),
            _gate_closeout_job(),
        ]
    if profile == "all":
        return [_ci_smoke_job(), *_jobs_for_profile("full")]
    raise ValueError(f"unsupported profile: {profile}")


def _run_step(step: Step) -> int:
    env = os.environ.copy()
    if step.env:
        env.update(step.env)
    shell = sys.platform == "win32"
    command = step.command
    if shell and "&&" in command:
        command_str = " ".join(command)
        print(f"  $ {command_str}")
        completed = subprocess.run(command_str, cwd=_REPO, env=env, shell=True, check=False)
        return completed.returncode
    print(f"  $ {' '.join(command)}")
    completed = subprocess.run(command, cwd=_REPO, env=env, check=False)
    return completed.returncode


def _run_job(job: Job) -> tuple[int, float]:
    started = time.monotonic()
    print(f"\n=== job: {job.name} ===")
    for step in job.steps:
        print(f"\n-- {step.name}")
        code = _run_step(step)
        if code != 0:
            elapsed = time.monotonic() - started
            print(f"\nFAILED job={job.name} step={step.name} exit={code}")
            return code, elapsed
    elapsed = time.monotonic() - started
    print(f"\nPASSED job={job.name} ({elapsed:.1f}s)")
    return 0, elapsed


def _print_summary(results: Sequence[tuple[str, int, float]]) -> int:
    print("\n=== summary ===")
    overall = 0
    for job_name, code, elapsed in results:
        status = "PASS" if code == 0 else "FAIL"
        print(f"{status:4}  {job_name:28}  exit={code}  {elapsed:.1f}s")
        if code != 0:
            overall = code
    return overall


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Regression gate CI profiles locally (unit-tests.yml parity)."
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "full", "all"),
        default="all",
        help="smoke=PR/push CI; full=nightly jobs; all=smoke then full (default).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    jobs = _jobs_for_profile(args.profile)
    results: list[tuple[str, int, float]] = []
    for job in jobs:
        code, elapsed = _run_job(job)
        results.append((job.name, code, elapsed))
        if code != 0:
            return _print_summary(results)

    return _print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
