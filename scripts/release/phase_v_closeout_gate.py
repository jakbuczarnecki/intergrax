#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Phase V closeout gate: capability graph guard, governance artifacts, L3/L4 evidence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_DIR = REPO_ROOT / "scripts" / "ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))
from script_paths import resolve_script  # noqa: E402

for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.runtime.architecture.adaptive_governance import (
    build_default_adaptive_proposals,
    evaluate_adaptive_governance,
)
from intergrax.runtime.architecture.maturity_gate_evidence import (
    collect_harness_governance_signals,
    evaluate_maturity_gate_evidence,
)


class ReportWriter(Protocol):
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        ...


class JsonReportWriter:
    def write(self, *, output_path: Path, payload: BaseModel) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Return non-zero when L3 maturity gate evidence fails.",
    )
    parser.add_argument(
        "--enforce-l4",
        action="store_true",
        help="Return non-zero when L4 maturity gate evidence fails (implies L3).",
    )
    parser.add_argument(
        "--skip-scripts",
        action="store_true",
        help="Skip subprocess report scripts; only evaluate maturity evidence.",
    )
    return parser.parse_args()


def _run_script(script_name: str, *extra_args: str) -> int:
    command = [sys.executable, str(resolve_script(script_name)), *extra_args]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    args = _parse_args()
    output_dir = REPO_ROOT / "build" / "architecture_hardening"
    writer: ReportWriter = JsonReportWriter()

    if not args.skip_scripts:
        foundations_exit = _run_script("phase_v_foundations_report.py")
        if foundations_exit != 0:
            return foundations_exit
        guard_exit = _run_script("phase_v_capability_graph_guard.py", "--enforce")
        if guard_exit != 0:
            return guard_exit
        governance_exit = _run_script("phase_v_governance_report.py")
        if governance_exit != 0:
            return governance_exit
        critic_exit = _run_script("check_harness_critic_wiring.py")
        if critic_exit != 0:
            return critic_exit

    adaptive_report = evaluate_adaptive_governance(build_default_adaptive_proposals())
    inputs = collect_harness_governance_signals()
    maturity_report = evaluate_maturity_gate_evidence(inputs)

    writer.write(
        output_path=output_dir / "adaptive_governance_report.json",
        payload=adaptive_report,
    )
    writer.write(
        output_path=output_dir / "maturity_gate_evidence_report.json",
        payload=maturity_report,
    )

    print("phase-v closeout gate: OK")
    print(f"l3_passed: {maturity_report.l3.passed}")
    print(f"l4_governance_passed: {maturity_report.l4_governance.passed}")
    print(f"l4_runtime_passed: {maturity_report.l4_runtime.passed}")
    print(f"l4_passed: {maturity_report.l4.passed}")
    print(f"artifacts: {output_dir.as_posix()}")

    if args.enforce and not inputs.evaluation_registry_available:
        print("phase-v closeout gate: evaluation registry baseline missing")
        return 1
    if args.enforce_l4 and not maturity_report.l4.passed:
        return 1
    if args.enforce and not maturity_report.l3.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
