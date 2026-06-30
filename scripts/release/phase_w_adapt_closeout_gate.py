#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Phase W-ADAPT closeout gate: verification report + L4 runtime evidence (W-ADAPT-5.6)."""

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

from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.l4_runtime_evidence import (
    build_harness_baseline_l4_evidence,
    build_harness_baseline_signals,
    build_l4_runtime_evidence_from_signals,
)
from intergrax.runtime.adaptive.loop_apply_block_store import InMemoryLoopApplyBlockStore
from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
from intergrax.runtime.adaptive.profile_pointer_store import InMemoryProfileActivePointerStore
from intergrax.runtime.adaptive.profile_version_store import InMemoryProfileVersionStore
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.adaptive.verification_loop import VerificationLoop
from intergrax.runtime.adaptive.verification_models import VerificationContext
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
        "--enforce-l4-runtime",
        action="store_true",
        help="Return non-zero when runtime L4 closed-loop evidence fails.",
    )
    parser.add_argument(
        "--skip-scripts",
        action="store_true",
        help="Skip subprocess adaptive report scripts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "build" / "adaptive_harness",
        help="Adaptive harness artifact output directory",
    )
    return parser.parse_args()


def _run_script(script_name: str, *extra_args: str) -> int:
    command = [sys.executable, str(resolve_script(script_name)), *extra_args]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def _build_verification_report(*, use_baseline: bool) -> tuple[object, object]:
    signal_store = InMemorySignalStore()
    if use_baseline:
        for signal in build_harness_baseline_signals():
            signal_store.append(signal)

    profile_store = InMemoryProfileVersionStore()
    pointer_store = InMemoryProfileActivePointerStore()
    lifecycle = ProfileVersionLifecycleManager(store=profile_store)
    executor = AdaptationExecutor(
        profile_store=profile_store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle,
    )
    verification_loop = VerificationLoop(
        signal_store=signal_store,
        profile_store=profile_store,
        executor=executor,
        block_store=InMemoryLoopApplyBlockStore(),
    )
    context = VerificationContext(
        min_improvement_delta=0.0,
        min_run_count=3,
        auto_rollback_enabled=False,
    )
    verification_report = verification_loop.verify_active_profiles(context=context)
    if use_baseline:
        l4_evidence = build_harness_baseline_l4_evidence()
    else:
        l4_evidence = build_l4_runtime_evidence_from_signals(
            signal_store,
            verification_report=verification_report,
        )
    return verification_report, l4_evidence


def main() -> int:
    args = _parse_args()
    writer: ReportWriter = JsonReportWriter()

    if not args.skip_scripts:
        report_exit = _run_script(
            "phase_w_adapt_report.py",
            "--verification-output",
            str(args.output_dir / "verification_report.json"),
        )
        if report_exit != 0:
            return report_exit

    verification_report, l4_evidence = _build_verification_report(use_baseline=True)
    writer.write(
        output_path=args.output_dir / "verification_report.json",
        payload=verification_report,
    )
    writer.write(
        output_path=args.output_dir / "l4_runtime_evidence.json",
        payload=l4_evidence,
    )

    governance_inputs = collect_harness_governance_signals()
    maturity_inputs = governance_inputs.model_copy(
        update={"runtime_l4_closed_loop_passed": l4_evidence.runtime_l4_closed_loop_passed}
    )
    maturity_report = evaluate_maturity_gate_evidence(maturity_inputs)
    writer.write(
        output_path=args.output_dir / "runtime_l4_maturity_report.json",
        payload=maturity_report,
    )

    print("phase-w-adapt closeout gate: OK")
    print(f"verification_passed: {verification_report.passed}")
    print(f"runtime_l4_closed_loop_passed: {l4_evidence.runtime_l4_closed_loop_passed}")
    print(f"l4_runtime_maturity_passed: {maturity_report.l4_runtime.passed}")
    print(f"artifacts: {args.output_dir.as_posix()}")

    if args.enforce_l4_runtime and not l4_evidence.runtime_l4_closed_loop_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
