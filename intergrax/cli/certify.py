# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""``intergrax certify`` — harness runtime evidence CLI (HEP Band 2ae · EVID-CORE-04)."""

from __future__ import annotations

import argparse
from pathlib import Path

from intergrax.runtime.evidence.certification_report import DEFAULT_CORE_CERTIFICATION_OUTPUT_DIR
from intergrax.runtime.evidence.core_certification_spec import CoreCertificationMode
from intergrax.runtime.evidence.scenario_runner import run_core_certification


def register_parser(sub: argparse._SubParsersAction) -> None:
    certify = sub.add_parser("certify", help="Harness certification and runtime evidence.")
    certify_sub = certify.add_subparsers(dest="certify_command", required=True)
    core = certify_sub.add_parser("core", help="Run controlled core harness certification scenarios.")
    core.add_argument(
        "--level",
        default="L2",
        help="Certification depth: CORE-L1/L2/L3 or aliases l1, l2, l3 (default: L2).",
    )
    core.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report output directory (default: build/evidence/core_certification).",
    )
    core.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root when resolving default output directory.",
    )


def run_certify_core(args: argparse.Namespace) -> int:
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.root.resolve() / DEFAULT_CORE_CERTIFICATION_OUTPUT_DIR
    else:
        output_dir = output_dir.resolve()

    report = run_core_certification(
        args.level,
        output_dir=output_dir,
        mode=CoreCertificationMode.OPERATOR_LOCAL,
    )
    print(f"core certification level: {report.certification_level.value}")
    print(f"passed: {report.passed}")
    print(f"scenarios: {report.scenarios_passed}/{report.scenarios_total} passed")
    print(f"report: {output_dir / 'report.json'}")
    if not report.passed:
        for result in report.scenario_results:
            if result.status.value != "passed":
                print(f"  FAIL {result.scenario_id}: {result.message}")
        return 1
    return 0


def run_certify(args: argparse.Namespace) -> int:
    if args.certify_command == "core":
        return run_certify_core(args)
    return 2
