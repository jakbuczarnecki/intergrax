"""VPI embedding performance qualification CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_environment import load_proof_environment

from platform_proofs.scenarios.verified_product_identification.qualification.reporting import (
    write_qualification_report,
)
from platform_proofs.scenarios.verified_product_identification.qualification.runner import (
    DEFAULT_MICROBENCHMARK_RECORDS,
    DEFAULT_RECORD_TARGET,
    run_vpi_embedding_qualification,
)

_SCENARIO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCENARIO_DIR.parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VPI real embedding and storage E2E performance qualification (5C4A)"
    )
    parser.add_argument(
        "--record-target",
        type=int,
        default=DEFAULT_RECORD_TARGET,
        help=f"Real materialization target (default: {DEFAULT_RECORD_TARGET})",
    )
    parser.add_argument(
        "--microbenchmark-records",
        type=int,
        default=DEFAULT_MICROBENCHMARK_RECORDS,
        help=f"Real-WDC microbenchmark sample size (default: {DEFAULT_MICROBENCHMARK_RECORDS})",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Isolated artifact directory (default: .tmp/session/vpi-5c4a/artifact-1k)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON qualification report path (default: .tmp/session/vpi-5c4a/qualification-report.json)",
    )
    parser.add_argument(
        "--run-target-extension",
        action="store_true",
        help="Optionally extend materialization target by +100 after 1K proof",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_proof_environment(proof_package_dir=_SCENARIO_DIR, repository_root=_REPO_ROOT)

    session_dir = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4a"
    artifact_dir = args.artifact_dir or (session_dir / "artifact-1k")
    report_path = args.report_path or (session_dir / "qualification-report.json")

    report = run_vpi_embedding_qualification(
        record_target=args.record_target,
        microbenchmark_records=args.microbenchmark_records,
        artifact_dir=artifact_dir,
        run_target_extension=args.run_target_extension,
    )
    write_qualification_report(report_path, report)

    print(f"status={report.status.value}")
    print(f"report={report_path}")
    if report.materialization is not None:
        print(
            "materialization="
            f"state={report.materialization.state} "
            f"rows={report.materialization.rows} "
            f"embed_calls={report.materialization.embedding_calls} "
            f"records_per_sec={report.materialization.embedding_records_per_second:.2f}"
        )
    if report.full_build_estimate is not None:
        print(
            "full_build_estimate="
            f"embedding_hours={report.full_build_estimate.estimated_embedding_hours:.1f} "
            f"total_hours={report.full_build_estimate.estimated_total_hours:.1f}"
        )
    for warning in report.warnings:
        print(f"warning={warning}", file=sys.stderr)

    return 0 if report.status.value == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
