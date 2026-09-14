# © Artur Czarnecki. All rights reserved.

"""CLI entry for live qualification performance benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from testing_support.execution_qualification.configuration import (
    resolve_execution_qualification_max_parallel,
)
from testing_support.execution_qualification.contracts import QualificationRunStatus
from testing_support.execution_qualification.performance.certification import (
    MANDATORY_BENCHMARK_PROFILE_IDS,
    PRIMARY_BENCHMARK_PROFILE_ID,
    assemble_certification_report,
    build_default_benchmark_runner,
    build_structural_profile_matrix,
    merge_measured_primary_profile,
    read_git_head,
    resolve_repo_root,
)
from testing_support.execution_qualification.performance.serialization import (
    serialize_performance_certification_report,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical qualification performance benchmark (real pytest subprocesses).",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Canonical profile id (e.g. npsc5f-final).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Benchmark repetitions for the selected profile (>=1).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override max_parallel (default: env or qualified default=2).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Artifact base directory (default: .tmp/session/qualification-performance).",
    )
    parser.add_argument(
        "--structural-matrix",
        action="store_true",
        help="Include static execution-count matrix for mandatory profiles.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional path to write JSON certification report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")

    repo_root = resolve_repo_root()
    artifact_base = args.artifact_dir
    if artifact_base is None:
        artifact_base = repo_root / ".tmp" / "session" / "qualification-performance"
    runner = build_default_benchmark_runner(repo_root, artifact_base=artifact_base)

    profile_ids = MANDATORY_BENCHMARK_PROFILE_IDS
    structural = (
        build_structural_profile_matrix(runner, profile_ids)
        if args.structural_matrix
        else ()
    )

    outcomes = []
    canonical_failed = False
    for index in range(args.repetitions):
        outcome = runner.run_profile_once(
            args.profile,
            repetition_index=index,
            max_parallel=args.max_parallel,
        )
        if outcome.plan_result.status is not QualificationRunStatus.PASS:
            canonical_failed = True
        outcomes.append(outcome)

    measured = runner.build_profile_result_from_outcomes(
        args.profile,
        tuple(outcomes),
        max_parallel=args.max_parallel,
    )
    profiles = structural
    if profiles:
        profiles = merge_measured_primary_profile(profiles, measured)
    else:
        profiles = (measured,)

    resolved_parallel = resolve_execution_qualification_max_parallel(
        explicit_value=args.max_parallel,
    )
    report = assemble_certification_report(
        git_head=read_git_head(repo_root),
        environment_max_parallel=resolved_parallel,
        profiles=profiles,
        primary_profile_id=PRIMARY_BENCHMARK_PROFILE_ID
        if args.profile == PRIMARY_BENCHMARK_PROFILE_ID
        else args.profile,
        canonical_run_failed=canonical_failed,
    )
    serialized = serialize_performance_certification_report(report)
    sys.stdout.write(serialized)
    sys.stdout.write("\n")
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(serialized, encoding="utf-8")
    return 1 if canonical_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
