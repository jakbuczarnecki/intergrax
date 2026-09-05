"""VPI embedding model arena CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_environment import load_proof_environment

from platform_proofs.scenarios.verified_product_identification.arena.composition.execution_profiles import (
    STANDARD_ARENA_PROFILE_ID,
    list_execution_profile_ids,
    resolve_execution_budget,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    ArenaExecutionEnvironmentError,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.reporting import (
    write_arena_report,
    write_arena_summary,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.runner import (
    run_embedding_arena,
)

_SCENARIO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCENARIO_DIR.parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VPI embedding model quality / throughput arena (5C4B)"
    )
    parser.add_argument(
        "--include-e5-control",
        action="store_true",
        help="Include optional E5-instruct control candidate",
    )
    parser.add_argument(
        "--skip-gpu-stages",
        action="store_true",
        help="Build sample/query evidence only; skip GPU embedding stages",
    )
    parser.add_argument(
        "--profile",
        choices=list_execution_profile_ids(),
        default=STANDARD_ARENA_PROFILE_ID,
        help="Arena execution profile (safe-local-gpu = resource-bounded micro arena)",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Session output directory (default: .tmp/session/vpi-5c4b)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_proof_environment(proof_package_dir=_SCENARIO_DIR, repository_root=_REPO_ROOT)

    session_dir = args.session_dir or (_REPO_ROOT / ".tmp" / "session" / "vpi-5c4b")
    execution_budget = resolve_execution_budget(args.profile)
    try:
        report = run_embedding_arena(
            include_e5_control=args.include_e5_control,
            run_gpu_stages=not args.skip_gpu_stages,
            session_dir=str(session_dir),
            execution_budget=execution_budget,
        )
    except ArenaExecutionEnvironmentError as exc:
        snapshot = exc.snapshot
        print(f"status={snapshot.status.value}")
        print(f"profile={snapshot.profile_id}")
        print(f"python_executable={snapshot.python_executable}")
        if snapshot.detail is not None:
            print(f"detail={snapshot.detail}")
        return 1
    write_arena_report(session_dir / "arena-report.json", report)
    write_arena_summary(session_dir / "ARENA_SUMMARY.md", report)

    print(f"profile={report.execution_profile_id}")
    print(f"decision={report.decision.value}")
    print(f"report={session_dir / 'arena-report.json'}")
    print(f"summary={session_dir / 'ARENA_SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
