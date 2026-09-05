"""Run a single embedding arena candidate in an isolated process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.proof.intergrax_proof_environment import load_proof_environment

from platform_proofs.scenarios.verified_product_identification.arena.composition.execution_profiles import (
    resolve_execution_budget,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    ArenaExecutionEnvironmentError,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.candidate_isolation import (
    write_candidate_phase_artifact,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.execution_environment import (
    validate_arena_execution_environment,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.runner import (
    execute_candidate_stage_ab,
    execute_candidate_stage_c,
)

_SCENARIO_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCENARIO_DIR.parents[2]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VPI embedding arena single-candidate worker")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--phase", choices=("stage_ab", "stage_c"), required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--include-e5-control", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_proof_environment(proof_package_dir=_SCENARIO_DIR, repository_root=_REPO_ROOT)

    execution_budget = resolve_execution_budget(args.profile)
    try:
        validate_arena_execution_environment(execution_budget)
    except ArenaExecutionEnvironmentError as exc:
        snapshot = exc.snapshot
        print(f"status={snapshot.status.value}", file=sys.stderr)
        if snapshot.detail is not None:
            print(snapshot.detail, file=sys.stderr)
        return 1
    if args.phase == "stage_ab":
        payload = execute_candidate_stage_ab(
            candidate_id=args.candidate_id,
            execution_budget=execution_budget,
            session_dir=str(args.session_dir),
            include_e5_control=args.include_e5_control,
        )
    else:
        payload = execute_candidate_stage_c(
            candidate_id=args.candidate_id,
            execution_budget=execution_budget,
            session_dir=str(args.session_dir),
            include_e5_control=args.include_e5_control,
        )
    write_candidate_phase_artifact(
        args.session_dir,
        args.candidate_id,
        args.phase,
        payload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
