# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""CLI: DS-E2E-15J-L1.R6 multi-model alignment reliability qualification."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from testing_support.decision_e2e.local_ai_incident_qualification import (
    QualificationCliExit,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)
from testing_support.decision_e2e.model_matrix.analysis import (
    run_multi_model_qualification_analysis,
)
from testing_support.decision_e2e.model_matrix.qualification_cohort_executor import (
    CohortExecutionStatus,
)
from testing_support.decision_e2e.model_matrix.qualification_execution_pipeline import (
    run_qualification_execution_pipeline,
)
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    DEFAULT_COHORT_RUN_COUNT,
    R6_TASK_ID,
    build_cohort_plans,
    qualification_artifact_root,
)
from testing_support.decision_e2e.model_matrix.model_qualification_contract import (
    contract_for_profile,
)
from testing_support.decision_e2e.model_matrix.registry import (
    QualificationRegistry,
    profile_by_key,
)
from testing_support.decision_e2e.model_matrix.source_freeze import (
    verify_model_matrix_source_freeze,
)


def _resolve_models(args: argparse.Namespace) -> tuple | None:
    if args.profile_key and args.all_models:
        print("BLOCKED: use either --profile-key or --all-models")
        return None
    if args.profile_key:
        selected = profile_by_key(args.profile_key)
        if selected is None:
            print(f"BLOCKED: unknown profile_key={args.profile_key}")
            return None
        return (contract_for_profile(selected),)
    return QualificationRegistry.contracts()


async def _run_all_cohorts(args: argparse.Namespace) -> int:
    models = _resolve_models(args)
    if models is None:
        return int(QualificationCliExit.BLOCKED_PRECONDITION)
    env_digest = (
        os.environ.get("INTERGRAX_QUALIFICATION_MODEL_DIGEST", "") or None
        if args.profile_key
        else None
    )
    pipeline = await run_qualification_execution_pipeline(
        args.repo_root,
        models,
        run_count=args.run_count,
        ollama_base_url=args.ollama_url,
        env_digest=env_digest,
        resume=args.resume,
        finalize_only=args.finalize_only,
    )
    for result in pipeline.cohort_results:
        if result.status is CohortExecutionStatus.MODEL_UNAVAILABLE:
            print(f"MODEL_UNAVAILABLE profile={result.profile_key}")
            continue
        if result.status is CohortExecutionStatus.BLOCKED_PRECONDITION:
            print(f"BLOCKED: digest missing profile={result.profile_key}")
            continue
        if result.session_state is not None:
            print(
                f"profile={result.profile_key} session_state={result.session_state.value}"
            )
    return pipeline.worst_exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=R6_TASK_ID)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Qualification artifact root (default: .artifacts/qualification/DS-E2E-15J-L1.R6)",
    )
    parser.add_argument("--run-count", type=int, default=DEFAULT_COHORT_RUN_COUNT)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--profile-key", default=None, help="Run a single registry profile")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run all profiles from the qualification registry",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument(
        "--source-freeze-only",
        action="store_true",
        help="Phase A repository source freeze verification only",
    )
    parser.add_argument(
        "--cohort-only",
        action="store_true",
        help="Execute cohort(s) only",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Aggregate multi-model analysis only",
    )
    args = parser.parse_args(argv)
    args.artifact_root = args.artifact_root or qualification_artifact_root(args.repo_root)

    freeze = verify_model_matrix_source_freeze(args.repo_root)
    print(f"SOURCE_FREEZE_STATUS={freeze.status.value}")
    if freeze.status is not SourceFreezeStatus.PASS:
        return 2
    if args.source_freeze_only:
        return 0

    env_digest = (
        os.environ.get("INTERGRAX_QUALIFICATION_MODEL_DIGEST", "") or None
        if args.profile_key
        else None
    )
    models = _resolve_models(args)
    if models is None:
        return int(QualificationCliExit.BLOCKED_PRECONDITION)
    from testing_support.decision_e2e.model_matrix.model_qualification_contract import (
        profiles_from_contracts,
    )

    plans = build_cohort_plans(
        args.repo_root,
        profiles_from_contracts(models),
        run_count=args.run_count,
        ollama_base_url=args.ollama_url,
        env_digest=env_digest,
    )

    if args.analyze_only:
        result = run_multi_model_qualification_analysis(
            repo_root=args.repo_root,
            plans=plans,
        )
        print(f"matrix_status={result.matrix_status}")
        for metrics in result.per_model:
            print(
                f"profile={metrics.profile_key} 15K_B_EFFECT={metrics.fifteen_kb_effect.value}"
            )
        return 0 if result.matrix_status == "PASS" else 4

    if args.cohort_only or not args.analyze_only:
        exit_code = asyncio.run(_run_all_cohorts(args))
        if exit_code != 0:
            return exit_code
        if args.cohort_only:
            return 0

    result = run_multi_model_qualification_analysis(
        repo_root=args.repo_root,
        plans=plans,
    )
    print(f"matrix_status={result.matrix_status}")
    for metrics in result.per_model:
        print(
            f"profile={metrics.profile_key} 15K_B_EFFECT={metrics.fifteen_kb_effect.value}"
        )
    return 0 if result.matrix_status == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
