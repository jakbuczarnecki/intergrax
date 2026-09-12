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

from testing_support.decision_e2e.env_bootstrap import bootstrap_qualification_environment
from testing_support.decision_e2e.local_ai_incident_qualification import (
    AiIncidentSingleRunExecutor,
    OllamaProviderIdentityProbe,
    QualificationCliExit,
    resolve_repository_head_sha,
    run_local_ai_incident_qualification,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)
from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
    OllamaProbeConfig,
)
from testing_support.decision_e2e.model_matrix.analysis import (
    run_multi_model_qualification_analysis,
)
from testing_support.decision_e2e.model_matrix.availability import ModelAvailability
from testing_support.decision_e2e.model_matrix.qualification_plan import (
    DEFAULT_COHORT_RUN_COUNT,
    R6_TASK_ID,
    build_cohort_plans,
    build_qualification_spec_for_profile,
    qualification_artifact_root,
)
from testing_support.decision_e2e.model_matrix.registry import (
    QualificationRegistry,
    profile_by_key,
)
from testing_support.decision_e2e.model_matrix.source_freeze import (
    verify_model_matrix_source_freeze,
)


def _apply_runtime_env(profile_provider: str, profile_model: str) -> None:
    os.environ["INTERGRAX_LLM_PROVIDER"] = profile_provider
    os.environ["INTERGRAX_LLM_MODEL"] = profile_model
    os.environ["INTERGRAX_DECISION_E2E_QUALIFICATION"] = "1"


async def _run_profile_cohort(
    *,
    repo_root: Path,
    plan,
    ollama_url: str,
    resume: bool,
    finalize_only: bool,
) -> int:
    if plan.availability is ModelAvailability.MODEL_UNAVAILABLE:
        print(f"MODEL_UNAVAILABLE profile={plan.profile.profile_key}")
        return 0
    if not plan.profile.digest:
        print(f"BLOCKED: digest missing profile={plan.profile.profile_key}")
        return int(QualificationCliExit.BLOCKED_PRECONDITION)

    _apply_runtime_env(plan.profile.provider, plan.profile.model_name)
    bootstrap_qualification_environment(start_path=repo_root)
    head_sha = resolve_repository_head_sha(repo_root)
    spec, config_fp, frozen = build_qualification_spec_for_profile(
        repo_root,
        plan.profile,
        repository_head_sha=head_sha,
        run_count=plan.run_count,
    )
    session_dir = plan.session_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    probe = OllamaProviderIdentityProbe(OllamaProbeConfig(base_url=ollama_url))
    result = await run_local_ai_incident_qualification(
        repo_root=repo_root,
        session_dir=session_dir,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=frozen.semantic_fingerprint(),
        provider_probe=probe,
        run_executor=AiIncidentSingleRunExecutor(),
        resume=resume,
        finalize_only=finalize_only,
        repository_head_sha=head_sha,
        task_id=R6_TASK_ID,
        temperature=plan.profile.temperature,
    )
    print(
        f"profile={plan.profile.profile_key} session_state={result.session_state.value}"
    )
    return int(result.exit_code)


def _resolve_profiles(args: argparse.Namespace) -> tuple | None:
    if args.profile_key and args.all_models:
        print("BLOCKED: use either --profile-key or --all-models")
        return None
    if args.profile_key:
        selected = profile_by_key(args.profile_key)
        if selected is None:
            print(f"BLOCKED: unknown profile_key={args.profile_key}")
            return None
        return (selected,)
    return QualificationRegistry.profiles()


async def _run_all_cohorts(args: argparse.Namespace) -> int:
    profiles = _resolve_profiles(args)
    if profiles is None:
        return int(QualificationCliExit.BLOCKED_PRECONDITION)
    env_digest = (
        os.environ.get("INTERGRAX_QUALIFICATION_MODEL_DIGEST", "") or None
        if args.profile_key
        else None
    )

    plans = build_cohort_plans(
        args.repo_root,
        profiles,
        run_count=args.run_count,
        ollama_base_url=args.ollama_url,
        env_digest=env_digest,
    )
    worst = 0
    for plan in plans:
        code = await _run_profile_cohort(
            repo_root=args.repo_root,
            plan=plan,
            ollama_url=args.ollama_url,
            resume=args.resume,
            finalize_only=args.finalize_only,
        )
        worst = max(worst, code)
    return worst


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
    profiles = _resolve_profiles(args)
    if profiles is None:
        return int(QualificationCliExit.BLOCKED_PRECONDITION)
    plans = build_cohort_plans(
        args.repo_root,
        profiles,
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
