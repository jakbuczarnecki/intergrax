# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""CLI: DS-E2E-15J-L1.R4.R5 natural local model alignment qualification."""

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
    R4R1_MODEL_NAME,
    R4R1_PROVIDER,
    R4R1ProfileParams,
    R4R1_RUN_COUNT,
    R4R5_TASK_ID,
    build_r4r5_qualification_spec,
    resolve_repository_head_sha,
    run_local_ai_incident_qualification,
)
from testing_support.decision_e2e.local_qualification_session.ollama_probe import (
    OllamaProbeConfig,
)
from testing_support.decision_e2e.natural_alignment.analysis import (
    COHORT_TASK_ID,
    run_natural_qualification_analysis,
)
from testing_support.decision_e2e.natural_alignment.source_freeze import (
    verify_natural_alignment_source_freeze,
    write_natural_alignment_source_freeze_baseline,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)


def _default_session_dir(repo_root: Path) -> Path:
    return repo_root / ".artifacts" / "qualification" / COHORT_TASK_ID


def _apply_runtime_env() -> None:
    os.environ["INTERGRAX_LLM_PROVIDER"] = R4R1_PROVIDER
    os.environ["INTERGRAX_LLM_MODEL"] = R4R1_MODEL_NAME
    os.environ["INTERGRAX_DECISION_E2E_QUALIFICATION"] = "1"


async def _run_cohort(args: argparse.Namespace) -> int:
    _apply_runtime_env()
    bootstrap_qualification_environment(start_path=_REPO_ROOT)
    digest = os.environ.get("INTERGRAX_QUALIFICATION_MODEL_DIGEST", "")
    if not digest:
        probe = OllamaProviderIdentityProbe(OllamaProbeConfig(base_url=args.ollama_url))
        observed = probe.probe(model_name=R4R1_MODEL_NAME)
        if observed is None or observed.model_digest is None:
            print("BLOCKED: model digest unavailable; set INTERGRAX_QUALIFICATION_MODEL_DIGEST")
            return int(QualificationCliExit.BLOCKED_PRECONDITION)
        digest = observed.model_digest
        print(f"model_digest={digest}")

    head_sha = resolve_repository_head_sha(_REPO_ROOT)
    params = R4R1ProfileParams(
        model_digest=digest,
        run_count=args.run_count,
    )
    spec, config_fp, frozen = build_r4r5_qualification_spec(
        _REPO_ROOT,
        params=params,
        repository_head_sha=head_sha,
    )
    session_dir = args.session_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    probe = OllamaProviderIdentityProbe(OllamaProbeConfig(base_url=args.ollama_url))
    result = await run_local_ai_incident_qualification(
        repo_root=_REPO_ROOT,
        session_dir=session_dir,
        spec=spec,
        config_fingerprint=config_fp,
        source_fingerprint=frozen.semantic_fingerprint(),
        provider_probe=probe,
        run_executor=AiIncidentSingleRunExecutor(),
        resume=args.resume,
        finalize_only=args.finalize_only,
        repository_head_sha=head_sha,
        task_id=R4R5_TASK_ID,
    )
    print(f"session_state={result.session_state.value}")
    return int(result.exit_code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=COHORT_TASK_ID)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Qualification artifact root",
    )
    parser.add_argument("--run-count", type=int, default=R4R1_RUN_COUNT)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--write-source-freeze-baseline", action="store_true")
    parser.add_argument(
        "--source-freeze-only",
        action="store_true",
        help="Phase 1 repository source freeze verification only",
    )
    parser.add_argument(
        "--cohort-only",
        action="store_true",
        help="Phase 3 natural cohort execution only",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Phase 5–7 analysis and artifact emission only",
    )
    args = parser.parse_args(argv)
    args.session_dir = args.session_dir or _default_session_dir(args.repo_root)

    if args.write_source_freeze_baseline:
        path = write_natural_alignment_source_freeze_baseline(args.repo_root)
        print(f"baseline_written={path}")
        return 0

    freeze = verify_natural_alignment_source_freeze(args.repo_root)
    print(f"SOURCE_FREEZE_STATUS={freeze.status.value}")
    if freeze.status is not SourceFreezeStatus.PASS:
        return 2
    if args.source_freeze_only:
        return 0

    if args.analyze_only:
        if not (args.session_dir / "runs.json").is_file():
            print("BLOCKED: session runs.json missing")
            return 3
        result = run_natural_qualification_analysis(
            repo_root=args.repo_root,
            session_dir=args.session_dir,
        )
        print(f"15K_B_EFFECT={result.fifteen_kb_effect.value}")
        print(f"qualification_outcome={result.qualification_outcome.value}")
        print(f"output_dir={result.output_dir}")
        if result.qualification_outcome.value == "FAIL":
            return 4
        return 0

    if args.cohort_only or not args.analyze_only:
        exit_code = asyncio.run(_run_cohort(args))
        if exit_code != 0:
            return exit_code
        if args.cohort_only:
            return 0

    result = run_natural_qualification_analysis(
        repo_root=args.repo_root,
        session_dir=args.session_dir,
    )
    print(f"15K_B_EFFECT={result.fifteen_kb_effect.value}")
    print(f"qualification_outcome={result.qualification_outcome.value}")
    print(f"output_dir={result.output_dir}")
    if result.qualification_outcome.value == "FAIL":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
