# © Artur Czarnecki. All rights reserved.
# ruff: noqa: E402

"""DS-E2E-15J-QI2 canonical local AI Incident behavioral qualification entrypoint."""

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
    R4R1_PROFILE_ID,
    R4R1_PROVIDER,
    R4R1ProfileParams,
    build_r4r1_qualification_spec,
    resolve_repository_head_sha,
    run_local_ai_incident_qualification,
)
from testing_support.decision_e2e.local_qualification_session.ollama_probe import OllamaProbeConfig


def _apply_profile_env(provider: str, model: str) -> None:
    os.environ["INTERGRAX_LLM_PROVIDER"] = provider
    os.environ["INTERGRAX_LLM_MODEL"] = model
    os.environ["INTERGRAX_DECISION_E2E_QUALIFICATION"] = "1"


async def _async_main(args: argparse.Namespace) -> int:
    if args.profile != R4R1_PROFILE_ID:
        print(f"unsupported profile: {args.profile}")
        return int(QualificationCliExit.CRITICAL_SAFETY_FAILURE)

    provider = args.provider or R4R1_PROVIDER
    model = args.model or R4R1_MODEL_NAME
    _apply_profile_env(provider, model)
    bootstrap_qualification_environment(start_path=_REPO_ROOT)

    digest = os.environ.get("INTERGRAX_QUALIFICATION_MODEL_DIGEST", "")
    if not digest:
        probe = OllamaProviderIdentityProbe(
            OllamaProbeConfig(base_url=args.ollama_url),
        )
        observed = probe.probe(model_name=model)
        if observed is None or observed.model_digest is None:
            print("BLOCKED: model digest unavailable; set INTERGRAX_QUALIFICATION_MODEL_DIGEST")
            return int(QualificationCliExit.BLOCKED_PRECONDITION)
        digest = observed.model_digest

    head_sha = resolve_repository_head_sha(_REPO_ROOT)
    params = R4R1ProfileParams(
        model_digest=digest,
        run_count=args.run_count,
    )
    spec, config_fp, frozen = build_r4r1_qualification_spec(
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
    )
    print(f"session_state={result.session_state.value}")
    return int(result.exit_code)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DS-E2E-15J local behavioral qualification (QI2)",
    )
    parser.add_argument(
        "--profile",
        default=R4R1_PROFILE_ID,
        help="Frozen qualification profile identifier",
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--run-count", type=int, default=1)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
