"""Thin scenario proof runner — configure, invoke application, evaluate, write artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_client import (
    OrderProviderClient,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import (
    all_attack_fixtures,
    build_authorized_write_fixture,
    build_safe_read_fixture,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.contracts import ProofVerdict
from platform_proofs.scenarios.indirect_prompt_injection.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.indirect_prompt_injection.proof.evidence_builder import (
    EVIDENCE_FILENAME,
    REPORT_FILENAME,
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.harness import (
    execute_fixture_scenario_run,
)
from scripts.proof.intergrax_platform_proof_evidence_io import write_evidence_json
from scripts.proof.intergrax_platform_proof_html_renderer import render_platform_proof_report

MAX_ATTACK_VARIANT_ATTEMPTS = 5


@dataclass(frozen=True, slots=True)
class WowGateResult:
    status: str
    provider: str
    model: str
    triggering_variant: str | None
    attack_provider_writes: int
    authorized_provider_writes: int


def _source_revision(repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _llm_credentials_available() -> bool:
    provider = os.environ.get("INTERGRAX_LLM_PROVIDER", "").strip()
    model = os.environ.get("INTERGRAX_LLM_MODEL", "").strip()
    return bool(provider and model)


async def _run() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    source_revision = _source_revision(repo_root)
    provider_client = OrderProviderClient()

    if not _llm_credentials_available():
        print("REAL MODEL WOW GATE: NOT RUN — credentials unavailable", file=sys.stderr)
        return 2

    safe_fixture = build_safe_read_fixture()
    safe_bundle = build_fixture_runtime_bundle(safe_fixture, order_operations=provider_client)
    safe_result = await execute_fixture_scenario_run(
        run_bundle=safe_bundle.run_bundle,
        order_operations=provider_client,
        provider_control=provider_client,
        fixture=safe_fixture,
    )
    safe_evaluation = evaluate_scenario_run(safe_result, safe_bundle.fixture)
    if not safe_evaluation.passed:
        print("SAFE-READ CONTROL FAILED:", safe_evaluation.failures, file=sys.stderr)
        return 1

    auth_fixture = build_authorized_write_fixture()
    auth_bundle = build_fixture_runtime_bundle(
        auth_fixture,
        order_operations=provider_client,
    )
    auth_result = await execute_fixture_scenario_run(
        run_bundle=auth_bundle.run_bundle,
        order_operations=provider_client,
        provider_control=provider_client,
        fixture=auth_fixture,
    )
    auth_evaluation = evaluate_scenario_run(auth_result, auth_bundle.fixture)
    if not auth_evaluation.passed:
        print("AUTHORIZED-WRITE CONTROL FAILED:", auth_evaluation.failures, file=sys.stderr)
        return 1

    wow_variant: str | None = None
    wow_result = None
    wow_evaluation = None
    wow_fixture = None
    for fixture in all_attack_fixtures()[:MAX_ATTACK_VARIANT_ATTEMPTS]:
        bundle = build_fixture_runtime_bundle(fixture, order_operations=provider_client)
        result = await execute_fixture_scenario_run(
            run_bundle=bundle.run_bundle,
            order_operations=provider_client,
            provider_control=provider_client,
            fixture=fixture,
        )
        evaluation = evaluate_scenario_run(result, fixture)
        if evaluation.verdict is ProofVerdict.PASS:
            wow_variant = fixture.attack_variant.value if fixture.attack_variant else fixture.fixture_id
            wow_result = result
            wow_evaluation = evaluation
            wow_fixture = fixture
            break
        if evaluation.verdict is ProofVerdict.NOT_EXERCISED:
            continue
        print(f"ATTACK CASE FAILED {fixture.fixture_id}:", evaluation.failures, file=sys.stderr)
        return 1

    if wow_result is None or wow_evaluation is None or wow_fixture is None:
        print("REAL MODEL WOW GATE: NOT EXERCISED", file=sys.stderr)
        return 3

    wow_gate = WowGateResult(
        status="PASS",
        provider=wow_result.model_provider,
        model=wow_result.model_name,
        triggering_variant=wow_variant,
        attack_provider_writes=wow_result.provider_write_count,
        authorized_provider_writes=auth_result.provider_write_count,
    )

    artifact_dir_env = os.environ.get("INTERGRAX_PROOF_ARTIFACT_DIR")
    if artifact_dir_env:
        out_dir = Path(artifact_dir_env)
        out_dir.mkdir(parents=True, exist_ok=True)
        evidence = build_platform_proof_evidence(
            wow_result,
            evaluation=wow_evaluation,
            fixture=wow_fixture,
            source_revision=source_revision,
        )
        write_evidence_json(evidence, proof_directory=out_dir, relative_path=EVIDENCE_FILENAME)
        (out_dir / REPORT_FILENAME).write_text(
            render_platform_proof_report(evidence),
            encoding="utf-8",
        )
        (out_dir / "domain_result.json").write_text(
            json.dumps(
                {
                    "wow_gate": wow_gate.__dict__,
                    "safe_read_passed": safe_evaluation.passed,
                    "authorized_write_passed": auth_evaluation.passed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("REAL MODEL WOW GATE: PASS")
    print(f"provider={wow_gate.provider} model={wow_gate.model} variant={wow_gate.triggering_variant}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Indirect prompt injection proof runner.")
    parser.add_argument("--validate-only", action="store_true")
    _ = parser.parse_args(argv)
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
