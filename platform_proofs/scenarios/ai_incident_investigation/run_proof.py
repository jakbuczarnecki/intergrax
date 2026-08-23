# © Artur Czarnecki. All rights reserved.

"""Platform Proof entry — skeleton only; not a published public proof."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.proof.intergrax_platform_proof_evidence_io import write_evidence_json
from scripts.proof.intergrax_platform_proof_html_renderer import render_platform_proof_report
from platform_proofs.scenarios.ai_incident_investigation.evidence_builder import (
    EVIDENCE_RESOLVED_FILENAME,
    EVIDENCE_UNRESOLVED_FILENAME,
    REPORT_RESOLVED_FILENAME,
    REPORT_UNRESOLVED_FILENAME,
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.fixtures import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    build_runtime_bundle,
    execute_resolved_skeleton,
)


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


async def _run_skeleton() -> int:
    resolved_bundle = build_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    resolved_result = await execute_resolved_skeleton(resolved_bundle)
    resolved_evaluation = evaluate_scenario_run(resolved_result, resolved_bundle.fixture)
    if not resolved_evaluation.passed:
        print("SCENARIO FULL-1 EVALUATION FAILED:", resolved_evaluation.failures, file=sys.stderr)
        return 1

    unresolved_bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    unresolved_result = await execute_resolved_skeleton(unresolved_bundle)
    unresolved_evaluation = evaluate_scenario_run(unresolved_result, unresolved_bundle.fixture)
    if not unresolved_evaluation.passed:
        print("SCENARIO FULL-2 EVALUATION FAILED:", unresolved_evaluation.failures, file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[3]
    source_revision = _source_revision(repo_root)
    resolved_evidence = build_platform_proof_evidence(
        resolved_result,
        variant=ScenarioVariant.RESOLVED,
        source_revision=source_revision,
    )
    unresolved_evidence = build_platform_proof_evidence(
        unresolved_result,
        variant=ScenarioVariant.UNRESOLVED,
        source_revision=source_revision,
    )

    artifact_dir_env = os.environ.get("INTERGRAX_PROOF_ARTIFACT_DIR")
    if artifact_dir_env:
        out_dir = Path(artifact_dir_env)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_evidence_json(
            resolved_evidence,
            proof_directory=out_dir,
            relative_path=EVIDENCE_RESOLVED_FILENAME,
        )
        write_evidence_json(
            unresolved_evidence,
            proof_directory=out_dir,
            relative_path=EVIDENCE_UNRESOLVED_FILENAME,
        )
        (out_dir / REPORT_RESOLVED_FILENAME).write_text(
            render_platform_proof_report(resolved_evidence),
            encoding="utf-8",
        )
        (out_dir / REPORT_UNRESOLVED_FILENAME).write_text(
            render_platform_proof_report(unresolved_evidence),
            encoding="utf-8",
        )
        (out_dir / "domain_result.json").write_text(
            json.dumps(
                {
                    "resolved": {
                        "outcome": resolved_result.outcome,
                        "checks": list(resolved_evaluation.checks),
                    },
                    "unresolved": {
                        "outcome": unresolved_result.outcome,
                        "checks": list(unresolved_evaluation.checks),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("SCENARIO-AI-INCIDENT-INVESTIGATION-FULL-1: PASS")
    print("SCENARIO-AI-INCIDENT-INVESTIGATION-FULL-2: PASS")
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(_run_skeleton())


if __name__ == "__main__":
    raise SystemExit(main())
