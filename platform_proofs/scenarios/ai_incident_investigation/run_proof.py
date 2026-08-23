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
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.evaluator import evaluate_scenario_run
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
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    evaluation = evaluate_scenario_run(result, bundle.fixture)
    if not evaluation.passed:
        print("SCENARIO SKELETON EVALUATION FAILED:", evaluation.failures, file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[3]
    source_revision = _source_revision(repo_root)
    evidence = build_platform_proof_evidence(
        result,
        source_revision=source_revision,
    )

    artifact_dir_env = os.environ.get("INTERGRAX_PROOF_ARTIFACT_DIR")
    if artifact_dir_env:
        out_dir = Path(artifact_dir_env)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_evidence_json(evidence, proof_directory=out_dir)
        report_html = render_platform_proof_report(evidence)
        (out_dir / "report.html").write_text(report_html, encoding="utf-8")
        (out_dir / "domain_result.json").write_text(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "checks": list(evaluation.checks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("SCENARIO-AI-INCIDENT-INVESTIGATION-SKELETON: PASS")
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(_run_skeleton())


if __name__ == "__main__":
    raise SystemExit(main())
