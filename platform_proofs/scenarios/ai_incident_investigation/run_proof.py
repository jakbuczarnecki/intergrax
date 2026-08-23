# © Artur Czarnecki. All rights reserved.

"""Platform Proof entry — skeleton only; not a published public proof."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.proof.intergrax_platform_proof_evidence_io import write_evidence_json
from scripts.proof.intergrax_platform_proof_evidence_verifier import (
    EvidenceVerificationStatus,
    verify_platform_proof_evidence,
)
from scripts.proof.intergrax_platform_proof_execution import ProofExecutionSpec
from scripts.proof.intergrax_platform_proof_html_renderer import render_platform_proof_report
from scripts.proof.intergrax_proof_contracts import (
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofRunResult,
    ProofSafetyClass,
    ProofStatus,
)
from platform_proofs.scenarios.ai_incident_investigation.evidence_builder import (
    PROOF_ID,
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


def _skeleton_execution_spec() -> ProofExecutionSpec:
    return ProofExecutionSpec(
        manifest_entry=ProofManifestEntry(
            proof_id=PROOF_ID,
            title="AI Incident Investigation — platform-native skeleton",
            profiles=frozenset({ProofProfile.QUICK}),
            proof_kind="scenario_skeleton",
            command=ProofArgvCommand(
                executable="python",
                argv=("platform_proofs/scenarios/ai_incident_investigation/run_proof.py",),
            ),
            safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
        ),
        evidence_required=True,
    )


def _verify_written_evidence(
    *,
    artifact_dir: Path,
    evidence_path: Path,
    source_revision: str,
) -> int:
    transport = ProofRunResult(
        proof_id=PROOF_ID,
        status=ProofStatus.PASS,
        exit_code=0,
        duration_seconds=0.0,
    )
    verification = verify_platform_proof_evidence(
        evidence_path=evidence_path,
        artifact_root=artifact_dir,
        spec=_skeleton_execution_spec(),
        subprocess_result=transport,
        expected_source_revision=source_revision,
    )
    if verification.status is not EvidenceVerificationStatus.PASS:
        print(
            "CANONICAL EVIDENCE VERIFICATION FAILED:",
            verification.diagnostic_code,
            verification.diagnostic_summary,
            file=sys.stderr,
        )
        return 1
    return 0


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
        evidence_path = write_evidence_json(evidence, proof_directory=out_dir)
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
        verify_code = _verify_written_evidence(
            artifact_dir=out_dir,
            evidence_path=evidence_path,
            source_revision=source_revision,
        )
        if verify_code != 0:
            return verify_code
    else:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            evidence_path = write_evidence_json(evidence, proof_directory=out_dir)
            verify_code = _verify_written_evidence(
                artifact_dir=out_dir,
                evidence_path=evidence_path,
                source_revision=source_revision,
            )
            if verify_code != 0:
                return verify_code

    print("SCENARIO-AI-INCIDENT-INVESTIGATION-SKELETON: PASS")
    return 0


def main() -> int:
    import asyncio

    return asyncio.run(_run_skeleton())


if __name__ == "__main__":
    raise SystemExit(main())
