# © Artur Czarnecki. All rights reserved.

from __future__ import annotations
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import build_fixture_runtime_bundle, build_runtime_bundle

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from intergrax.contracts.evidence_claims import ClaimResolution, EvidenceClaimSet
from scripts.proof.intergrax_platform_proof_evidence import PlatformProofEvidence
from scripts.proof.intergrax_platform_proof_execution import (
    INTERGRAX_PROOF_ARTIFACT_DIR_ENV,
    load_manifest_bundle,
)
from scripts.proof.intergrax_proof_contracts import (
    EvidenceVerificationStatus as ContractEvidenceStatus,
    ProofStatus,
)
from scripts.proof.intergrax_proof_runner import execute_proof, read_git_metadata
from platform_proofs.scenarios.ai_incident_investigation.proof.evidence_builder import (
    EVIDENCE_RESOLVED_FILENAME,
    EVIDENCE_UNRESOLVED_FILENAME,
    PROOF_ID,
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.application.incident_reasoning import (
    claim_id_for_hypothesis,
    parse_claim_hypothesis_bindings,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    H3_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_UNRESOLVED,
    execute_resolved_skeleton,
)

pytestmark = pytest.mark.unit


def _skeleton_manifest_entry(repo_root: Path):
    bundle = load_manifest_bundle(repo_root=repo_root)
    entry = next(item for item in bundle.manifest.entries if item.proof_id == PROOF_ID)
    return entry, bundle.execution_specs[PROOF_ID]


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


@pytest.mark.asyncio
async def test_unresolved_canonical_evidence_content() -> None:
    fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    bundle = fixture_bundle.bundle
    result = await execute_resolved_skeleton(bundle)
    evaluation = evaluate_scenario_run(result, fixture_bundle.fixture)
    evidence = build_platform_proof_evidence(
        result,
        variant=ScenarioVariant.UNRESOLVED,
        evaluation=evaluation,
        source_revision="testsha",
    )
    parsed = PlatformProofEvidence.model_validate(evidence.model_dump(mode="json"))
    claim_set = EvidenceClaimSet.model_validate(result.claim_set)

    assert parsed.scenarios[0].execution_status.value == "PASS"
    assert not any(
        claim.resolution is ClaimResolution.SUPPORTED for claim in claim_set.claims
    )
    bindings = parse_claim_hypothesis_bindings(result.claim_hypothesis_bindings)
    h3_claim_id = claim_id_for_hypothesis(bindings, "H3")
    assert h3_claim_id is not None
    h3 = next(c for c in claim_set.claims if str(c.claim_id) == h3_claim_id)
    assert h3.resolution is ClaimResolution.INSUFFICIENT_EVIDENCE
    if parsed.evidence_claims.challenges:
        assert parsed.evidence_claims.challenges[0].resolution.value == "open"

    telemetry_node = next(
        node
        for node in parsed.evidence_graph.nodes
        if node.evidence_id == str(TELEMETRY_EVIDENCE_ID)
    )
    assert "unavailable" in telemetry_node.summary.text.lower()
    assert result.outcome == OUTCOME_UNRESOLVED
    assert parsed.provenance.source_revision == "testsha"
    assert "unresolved" in parsed.provenance.execution_id


@pytest.mark.asyncio
async def test_resolved_and_unresolved_artifact_identities_distinct() -> None:
    resolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    resolved_bundle = resolved_fixture_bundle.bundle
    unresolved_fixture_bundle = build_fixture_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    unresolved_bundle = unresolved_fixture_bundle.bundle
    resolved_result = await execute_resolved_skeleton(resolved_bundle)
    unresolved_result = await execute_resolved_skeleton(unresolved_bundle)
    resolved_evaluation = evaluate_scenario_run(resolved_result, resolved_fixture_bundle.fixture)
    unresolved_evaluation = evaluate_scenario_run(unresolved_result, unresolved_fixture_bundle.fixture)
    resolved_evidence = build_platform_proof_evidence(
        resolved_result,
        evaluation=resolved_evaluation,
        variant=ScenarioVariant.RESOLVED,
        source_revision="same-rev",
    )
    unresolved_evidence = build_platform_proof_evidence(
        unresolved_result,
        evaluation=unresolved_evaluation,
        variant=ScenarioVariant.UNRESOLVED,
        source_revision="same-rev",
    )
    assert resolved_evidence.provenance.execution_id != unresolved_evidence.provenance.execution_id
    assert resolved_evidence.provenance.artifact_identity != unresolved_evidence.provenance.artifact_identity
    assert resolved_evidence.proof_identity.source_revision == "same-rev"
    assert unresolved_evidence.proof_identity.source_revision == "same-rev"


@pytest.mark.asyncio
async def test_parent_runner_integration_both_canonical_evidence_valid(repo_root: Path) -> None:
    entry, spec = _skeleton_manifest_entry(repo_root)
    git = read_git_metadata(repo_root)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        result = execute_proof(
            entry,
            repo_root=repo_root,
            execution_spec=spec,
            proof_artifact_directory=artifact_dir,
            git_commit_sha=git.commit_sha,
        )
        assert result.exit_code == 0
        assert result.status == ProofStatus.PASS
        assert result.evidence_verification_status == ContractEvidenceStatus.PASS
        assert (artifact_dir / EVIDENCE_RESOLVED_FILENAME).is_file()
        assert (artifact_dir / EVIDENCE_UNRESOLVED_FILENAME).is_file()


@pytest.mark.asyncio
async def test_parent_runner_missing_unresolved_evidence_fails(repo_root: Path) -> None:
    entry, spec = _skeleton_manifest_entry(repo_root)
    git = read_git_metadata(repo_root)

    def _remove_unresolved(command, **kwargs):
        completed = subprocess.run(command, **kwargs)
        artifact_dir = Path(kwargs["env"][INTERGRAX_PROOF_ARTIFACT_DIR_ENV])
        (artifact_dir / EVIDENCE_UNRESOLVED_FILENAME).unlink(missing_ok=True)
        return completed

    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        result = execute_proof(
            entry,
            repo_root=repo_root,
            execution_spec=spec,
            proof_artifact_directory=artifact_dir,
            git_commit_sha=git.commit_sha,
            subprocess_runner=_remove_unresolved,
        )
    assert result.status == ProofStatus.FAIL
    assert result.evidence_verification_status == ContractEvidenceStatus.MISSING


@pytest.mark.asyncio
async def test_parent_runner_tampered_unresolved_evidence_fails(repo_root: Path) -> None:
    entry, spec = _skeleton_manifest_entry(repo_root)
    git = read_git_metadata(repo_root)

    def _tamper_unresolved(command, **kwargs):
        completed = subprocess.run(command, **kwargs)
        artifact_dir = Path(kwargs["env"][INTERGRAX_PROOF_ARTIFACT_DIR_ENV])
        evidence_path = artifact_dir / EVIDENCE_UNRESOLVED_FILENAME
        tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
        for claim in tampered["evidence_claims"]["claims"]:
            if claim.get("resolution") == "insufficient_evidence":
                claim["resolution"] = "supported"
                break
        evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
        return completed

    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        result = execute_proof(
            entry,
            repo_root=repo_root,
            execution_spec=spec,
            proof_artifact_directory=artifact_dir,
            git_commit_sha=git.commit_sha,
            subprocess_runner=_tamper_unresolved,
        )
    assert result.status == ProofStatus.FAIL
    assert result.evidence_verification_status in {
        ContractEvidenceStatus.INVALID,
        ContractEvidenceStatus.FAIL,
    }


@pytest.mark.asyncio
async def test_parent_runner_tampered_resolved_evidence_fails(repo_root: Path) -> None:
    entry, spec = _skeleton_manifest_entry(repo_root)
    git = read_git_metadata(repo_root)

    def _tamper_resolved(command, **kwargs):
        completed = subprocess.run(command, **kwargs)
        artifact_dir = Path(kwargs["env"][INTERGRAX_PROOF_ARTIFACT_DIR_ENV])
        evidence_path = artifact_dir / EVIDENCE_RESOLVED_FILENAME
        tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
        tampered["evidence_claims"]["claims"][0]["supporting_evidence_ids"] = [
            "evidence.tampered.missing"
        ]
        evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
        return completed

    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = Path(tmp)
        result = execute_proof(
            entry,
            repo_root=repo_root,
            execution_spec=spec,
            proof_artifact_directory=artifact_dir,
            git_commit_sha=git.commit_sha,
            subprocess_runner=_tamper_resolved,
        )
    assert result.status == ProofStatus.FAIL
    assert result.evidence_verification_status == ContractEvidenceStatus.INVALID
