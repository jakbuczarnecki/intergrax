# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

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
from platform_proofs.scenarios.ai_incident_investigation.evidence_builder import (
    EVIDENCE_RESOLVED_FILENAME,
    EVIDENCE_UNRESOLVED_FILENAME,
    PROOF_ID,
    build_platform_proof_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    H3_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    OUTCOME_UNRESOLVED,
    build_runtime_bundle,
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
    bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    result = await execute_resolved_skeleton(bundle)
    evidence = build_platform_proof_evidence(
        result,
        variant=ScenarioVariant.UNRESOLVED,
        source_revision="testsha",
    )
    parsed = PlatformProofEvidence.model_validate(evidence.model_dump(mode="json"))
    claim_set = EvidenceClaimSet.model_validate(result.claim_set)

    assert parsed.scenarios[0].execution_status.value == "PASS"
    assert not any(
        claim.resolution is ClaimResolution.SUPPORTED for claim in claim_set.claims
    )
    h3 = next(c for c in claim_set.claims if c.claim_id == H3_CLAIM_ID)
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
    resolved_bundle = build_runtime_bundle(variant=ScenarioVariant.RESOLVED)
    unresolved_bundle = build_runtime_bundle(variant=ScenarioVariant.UNRESOLVED)
    resolved_result = await execute_resolved_skeleton(resolved_bundle)
    unresolved_result = await execute_resolved_skeleton(unresolved_bundle)
    resolved_evidence = build_platform_proof_evidence(
        resolved_result,
        variant=ScenarioVariant.RESOLVED,
        source_revision="same-rev",
    )
    unresolved_evidence = build_platform_proof_evidence(
        unresolved_result,
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
            if claim.get("claim_id") == str(H3_CLAIM_ID):
                claim["resolution"] = "supported"
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
