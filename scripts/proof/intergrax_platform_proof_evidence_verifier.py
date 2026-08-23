# © Artur Czarnecki. All rights reserved.

"""Machine verification of Platform Proof evidence artifacts (PP-SUITE-3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from scripts.proof.intergrax_platform_proof_descriptor import ExpectedArtifactKind
from scripts.proof.intergrax_platform_proof_evidence import (
    PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION,
    PlatformProofEvidence,
    ProofEvidenceExecutionStatus,
    iter_evidence_claim_graph_binding_violations,
)
from scripts.proof.intergrax_platform_proof_evidence_io import (
    compute_evidence_checksum,
    evidence_payload_dict,
)
from scripts.proof.intergrax_platform_proof_execution import ProofExecutionSpec
from scripts.proof.intergrax_proof_contracts import (
    EvidenceVerificationStatus,
    ProofRunResult,
    ProofStatus,
)


@dataclass(frozen=True, slots=True)
class EvidenceVerificationResult:
    status: EvidenceVerificationStatus
    proof_id: str
    evidence_path: Path | None
    diagnostic_code: str
    diagnostic_summary: str
    parsed_evidence: PlatformProofEvidence | None = None


def resolve_expected_evidence_path(
    proof_artifact_directory: Path,
    spec: ProofExecutionSpec,
) -> Path:
    relative = "evidence.json"
    for artifact in spec.expected_artifacts:
        if artifact.kind == ExpectedArtifactKind.EVIDENCE_JSON:
            relative = artifact.relative_path
            break
    return proof_artifact_directory / relative


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _bounded_message(prefix: str, detail: str, *, limit: int = 200) -> str:
    text = f"{prefix}: {detail}"
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _verify_checksum(evidence: PlatformProofEvidence) -> bool:
    checksum = evidence.provenance.evidence_checksum
    if checksum is None:
        return True
    payload = evidence_payload_dict(evidence)
    provenance = dict(payload["provenance"])
    provenance["evidence_checksum"] = None
    payload["provenance"] = provenance
    return checksum == compute_evidence_checksum(payload)


def _evidence_claim_bindings_consistent(
    evidence: PlatformProofEvidence,
) -> tuple[bool, str]:
    evidence_graph_ids = frozenset(
        node.evidence_id for node in evidence.evidence_graph.nodes
    )
    violations = iter_evidence_claim_graph_binding_violations(
        evidence.evidence_claims,
        evidence_graph_ids,
    )
    if not violations:
        return True, ""
    diagnostic_code = violations[0].split(":", 1)[0]
    return False, diagnostic_code


def _evaluator_consistent(evidence: PlatformProofEvidence) -> tuple[bool, str]:
    if evidence.execution.status != ProofEvidenceExecutionStatus.PASS:
        return True, ""

    if evidence.evaluator is not None and not evidence.evaluator.passed:
        return False, "evaluator_passed_false"

    for scenario in evidence.scenarios:
        if scenario.execution_status != ProofEvidenceExecutionStatus.PASS:
            continue
        if scenario.evaluator is not None and not scenario.evaluator.passed:
            return False, f"scenario_evaluator_false:{scenario.scenario_id}"

    return True, ""


def _transport_consistent(
    exit_code: int | None,
    execution_status: ProofEvidenceExecutionStatus,
) -> tuple[bool, str]:
    if exit_code is None:
        return True, ""

    if exit_code == 0:
        if execution_status != ProofEvidenceExecutionStatus.PASS:
            return False, "evidence_status_mismatch"
        return True, ""

    if execution_status == ProofEvidenceExecutionStatus.PASS:
        return False, "transport_evidence_mismatch"

    if execution_status not in {
        ProofEvidenceExecutionStatus.FAIL,
        ProofEvidenceExecutionStatus.CRASH,
    }:
        return False, "evidence_status_mismatch"

    return True, ""


def verify_platform_proof_evidence(
    *,
    evidence_path: Path,
    artifact_root: Path,
    spec: ProofExecutionSpec,
    subprocess_result: ProofRunResult,
    expected_source_revision: str | None = None,
) -> EvidenceVerificationResult:
    proof_id = spec.manifest_entry.proof_id
    resolved_path = evidence_path.resolve()
    resolved_root = artifact_root.resolve()

    if not _path_within_root(resolved_path, resolved_root):
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="evidence_path_outside_artifact_root",
            diagnostic_summary="evidence path escapes proof artifact directory",
        )

    if not resolved_path.is_file():
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.MISSING,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="missing_required_evidence",
            diagnostic_summary="required evidence.json is missing",
        )

    try:
        raw_text = resolved_path.read_text(encoding="utf-8")
    except OSError as exc:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary=_bounded_message("evidence read failed", str(exc)),
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="evidence is not valid UTF-8 JSON",
        )

    try:
        evidence = PlatformProofEvidence.model_validate(payload)
    except ValidationError:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="evidence failed PlatformProofEvidence validation",
        )

    expected_schema = spec.evidence_schema or PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION
    if evidence.schema_version != expected_schema:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="evidence schema version mismatch",
            parsed_evidence=evidence,
        )

    if evidence.proof_identity.proof_id != proof_id:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="proof_identity.proof_id mismatch",
            parsed_evidence=evidence,
        )

    if spec.expected_domains_exercised is not None:
        if (
            evidence.proof_identity.domains_exercised
            != spec.expected_domains_exercised
        ):
            return EvidenceVerificationResult(
                status=EvidenceVerificationStatus.INVALID,
                proof_id=proof_id,
                evidence_path=resolved_path,
                diagnostic_code="proof_identity_domains_mismatch",
                diagnostic_summary="proof_identity.domains_exercised mismatch",
                parsed_evidence=evidence,
            )

    if evidence.provenance.proof_id != proof_id:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="provenance.proof_id mismatch",
            parsed_evidence=evidence,
        )

    if (
        expected_source_revision
        and expected_source_revision != "unknown"
        and evidence.proof_identity.source_revision != expected_source_revision
    ):
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="proof_identity.source_revision mismatch",
            parsed_evidence=evidence,
        )

    if evidence.provenance.source_revision != evidence.proof_identity.source_revision:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="provenance.source_revision mismatch",
            parsed_evidence=evidence,
        )

    if not _verify_checksum(evidence):
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code="invalid_evidence",
            diagnostic_summary="provenance.evidence_checksum mismatch",
            parsed_evidence=evidence,
        )

    bindings_ok, bindings_code = _evidence_claim_bindings_consistent(evidence)
    if not bindings_ok:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.INVALID,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code=bindings_code,
            diagnostic_summary=bindings_code,
            parsed_evidence=evidence,
        )

    transport_ok, transport_code = _transport_consistent(
        subprocess_result.exit_code,
        evidence.execution.status,
    )
    if not transport_ok:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.FAIL,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code=transport_code,
            diagnostic_summary=transport_code,
            parsed_evidence=evidence,
        )

    evaluator_ok, evaluator_code = _evaluator_consistent(evidence)
    if not evaluator_ok:
        return EvidenceVerificationResult(
            status=EvidenceVerificationStatus.FAIL,
            proof_id=proof_id,
            evidence_path=resolved_path,
            diagnostic_code=evaluator_code,
            diagnostic_summary=evaluator_code,
            parsed_evidence=evidence,
        )

    return EvidenceVerificationResult(
        status=EvidenceVerificationStatus.PASS,
        proof_id=proof_id,
        evidence_path=resolved_path,
        diagnostic_code="evidence_verified",
        diagnostic_summary="evidence_verified",
        parsed_evidence=evidence,
    )


def apply_evidence_verification(
    subprocess_result: ProofRunResult,
    verification: EvidenceVerificationResult,
) -> ProofRunResult:
    evidence_updates = {
        "evidence_verification_status": verification.status,
        "evidence_path": (
            verification.evidence_path.as_posix()
            if verification.evidence_path is not None
            else None
        ),
    }

    if verification.status == EvidenceVerificationStatus.PASS:
        if subprocess_result.status == ProofStatus.PASS:
            return subprocess_result.model_copy(
                update={
                    **evidence_updates,
                    "diagnostic_summary": verification.diagnostic_summary,
                }
            )
        return subprocess_result.model_copy(update=evidence_updates)

    child_diagnostic = subprocess_result.diagnostic_summary
    if (
        subprocess_result.status == ProofStatus.FAIL
        and child_diagnostic
        and verification.diagnostic_code == "evidence_status_mismatch"
        and subprocess_result.exit_code
        and subprocess_result.exit_code != 0
    ):
        diagnostic = child_diagnostic
    elif (
        subprocess_result.status == ProofStatus.FAIL
        and child_diagnostic
        and verification.diagnostic_code == "transport_evidence_mismatch"
    ):
        diagnostic = verification.diagnostic_summary
    else:
        diagnostic = verification.diagnostic_summary

    return subprocess_result.model_copy(
        update={
            **evidence_updates,
            "status": ProofStatus.FAIL,
            "diagnostic_summary": diagnostic,
        }
    )
