# © Artur Czarnecki. All rights reserved.

"""Fail-closed DS-E2E qualification constructors and proof requirements."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
    QualificationEvidenceRef,
)
from testing_support.decision_e2e.bindings import ProviderBindingEvidence
from testing_support.decision_e2e.evidence import (
    docker_crash_evidence_ref,
    provider_evidence_ref,
    scenario_execution_evidence_ref,
)
from testing_support.decision_e2e.independence import (
    council_requires_distinct_models,
    producer_verifier_requires_distinct_models,
)
from testing_support.decision_e2e.qualification_evidence import (
    DockerCrashEvidence,
    ScenarioExecutionEvidence,
)


@dataclass(frozen=True, slots=True)
class DecisionE2EProofRequirement:
    proof_id: DecisionE2EProofId
    requires_real_provider: bool
    requires_distinct_models: bool
    requires_docker_kill: bool
    requires_live_scenario: bool


PROOF_REQUIREMENTS: dict[DecisionE2EProofId, DecisionE2EProofRequirement] = {
    DecisionE2EProofId.DS_E2E_02: DecisionE2EProofRequirement(
        proof_id=DecisionE2EProofId.DS_E2E_02,
        requires_real_provider=True,
        requires_distinct_models=True,
        requires_docker_kill=False,
        requires_live_scenario=False,
    ),
    DecisionE2EProofId.DS_E2E_03: DecisionE2EProofRequirement(
        proof_id=DecisionE2EProofId.DS_E2E_03,
        requires_real_provider=True,
        requires_distinct_models=True,
        requires_docker_kill=False,
        requires_live_scenario=False,
    ),
    DecisionE2EProofId.DS_E2E_06: DecisionE2EProofRequirement(
        proof_id=DecisionE2EProofId.DS_E2E_06,
        requires_real_provider=False,
        requires_distinct_models=False,
        requires_docker_kill=True,
        requires_live_scenario=False,
    ),
    DecisionE2EProofId.DS_E2E_12: DecisionE2EProofRequirement(
        proof_id=DecisionE2EProofId.DS_E2E_12,
        requires_real_provider=True,
        requires_distinct_models=False,
        requires_docker_kill=False,
        requires_live_scenario=True,
    ),
    DecisionE2EProofId.DS_E2E_13: DecisionE2EProofRequirement(
        proof_id=DecisionE2EProofId.DS_E2E_13,
        requires_real_provider=True,
        requires_distinct_models=False,
        requires_docker_kill=False,
        requires_live_scenario=True,
    ),
}


def _blocked(
    proof_id: DecisionE2EProofId,
    reason: str,
    *,
    evidence: tuple[QualificationEvidenceRef, ...] = (),
) -> DecisionE2EQualificationResult:
    return DecisionE2EQualificationResult(
        proof_id=proof_id,
        disposition=QualificationDisposition.BLOCKED,
        evidence=evidence,
        reason=reason,
    )


def qualify_real_multi_model(
    *,
    council_bindings: tuple[ProviderBindingEvidence, ...],
    evidence: tuple[QualificationEvidenceRef, ...],
    reason: str | None = None,
) -> DecisionE2EQualificationResult:
    """DS-E2E-02 fail-closed constructor."""
    qualifies, block_reason = council_requires_distinct_models(council_bindings)
    if not qualifies:
        return _blocked(
            DecisionE2EProofId.DS_E2E_02,
            block_reason or "Council model independence prerequisite not met",
            evidence=evidence,
        )
    binding_evidence = tuple(provider_evidence_ref(binding) for binding in council_bindings)
    merged = binding_evidence + evidence
    return DecisionE2EQualificationResult(
        proof_id=DecisionE2EProofId.DS_E2E_02,
        disposition=QualificationDisposition.PASSED,
        evidence=merged,
        reason=reason,
    )


def qualify_independent_verifier(
    *,
    producer: ProviderBindingEvidence,
    verifier: ProviderBindingEvidence,
    evidence: tuple[QualificationEvidenceRef, ...],
    reason: str | None = None,
) -> DecisionE2EQualificationResult:
    """DS-E2E-03 fail-closed constructor."""
    qualifies, block_reason = producer_verifier_requires_distinct_models(producer, verifier)
    if not qualifies:
        return _blocked(
            DecisionE2EProofId.DS_E2E_03,
            block_reason or "Producer/verifier independence prerequisite not met",
            evidence=evidence,
        )
    merged = (
        provider_evidence_ref(producer),
        provider_evidence_ref(verifier),
    ) + evidence
    return DecisionE2EQualificationResult(
        proof_id=DecisionE2EProofId.DS_E2E_03,
        disposition=QualificationDisposition.PASSED,
        evidence=merged,
        reason=reason,
    )


def qualify_docker_crash_resume(
    *,
    crash_evidence: DockerCrashEvidence,
    evidence: tuple[QualificationEvidenceRef, ...] = (),
    reason: str | None = None,
) -> DecisionE2EQualificationResult:
    """DS-E2E-06 fail-closed constructor."""
    if crash_evidence.kill_method != "docker_kill":
        return _blocked(
            DecisionE2EProofId.DS_E2E_06,
            "DS-E2E-06 requires external docker kill evidence; subprocess-only proof is insufficient",
            evidence=evidence,
        )
    if not crash_evidence.killed_container_id:
        return _blocked(
            DecisionE2EProofId.DS_E2E_06,
            "Docker crash qualification missing killed container ID",
            evidence=evidence,
        )
    if not crash_evidence.resume_container_id:
        return _blocked(
            DecisionE2EProofId.DS_E2E_06,
            "Docker crash qualification missing resume container ID",
            evidence=evidence,
        )
    merged = (docker_crash_evidence_ref(crash_evidence),) + evidence
    return DecisionE2EQualificationResult(
        proof_id=DecisionE2EProofId.DS_E2E_06,
        disposition=QualificationDisposition.PASSED,
        evidence=merged,
        reason=reason,
    )


def qualify_live_scenario(
    *,
    proof_id: DecisionE2EProofId,
    scenario_evidence: ScenarioExecutionEvidence,
    evidence: tuple[QualificationEvidenceRef, ...] = (),
    reason: str | None = None,
) -> DecisionE2EQualificationResult:
    """DS-E2E-12/13 fail-closed constructor for live scenario execution."""
    if not scenario_evidence.executed:
        return _blocked(
            proof_id,
            scenario_evidence.block_reason or "Live scenario execution evidence missing",
            evidence=evidence,
        )
    if scenario_evidence.decision_path_exercised is False:
        return _blocked(
            proof_id,
            "Scenario did not exercise canonical Decision runtime path",
            evidence=evidence,
        )
    if scenario_evidence.used_mock_provider:
        return _blocked(
            proof_id,
            "Production qualification requires canonical real provider path",
            evidence=evidence,
        )
    merged = (scenario_execution_evidence_ref(scenario_evidence),) + evidence
    return DecisionE2EQualificationResult(
        proof_id=proof_id,
        disposition=QualificationDisposition.PASSED,
        evidence=merged,
        reason=reason,
    )


def validate_qualification_result(
    result: DecisionE2EQualificationResult,
) -> DecisionE2EQualificationResult:
    """Reject false PASSED rows for hardened proof contracts."""
    requirement = PROOF_REQUIREMENTS.get(result.proof_id)
    if requirement is None or result.disposition is not QualificationDisposition.PASSED:
        return result

    evidence_kinds = frozenset(item.kind for item in result.evidence)

    if requirement.requires_distinct_models:
        if result.proof_id is DecisionE2EProofId.DS_E2E_02:
            bindings = tuple(
                _binding_from_evidence(item)
                for item in result.evidence
                if item.kind == "provider_binding"
            )
            qualifies, block_reason = council_requires_distinct_models(bindings)
            if not qualifies:
                return _blocked(result.proof_id, block_reason or "model independence missing")
        if result.proof_id is DecisionE2EProofId.DS_E2E_03:
            bindings = tuple(
                _binding_from_evidence(item)
                for item in result.evidence
                if item.kind == "provider_binding"
            )
            if len(bindings) < 2:
                return _blocked(
                    result.proof_id,
                    "Producer and verifier provider bindings required for PASSED",
                )
            qualifies, block_reason = producer_verifier_requires_distinct_models(
                bindings[0],
                bindings[1],
            )
            if not qualifies:
                return _blocked(result.proof_id, block_reason or "model independence missing")

    if requirement.requires_docker_kill:
        if "docker_crash" not in evidence_kinds:
            return _blocked(
                result.proof_id,
                "DS-E2E-06 PASSED requires docker_crash evidence kind",
            )

    if requirement.requires_live_scenario:
        scenario_refs = tuple(
            item for item in result.evidence if item.kind == "scenario_execution"
        )
        if result.proof_id is DecisionE2EProofId.DS_E2E_13 and len(scenario_refs) < 2:
            return _blocked(
                result.proof_id,
                "DS-E2E-13 PASSED requires two live scenario execution evidence refs",
            )
        if not scenario_refs:
            return _blocked(
                result.proof_id,
                f"{result.proof_id.value} PASSED requires scenario_execution evidence kind",
            )

    return result


def _binding_from_evidence(item: QualificationEvidenceRef) -> ProviderBindingEvidence:
    provider = "unknown"
    model: str | None = None
    detail = item.detail or ""
    for segment in detail.split(";"):
        if segment.startswith("provider="):
            provider = segment.removeprefix("provider=")
        if segment.startswith("model="):
            model = segment.removeprefix("model=")
    return ProviderBindingEvidence(
        profile_id=item.ref,
        provider=provider,
        model=model,
    )
